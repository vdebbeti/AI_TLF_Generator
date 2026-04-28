import json
import re
from dataclasses import dataclass, asdict
from typing import Callable

from llm_client import call_llm
from prompts import JSON_REPAIR_SYSTEM


FLAG_VARS = {"SAFFL", "FASFL", "TRTEMFL", "ANL01FL", "ITTFL"}


@dataclass
class ValidationIssue:
    code: str
    message: str
    path: str
    severity: str = "ERROR"


def issue_dicts(issues: list[ValidationIssue]) -> list[dict]:
    return [asdict(i) for i in issues]


def _is_var_name(s: str | None) -> bool:
    return bool(s) and bool(re.match(r"^[A-Za-z][A-Za-z0-9_]*$", s))


def _is_ae_table(table_json: dict) -> bool:
    meta = table_json.get("table_metadata") or {}
    ds = (meta.get("dataset_source") or "").upper()
    title = (meta.get("title") or "").lower()
    return ds == "ADAE" or "adverse event" in title or "teae" in title


def validate_table_json(table_json: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(table_json, dict):
        return [ValidationIssue("schema.root", "Top-level table JSON must be an object.", "$")]

    meta = table_json.get("table_metadata")
    cols = table_json.get("columns")
    rows = table_json.get("rows")
    if not isinstance(meta, dict):
        issues.append(ValidationIssue("schema.table_metadata", "table_metadata must be an object.", "$.table_metadata"))
    if not isinstance(cols, list) or not cols:
        issues.append(ValidationIssue("schema.columns", "columns must be a non-empty list.", "$.columns"))
    if not isinstance(rows, list) or not rows:
        issues.append(ValidationIssue("schema.rows", "rows must be a non-empty list.", "$.rows"))
        return issues

    for i, r in enumerate(rows):
        p = f"$.rows[{i}]"
        if not isinstance(r, dict):
            issues.append(ValidationIssue("schema.row", "Each row must be an object.", p))
            continue
        if not r.get("label"):
            issues.append(ValidationIssue("schema.row.label", "Row label is required.", f"{p}.label"))
        if "indent_level" in r and not isinstance(r.get("indent_level"), int):
            issues.append(ValidationIssue("schema.row.indent", "indent_level must be integer.", f"{p}.indent_level"))
        av = r.get("analysis_var")
        if av and not _is_var_name(av):
            issues.append(ValidationIssue("semantics.analysis_var", "analysis_var must look like a variable name.", f"{p}.analysis_var"))
        if av in FLAG_VARS and r.get("row_type") != "subject_count":
            issues.append(
                ValidationIssue(
                    "semantics.flag_as_analysis",
                    f"{av} looks like a flag variable and should not be used as analysis_var.",
                    f"{p}.analysis_var",
                )
            )

    if _is_ae_table(table_json):
        has_soc = any((r.get("analysis_var") or "").upper() == "AEBODSYS" for r in rows if isinstance(r, dict))
        has_pt = any((r.get("analysis_var") or "").upper() == "AEDECOD" for r in rows if isinstance(r, dict))
        if not has_soc:
            issues.append(ValidationIssue("ae.missing_soc", "AE table should include SOC rows (AEBODSYS).", "$.rows"))
        if not has_pt:
            issues.append(ValidationIssue("ae.missing_pt", "AE table should include PT rows (AEDECOD).", "$.rows"))
    return issues


def validate_adam_specs(adam_specs: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(adam_specs, dict):
        return [ValidationIssue("schema.root", "AdaM specs must be an object.", "$")]
    if not isinstance(adam_specs.get("key_variables", []), list):
        issues.append(ValidationIssue("schema.key_variables", "key_variables must be a list.", "$.key_variables"))
    tv = adam_specs.get("treatment_variable")
    if tv and not _is_var_name(tv):
        issues.append(ValidationIssue("semantics.treatment_variable", "treatment_variable must be a variable name.", "$.treatment_variable"))
    return issues


def collect_allowed_vars(table_json: dict, adam_specs: dict | None) -> set[str]:
    allowed: set[str] = set()
    for r in table_json.get("rows", []):
        av = (r.get("analysis_var") or "").strip()
        if _is_var_name(av):
            allowed.add(av)
        db = (r.get("distinct_by") or "").strip()
        if _is_var_name(db):
            allowed.add(db)
    meta_flags = table_json.get("table_metadata", {}).get("population_flags", []) or []
    allowed.update([f for f in meta_flags if _is_var_name(f)])

    if adam_specs:
        for kv in adam_specs.get("key_variables", []):
            v = (kv or {}).get("variable")
            if _is_var_name(v):
                allowed.add(v)
        tv = adam_specs.get("treatment_variable")
        if _is_var_name(tv):
            allowed.add(tv)
        for pf in adam_specs.get("population_flags", []):
            v = (pf or {}).get("variable")
            if _is_var_name(v):
                allowed.add(v)
    return allowed


def validate_recipe(recipe: dict, table_json: dict, adam_specs: dict | None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(recipe, dict):
        return [ValidationIssue("schema.root", "Recipe must be an object.", "$")]
    if not isinstance(recipe.get("tables"), list) or not recipe.get("tables"):
        issues.append(ValidationIssue("schema.tables", "Recipe must contain at least one table.", "$.tables"))
        return issues

    allowed_vars = collect_allowed_vars(table_json, adam_specs)
    for dv in recipe.get("derived_vars", []) or []:
        name = (dv or {}).get("name")
        if _is_var_name(name):
            allowed_vars.add(name)
    ae_required = _is_ae_table(table_json)
    saw_nested_soc_pt = False

    for ti, t in enumerate(recipe.get("tables", [])):
        tp = f"$.tables[{ti}]"
        tv = t.get("treatment_var")
        if tv and tv not in allowed_vars:
            issues.append(ValidationIssue("semantics.treatment_var", f"Unknown treatment variable: {tv}", f"{tp}.treatment_var"))
        layers = t.get("layers", [])
        if not isinstance(layers, list) or not layers:
            issues.append(ValidationIssue("schema.layers", "Each table needs layers.", f"{tp}.layers"))
            continue
        for li, layer in enumerate(layers):
            lp = f"{tp}.layers[{li}]"
            var = layer.get("var")
            nested = layer.get("nested_var")
            by_var = layer.get("by_var")
            distinct = layer.get("distinct_by")
            for field_name, field_val in [("var", var), ("nested_var", nested), ("by_var", by_var), ("distinct_by", distinct)]:
                if field_val is None:
                    continue
                if not _is_var_name(field_val):
                    issues.append(ValidationIssue("semantics.var_name", f"{field_name} is not a valid variable name: {field_val}", f"{lp}.{field_name}"))
                elif field_val not in allowed_vars and not field_val.endswith("_FLAG"):
                    issues.append(ValidationIssue("semantics.unknown_var", f"{field_name} not in known variable set: {field_val}", f"{lp}.{field_name}"))
                if field_val in FLAG_VARS and field_name in {"var", "nested_var", "by_var"}:
                    issues.append(ValidationIssue("semantics.flag_misuse", f"{field_val} should be a filter, not {field_name}.", f"{lp}.{field_name}"))
            if (var or "").upper() == "AEBODSYS" and (nested or "").upper() == "AEDECOD":
                saw_nested_soc_pt = True

    if ae_required and not saw_nested_soc_pt:
        issues.append(
            ValidationIssue(
                "ae.recipe_nested_required",
                "AE recipe must include one nested SOC/PT layer: var=AEBODSYS, nested_var=AEDECOD.",
                "$.tables[*].layers[*]",
            )
        )
    return issues


def _strip_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json\n"):
            raw = raw[5:]
    return raw.strip()


def repair_json_with_llm(
    kind: str,
    candidate: dict,
    issues: list[ValidationIssue],
    context: dict,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    user_msg = (
        f"Target kind: {kind}\n\n"
        f"Validation issues:\n{json.dumps(issue_dicts(issues), indent=2)}\n\n"
        f"Current JSON:\n{json.dumps(candidate, indent=2)}\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
        "Return corrected JSON only."
    )
    raw = call_llm(
        system=JSON_REPAIR_SYSTEM,
        user=user_msg,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=3500,
        temperature=0.1,
        json_mode=True,
    )
    return json.loads(_strip_fences(raw))


def validate_and_repair(
    kind: str,
    candidate: dict,
    validator: Callable[[dict], list[ValidationIssue]],
    context: dict,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 2,
) -> tuple[dict, list[ValidationIssue], int]:
    repaired = candidate
    issues = validator(repaired)
    retries = 0
    while issues and retries < max_retries:
        repaired = repair_json_with_llm(kind, repaired, issues, context, provider, model, api_key)
        issues = validator(repaired)
        retries += 1
    return repaired, issues, retries
