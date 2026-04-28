import json
import traceback
from pathlib import Path

from guardrails import (
    validate_recipe,
    validate_table_json,
    validate_adam_specs,
    validate_and_repair,
)
from orchestrator import build_deterministic_recipe, generate_recipe
from table_classifier import route_table


def load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for p in sorted(cases_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            data["_path"] = str(p)
            cases.append(data)
    return cases


def run_suite(
    cases_dir: Path,
    provider: str,
    model: str,
    api_key: str,
    recipe_temperature: float = 0.2,
    run_llm_recipe: bool = True,
    repair_retries: int = 2,
    routing_mode: str = "heuristic",
    classifier_votes: int = 3,
    benchmark_routing: bool = True,
    event_logger=None,
    run_id: str | None = None,
) -> dict:
    cases = load_cases(cases_dir)
    results = []
    for case in cases:
        case_id = case.get("id", "unknown")
        if event_logger:
            event_logger(
                "eval_case_started",
                "INFO",
                {"run_id": run_id, "case_id": case_id},
            )
        table_json = case.get("table_json", {})
        adam_specs = case.get("adam_specs")
        table_issues = validate_table_json(table_json)
        adam_issues = validate_adam_specs(adam_specs) if adam_specs else []

        recipe = None
        recipe_issues = []
        recipe_issues_after_repair = []
        recipe_repair_retries = 0
        llm_error = ""
        cls = route_table(
            table_json=table_json,
            adam_specs=adam_specs,
            mode=routing_mode,
            provider=provider,
            model=model,
            api_key=api_key,
            votes=classifier_votes,
        )
        expected_type = (case.get("expected_table_type") or "").lower().strip() or None

        metrics = {
            "table_type": cls.table_type,
            "route_confidence": cls.confidence,
            "route_source": cls.source,
            "expected_table_type": expected_type,
            "route_correct": None,
            "ae_soc_pt_nested_present": None,
            "unknown_var_issue_count": 0,
            "flag_misuse_issue_count": 0,
            "pre_repair_recipe_issue_count": 0,
            "post_repair_recipe_issue_count": 0,
            "used_deterministic_fallback": False,
            "fallback_issue_count": 0,
            "llm_used_empty_tables_retry": False,
        }
        if benchmark_routing and expected_type:
            metrics["route_correct"] = cls.table_type == expected_type
        if run_llm_recipe and not table_issues:
            try:
                recipe = generate_recipe(
                    table_json=table_json,
                    adam_specs=adam_specs,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=recipe_temperature,
                    routing_mode=routing_mode,
                    classifier_votes=classifier_votes,
                )
                metrics["llm_used_empty_tables_retry"] = bool(
                    (recipe.get("_generation") or {}).get("used_empty_tables_retry")
                )
                recipe_issues = validate_recipe(recipe, table_json, adam_specs)
                metrics["pre_repair_recipe_issue_count"] = len(recipe_issues)

                repaired_recipe, recipe_issues_after_repair, recipe_repair_retries = validate_and_repair(
                    kind="recipe_json",
                    candidate=recipe,
                    validator=lambda r: validate_recipe(r, table_json, adam_specs),
                    context={"table_json": table_json, "adam_specs": adam_specs or {}},
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    max_retries=repair_retries,
                )
                recipe = repaired_recipe
                metrics["post_repair_recipe_issue_count"] = len(recipe_issues_after_repair)
                if recipe_issues_after_repair:
                    fallback_recipe = build_deterministic_recipe(table_json, adam_specs, route=cls.table_type)
                    fallback_issues = validate_recipe(fallback_recipe, table_json, adam_specs)
                    if not fallback_issues:
                        recipe = fallback_recipe
                        recipe_issues_after_repair = []
                        metrics["used_deterministic_fallback"] = True
                        metrics["post_repair_recipe_issue_count"] = 0
                    else:
                        metrics["used_deterministic_fallback"] = True
                        metrics["fallback_issue_count"] = len(fallback_issues)

                all_issues = recipe_issues + recipe_issues_after_repair
                metrics["unknown_var_issue_count"] = sum(1 for i in all_issues if i.code == "semantics.unknown_var")
                metrics["flag_misuse_issue_count"] = sum(1 for i in all_issues if i.code == "semantics.flag_misuse")

                if cls.table_type == "ae":
                    has_nested = False
                    for t in recipe.get("tables", []):
                        for l in t.get("layers", []):
                            if (l.get("var") or "").upper() == "AEBODSYS" and (l.get("nested_var") or "").upper() == "AEDECOD":
                                has_nested = True
                                break
                    metrics["ae_soc_pt_nested_present"] = has_nested
            except Exception as e:
                llm_error = str(e)
                metrics["ae_soc_pt_nested_present"] = False if cls.table_type == "ae" else None
                if event_logger:
                    event_logger(
                        "eval_case_failed",
                        "ERROR",
                        {
                            "run_id": run_id,
                            "case_id": case_id,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

        final_recipe_issues = recipe_issues_after_repair if run_llm_recipe else recipe_issues
        passed = (not table_issues) and (not adam_issues) and (not final_recipe_issues) and (not llm_error)
        row = (
            {
                "id": case_id,
                "description": case.get("description", ""),
                "path": case.get("_path"),
                "passed": passed,
                "table_issue_count": len(table_issues),
                "adam_issue_count": len(adam_issues),
                "recipe_issue_count": len(final_recipe_issues),
                "llm_error": llm_error,
                "table_issues": [i.__dict__ for i in table_issues],
                "adam_issues": [i.__dict__ for i in adam_issues],
                "recipe_issues": [i.__dict__ for i in final_recipe_issues],
                "recipe_issues_before_repair": [i.__dict__ for i in recipe_issues],
                "recipe_repair_retries": recipe_repair_retries,
                "metrics": metrics,
                "recipe": recipe,
            }
        )
        results.append(row)
        if event_logger:
            event_logger(
                "eval_case_completed",
                "SUCCESS" if passed else "WARNING",
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "passed": passed,
                    "route": metrics.get("table_type"),
                    "expected_route": metrics.get("expected_table_type"),
                    "route_correct": metrics.get("route_correct"),
                    "pre_repair_recipe_issue_count": metrics.get("pre_repair_recipe_issue_count"),
                    "post_repair_recipe_issue_count": metrics.get("post_repair_recipe_issue_count"),
                    "recipe_repair_retries": row.get("recipe_repair_retries"),
                    "used_deterministic_fallback": metrics.get("used_deterministic_fallback"),
                    "fallback_issue_count": metrics.get("fallback_issue_count"),
                    "llm_used_empty_tables_retry": metrics.get("llm_used_empty_tables_retry"),
                    "llm_error": row.get("llm_error"),
                },
            )

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    ae_cases = [r for r in results if r["metrics"]["table_type"] == "ae"]
    ae_scored = [r for r in ae_cases if r["metrics"]["ae_soc_pt_nested_present"] is not None]
    ae_nested_hits = sum(1 for r in ae_scored if r["metrics"]["ae_soc_pt_nested_present"] is True)
    unknown_var_total = sum(r["metrics"]["unknown_var_issue_count"] for r in results)
    flag_misuse_total = sum(r["metrics"]["flag_misuse_issue_count"] for r in results)
    routed_scored = [r for r in results if r["metrics"]["route_correct"] is not None]
    route_hits = sum(1 for r in routed_scored if r["metrics"]["route_correct"] is True)
    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": (passed / total * 100.0) if total else 0.0,
        "ae_nested_accuracy": (ae_nested_hits / len(ae_scored) * 100.0) if ae_scored else None,
        "routing_accuracy": (route_hits / len(routed_scored) * 100.0) if routed_scored else None,
        "routing_scored_cases": len(routed_scored),
        "unknown_var_issues_total": unknown_var_total,
        "flag_misuse_issues_total": flag_misuse_total,
        "results": results,
    }
    if event_logger:
        event_logger(
            "eval_run_summary",
            "INFO",
            {
                "run_id": run_id,
                "total_cases": summary["total_cases"],
                "passed_cases": summary["passed_cases"],
                "pass_rate": summary["pass_rate"],
                "routing_accuracy": summary["routing_accuracy"],
            },
        )
    return summary
