import json

from llm_client import call_llm
from prompts import (
    RECIPE_SYSTEM,
    RECIPE_SYSTEM_AE,
    RECIPE_SYSTEM_DEMOG,
    RECIPE_SYSTEM_RESPONSE,
)
from table_classifier import route_table


PACKAGE_BLOCK = """# Auto-install required packages
local_lib <- path.expand("~/R/library")
dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
locks <- list.files(local_lib, pattern = "^00LOCK-", full.names = TRUE)
unlink(locks, recursive = TRUE)
.libPaths(c(local_lib, .libPaths()))

pkgs <- c("Tplyr", "dplyr", "haven", "stringr", "tidyr")
for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", lib = local_lib,
      dependencies = c("Depends", "Imports", "LinkingTo"), INSTALL_opts = "--no-lock", Ncpus = 1L)
  }
}
library(Tplyr); library(dplyr); library(haven); library(stringr); library(tidyr)
"""


def _strip_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json\n"):
            raw = raw[5:]
    return raw.strip()


def generate_recipe(
    table_json: dict,
    adam_specs: dict | None,
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.2,
    routing_mode: str = "heuristic",
    classifier_votes: int = 3,
) -> dict:
    cls = route_table(
        table_json=table_json,
        adam_specs=adam_specs,
        mode=routing_mode,
        provider=provider,
        model=model,
        api_key=api_key,
        votes=classifier_votes,
    )
    route = cls.table_type
    system_prompt = RECIPE_SYSTEM
    if route == "ae":
        system_prompt = RECIPE_SYSTEM_AE
    elif route == "demog":
        system_prompt = RECIPE_SYSTEM_DEMOG
    elif route == "response":
        system_prompt = RECIPE_SYSTEM_RESPONSE

    starter_recipe = build_deterministic_recipe(table_json, adam_specs, route=route)
    prompt = (
        f"Required route: {route}\n\n"
        "Use the starter recipe as the structural contract. You may adjust filters, stats, and layers only when the input proves it.\n"
        "Do not return an empty tables array.\n\n"
        f"Starter recipe JSON:\n{json.dumps(starter_recipe, indent=2)}\n\n"
        f"Table JSON:\n{json.dumps(table_json, indent=2)}\n\n"
        f"AdaM Specs:\n{json.dumps(adam_specs or {}, indent=2)}"
    )
    raw = call_llm(
        system=system_prompt,
        user=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=4000,
        temperature=temperature,
        json_mode=True,
    )
    recipe = json.loads(_strip_fences(raw))
    used_empty_tables_retry = False
    if not isinstance(recipe.get("tables"), list) or not recipe.get("tables"):
        used_empty_tables_retry = True
        retry_prompt = (
            "Your previous recipe was invalid because `tables` was missing or empty.\n"
            "Return a full corrected recipe JSON now. Start from this valid starter recipe and adapt only if necessary.\n\n"
            f"Starter recipe JSON:\n{json.dumps(starter_recipe, indent=2)}\n\n"
            f"Previous response JSON:\n{json.dumps(recipe, indent=2)}\n\n"
            f"Table JSON:\n{json.dumps(table_json, indent=2)}\n\n"
            f"AdaM Specs:\n{json.dumps(adam_specs or {}, indent=2)}"
        )
        raw = call_llm(
            system=system_prompt,
            user=retry_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=4000,
            temperature=0.0,
            json_mode=True,
        )
        recipe = json.loads(_strip_fences(raw))
    recipe["_generation"] = {
        "used_starter_recipe": True,
        "used_empty_tables_retry": used_empty_tables_retry,
    }
    recipe["_routing"] = {
        "table_type": cls.table_type,
        "confidence": cls.confidence,
        "rationale": cls.rationale,
        "source": cls.source,
        "mode": routing_mode,
        "classifier_votes": classifier_votes,
    }
    return recipe


def build_deterministic_recipe(table_json: dict, adam_specs: dict | None = None, route: str | None = None) -> dict:
    """Construct a conservative recipe when the LLM returns an unusable skeleton."""
    meta = table_json.get("table_metadata", {}) or {}
    rows = table_json.get("rows", []) or []
    cols = table_json.get("columns", []) or []
    ds = (meta.get("dataset_source") or (adam_specs or {}).get("dataset") or "dataset").lower()
    treatment_var = (adam_specs or {}).get("treatment_variable") or _guess_treatment_var(ds)
    add_total = any((c.get("type") or "").lower() == "total" for c in cols if isinstance(c, dict))
    route = route or route_table(table_json, adam_specs, mode="heuristic").table_type

    pre_filters = _recipe_filters(table_json, adam_specs)
    derived_vars: list[dict] = []
    layers: list[dict] = []

    if route == "ae":
        if any((r.get("row_type") or "") == "subject_count" for r in rows if isinstance(r, dict)):
            derived_vars.append({"dataset_var": ds, "name": "ANY_EVENT", "expr": "'Yes'"})
            layers.append(
                {
                    "type": "group_count",
                    "var": "ANY_EVENT",
                    "nested_var": None,
                    "by_var": None,
                    "distinct_by": "USUBJID",
                }
            )
        layers.append(
            {
                "type": "group_count",
                "var": "AEBODSYS",
                "nested_var": "AEDECOD",
                "by_var": None,
                "distinct_by": "USUBJID",
            }
        )
    elif route == "response":
        if "PARAMCD == 'BOR'" not in pre_filters and _has_var(adam_specs, "PARAMCD"):
            pre_filters.append("PARAMCD == 'BOR'")
        distinct = "USUBJID" if _has_var(adam_specs, "USUBJID") else None
        layers.append(
            {
                "type": "group_count",
                "var": "AVALC",
                "nested_var": None,
                "by_var": None,
                "distinct_by": distinct,
            }
        )
        labels = " ".join((r.get("label") or "").upper() for r in rows if isinstance(r, dict))
        if "ORR" in labels:
            derived_vars.append(
                {"dataset_var": ds, "name": "ORR_FLAG", "expr": "ifelse(AVALC %in% c('CR','PR'), 'Yes', 'No')"}
            )
            layers.append({"type": "group_count", "var": "ORR_FLAG", "nested_var": None, "by_var": None, "distinct_by": distinct})
        if "DCR" in labels:
            derived_vars.append(
                {"dataset_var": ds, "name": "DCR_FLAG", "expr": "ifelse(AVALC %in% c('CR','PR','SD'), 'Yes', 'No')"}
            )
            layers.append({"type": "group_count", "var": "DCR_FLAG", "nested_var": None, "by_var": None, "distinct_by": distinct})
    elif route == "survival":
        return {
            "approach": "survival",
            "dataset_var": ds,
            "pre_filters": pre_filters,
            "derived_vars": [],
            "tables": [
                {
                    "table_var": "t1",
                    "dataset_var": ds,
                    "treatment_var": treatment_var,
                    "add_total": False,
                    "layers": [{"type": "group_desc", "var": "AVAL", "nested_var": None, "by_var": None, "distinct_by": None, "stats": ["median"]}],
                }
            ],
            "combine_method": "bind_rows",
            "_source": "deterministic_fallback",
        }
    else:
        seen: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            var = r.get("analysis_var")
            if not var or var in seen or var.endswith("FL"):
                continue
            row_type = r.get("row_type")
            if row_type == "continuous":
                layers.append(
                    {
                        "type": "group_desc",
                        "var": var,
                        "nested_var": None,
                        "by_var": None,
                        "distinct_by": None,
                        "stats": _normalise_stats(r.get("stats") or []),
                    }
                )
                seen.add(var)
            elif row_type == "category":
                layers.append({"type": "group_count", "var": var, "nested_var": None, "by_var": None, "distinct_by": r.get("distinct_by")})
                seen.add(var)

    return {
        "approach": "tplyr",
        "dataset_var": ds,
        "pre_filters": pre_filters,
        "derived_vars": derived_vars,
        "tables": [
            {
                "table_var": "t1",
                "dataset_var": ds,
                "treatment_var": treatment_var,
                "add_total": add_total,
                "layers": layers,
            }
        ],
        "combine_method": "bind_rows",
        "_source": "deterministic_fallback",
    }


def _guess_treatment_var(dataset_var: str) -> str:
    return "TRT01P" if dataset_var.lower() == "adsl" else "TRTP"


def _has_var(adam_specs: dict | None, var: str) -> bool:
    if not adam_specs:
        return True
    return any((kv or {}).get("variable") == var for kv in adam_specs.get("key_variables", []) or [])


def _recipe_filters(table_json: dict, adam_specs: dict | None) -> list[str]:
    filters: list[str] = []
    flags = list((table_json.get("table_metadata", {}) or {}).get("population_flags", []) or [])
    if adam_specs:
        flags += [(pf or {}).get("variable") for pf in adam_specs.get("population_flags", []) or []]
    for flag in flags:
        if flag and f"{flag} == 'Y'" not in filters:
            filters.append(f"{flag} == 'Y'")
    if adam_specs:
        for cond in adam_specs.get("analysis_conditions", []) or []:
            anl = (cond or {}).get("anl_flag")
            param = (cond or {}).get("paramcd_filter")
            if anl:
                expr = anl.replace("=", "==").replace("===", "==").replace('"', "'")
                if expr not in filters:
                    filters.append(expr)
            if param:
                expr = f"PARAMCD == '{param}'"
                if expr not in filters:
                    filters.append(expr)
    return filters


def _normalise_stats(stats: list[str]) -> list[str]:
    out: list[str] = []
    joined = " ".join(stats).lower()
    if "n" in joined:
        out.append("n")
    if "mean" in joined:
        out.append("mean")
    if "sd" in joined:
        out.append("sd")
    if "median" in joined:
        out.append("median")
    if "min" in joined:
        out.append("min")
    if "max" in joined:
        out.append("max")
    return out or ["n", "mean", "sd", "median", "min", "max"]


def assemble_r_from_recipe(recipe: dict) -> str:
    dataset_var = recipe.get("dataset_var", "dataset")
    lines = [PACKAGE_BLOCK, ""]
    lines += [
        "ext <- tolower(tools::file_ext(data_path))",
        'if (ext == "sas7bdat") {',
        f"  {dataset_var} <- haven::read_sas(data_path)",
        '} else if (ext == "csv") {',
        f"  {dataset_var} <- read.csv(data_path, stringsAsFactors = FALSE)",
        "} else {",
        "  env <- new.env()",
        "  load(data_path, envir = env)",
        f"  {dataset_var} <- get(ls(env)[1], envir = env)",
        "}",
        "",
    ]

    filters = recipe.get("pre_filters", []) or []
    if filters:
        lines.append(f"{dataset_var} <- {dataset_var} %>% filter({', '.join(filters)})")
        lines.append("")

    for dv in recipe.get("derived_vars", []) or []:
        ds = dv.get("dataset_var", dataset_var)
        nm = dv.get("name")
        ex = dv.get("expr")
        if nm and ex:
            lines.append(f"{ds} <- {ds} %>% mutate({nm} = {ex})")
    if recipe.get("derived_vars"):
        lines.append("")

    result_vars: list[str] = []
    for i, tbl in enumerate(recipe.get("tables", []), start=1):
        tvar = tbl.get("table_var", f"t{i}")
        out = f"{tvar}_df"
        result_vars.append(out)
        ds = tbl.get("dataset_var", dataset_var)
        trt = tbl.get("treatment_var", "TRTP")
        parts = [f"tplyr_table({ds}, {trt})"]
        if tbl.get("add_total"):
            parts.append("  add_total_group()")
        for layer in tbl.get("layers", []):
            parts.append(_assemble_layer(layer))
        lines.append(f"{tvar} <- " + " %>%\n".join(parts))
        lines.append(f"{out} <- {tvar} %>% build()")
        lines.append("")

    combine = recipe.get("combine_method", "bind_rows")
    if not result_vars:
        lines.append("final_df <- data.frame()")
    elif len(result_vars) == 1:
        lines.append(f"final_df <- {result_vars[0]}")
    else:
        lines.append(f"final_df <- {combine}({', '.join(result_vars)})")

    return "\n".join(lines)


def _assemble_layer(layer: dict) -> str:
    typ = layer.get("type", "group_count")
    var = layer.get("var")
    nested = layer.get("nested_var")
    by_var = layer.get("by_var")
    distinct = layer.get("distinct_by")
    stats = layer.get("stats", []) or []

    var_expr = f"vars({var}, {nested})" if nested else f"{var}"
    if by_var:
        var_expr = f"{var_expr}, by = {by_var}"

    if typ == "group_desc":
        fmt = []
        if "n" in stats:
            fmt.append('"n" = f_str("xx", n)')
        if "mean" in stats and "sd" in stats:
            fmt.append('"Mean (SD)" = f_str("xx.x (xx.xx)", mean, sd)')
        if "median" in stats:
            fmt.append('"Median" = f_str("xx.x", median)')
        if "min" in stats and "max" in stats:
            fmt.append('"Min, Max" = f_str("xx, xx", min, max)')
        fmt_block = " %>%\n      set_format_strings(" + ", ".join(fmt) + ")" if fmt else ""
        return f"  add_layer(\n    group_desc({var_expr}){fmt_block}\n  )"

    count_block = (
        f"  add_layer(\n"
        f"    group_count({var_expr}) %>%\n"
        f'      set_format_strings("n (%)" = f_str("xx (xx.x%)", n, pct))'
    )
    if distinct:
        count_block += f" %>%\n      set_distinct_by({distinct})"
    count_block += "\n  )"
    return count_block


def _r_expr_to_sas(expr: str) -> str:
    import re

    s = expr or ""
    s = s.replace("==", "=").replace("!=", "^=")
    s = re.sub(r"\s&&?\s", " and ", s)
    s = re.sub(r"\s\|\|?\s", " or ", s)
    s = re.sub(r"\s%in%\s", " in ", s)
    s = re.sub(r"\bTRUE\b", "1", s)
    s = re.sub(r"\bFALSE\b", "0", s)
    return s


def assemble_sas_from_recipe(recipe: dict) -> str:
    dataset_var = (recipe.get("dataset_var") or "dataset").lower()
    approach = recipe.get("approach", "tplyr")
    lines = [
        "/* Auto-generated SAS program from recipe */",
        "%let data_path = /path/to/your/dataset.sas7bdat;  /* override */",
        "",
        "%macro load_data;",
        "  %let _ext = %lowcase(%scan(&data_path, -1, .));",
        "  %if &_ext = sas7bdat %then %do;",
        "    %let _dir  = %substr(&data_path, 1, %eval(%length(&data_path) - %length(%scan(&data_path, -1, /\\)) - 1));",
        "    %let _file = %scan(%scan(&data_path, -1, /\\), 1, .);",
        "    libname _in \"&_dir\";",
        f"    data {dataset_var};",
        "      set _in.&_file;",
        "    run;",
        "  %end;",
        "  %else %if &_ext = csv %then %do;",
        f"    proc import datafile=\"&data_path\" out={dataset_var} dbms=csv replace;",
        "      getnames=yes;",
        "    run;",
        "  %end;",
        "%mend load_data;",
        "%load_data;",
        "",
    ]

    filters = recipe.get("pre_filters", []) or []
    derived = recipe.get("derived_vars", []) or []
    if filters or derived:
        lines.append(f"data {dataset_var};")
        lines.append(f"  set {dataset_var};")
        for dv in derived:
            nm = dv.get("name", "")
            ex = _r_expr_to_sas(dv.get("expr", ""))
            if nm:
                lines.append(f"  {nm} = {ex};")
        if filters:
            where_expr = " and ".join(f"({_r_expr_to_sas(f)})" for f in filters)
            lines.append(f"  where {where_expr};")
        lines.append("run;")
        lines.append("")

    if approach == "survival":
        trt = "TRTP"
        if recipe.get("tables"):
            trt = recipe["tables"][0].get("treatment_var", "TRTP")
        lines += [
            f"proc lifetest data={dataset_var} notable outsurv=_km;",
            "  time AVAL * CNSR(1);",
            f"  strata {trt};",
            "run;",
            "",
            "data final_df;",
            "  set _km;",
            "run;",
            "",
            "proc print data=final_df(obs=50); run;",
        ]
        return "\n".join(lines)

    out_datasets: list[str] = []
    for tbl in recipe.get("tables", []) or []:
        tvar = (tbl.get("table_var") or "t1").lower()
        work_ds = (tbl.get("dataset_var") or dataset_var).lower()
        trt = tbl.get("treatment_var", "TRTP")
        add_total = bool(tbl.get("add_total"))
        layers = tbl.get("layers", []) or []

        if add_total:
            total_ds = f"{tvar}_in"
            lines += [
                f"data {total_ds};",
                f"  set {work_ds} {work_ds}(in=_a);",
                f"  if _a then {trt} = 'Total';",
                "run;",
                "",
            ]
            work_ds = total_ds

        for li, layer in enumerate(layers, start=1):
            ltype = layer.get("type", "group_count")
            var = layer.get("var", "")
            nested = layer.get("nested_var")
            distinct = layer.get("distinct_by")
            stats = layer.get("stats", []) or []
            out_ds = f"{tvar}_l{li}"
            if not var:
                continue

            if ltype == "group_desc":
                stat_kw = []
                if "n" in stats:
                    stat_kw.append("n")
                if "mean" in stats:
                    stat_kw.append("mean")
                if "sd" in stats:
                    stat_kw.append("std")
                if "median" in stats:
                    stat_kw.append("median")
                if "min" in stats:
                    stat_kw.append("min")
                if "max" in stats:
                    stat_kw.append("max")
                if not stat_kw:
                    stat_kw = ["n", "mean", "std", "median", "min", "max"]
                output_kw = " ".join(f"{k}={var}_{k}" for k in stat_kw)
                lines += [
                    f"proc means data={work_ds} noprint nway;",
                    f"  class {trt};",
                    f"  var {var};",
                    f"  output out={out_ds}(drop=_type_ _freq_) {output_kw};",
                    "run;",
                    "",
                ]
                out_datasets.append(out_ds)
                continue

            tables_spec = f"{trt} * {var} * {nested}" if nested else f"{trt} * {var}"
            if distinct:
                dedup_ds = f"{out_ds}_u"
                keep_vars = [trt, var] + ([nested] if nested else []) + [distinct]
                lines += [
                    f"proc sort data={work_ds}(keep={' '.join(keep_vars)}) out={dedup_ds} nodupkey;",
                    f"  by {' '.join(keep_vars)};",
                    "run;",
                    "",
                    f"proc freq data={dedup_ds} noprint;",
                    f"  tables {tables_spec} / out={out_ds}(drop=percent) outpct;",
                    "run;",
                    "",
                ]
            else:
                lines += [
                    f"proc freq data={work_ds} noprint;",
                    f"  tables {tables_spec} / out={out_ds}(drop=percent) outpct;",
                    "run;",
                    "",
                ]
            out_datasets.append(out_ds)

    if not out_datasets:
        lines += ["data final_df;", "  stop;", "run;"]
    else:
        lines += [
            "data final_df;",
            "  set " + " ".join(out_datasets) + ";",
            "run;",
            "",
            "proc print data=final_df(obs=50); run;",
        ]
    return "\n".join(lines)
