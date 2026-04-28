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

    prompt = f"Table JSON:\n{json.dumps(table_json, indent=2)}\n\nAdaM Specs:\n{json.dumps(adam_specs or {}, indent=2)}"
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
    recipe["_routing"] = {
        "table_type": cls.table_type,
        "confidence": cls.confidence,
        "rationale": cls.rationale,
        "source": cls.source,
        "mode": routing_mode,
        "classifier_votes": classifier_votes,
    }
    return recipe


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
