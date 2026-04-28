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
