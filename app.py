import json
from pathlib import Path
from datetime import datetime, timezone
import copy
import time
import uuid
import traceback
from dataclasses import is_dataclass, asdict

import streamlit as st

from adam_parser import parse_adam_specs
from eval_harness import run_suite
from guardrails import (
    ValidationIssue,
    issue_dicts,
    validate_and_repair,
    validate_adam_specs,
    validate_recipe,
    validate_table_json,
)
from llm_client import PROVIDER_MODELS
from orchestrator import assemble_r_from_recipe, assemble_sas_from_recipe, build_deterministic_recipe, generate_recipe
from parsers import parse_shell
from table_classifier import route_table


BASE_DIR = Path(__file__).parent
EVAL_CASES_DIR = BASE_DIR / "eval_cases"

st.set_page_config(page_title="TLF V2 Compiler", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(1200px 500px at 0% -10%, #11335a 0%, transparent 60%),
              radial-gradient(900px 400px at 100% 0%, #3b1c56 0%, transparent 55%),
              linear-gradient(160deg, #07111f 0%, #0f1f33 55%, #0a1626 100%);
}
.hero {
  border: 1px solid rgba(255,255,255,0.13);
  background: linear-gradient(145deg, rgba(68,136,255,0.18), rgba(22,36,58,0.55));
  border-radius: 18px;
  padding: 22px 24px;
  box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}
.hero h1 {
  margin: 0 0 6px 0;
  color: #f4f7ff;
  font-size: 30px;
}
.hero p {
  margin: 0;
  color: #d5def2;
}
.stepgrid {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
  margin-top: 14px;
}
.stepcard {
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
  padding: 10px 10px;
  background: rgba(255,255,255,0.04);
  color: #dce7ff;
  min-height: 80px;
}
.stepnum { font-size: 11px; letter-spacing: 0.8px; color: #93b1ff; font-weight: 700; }
.steptitle { font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 4px; }
.stepsub { font-size: 11px; color: #b3c5ee; margin-top: 3px; }
.okchip {
  display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px;
  background: rgba(46,204,113,0.2); color:#b6ffcf; border:1px solid rgba(46,204,113,0.35);
}
.warnchip {
  display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px;
  background: rgba(241,196,15,0.2); color:#ffe6a6; border:1px solid rgba(241,196,15,0.35);
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
  <h1>TLF Compiler V2</h1>
  <p>Guardrailed shell/spec parsing, validated recipe generation, and golden-case evaluation harness.</p>
  <div class="stepgrid">
    <div class="stepcard"><div class="stepnum">STEP 1</div><div class="steptitle">Configure</div><div class="stepsub">Provider, model, temperatures</div></div>
    <div class="stepcard"><div class="stepnum">STEP 2</div><div class="steptitle">Parse Shell</div><div class="stepsub">Upload shell and parse JSON</div></div>
    <div class="stepcard"><div class="stepnum">STEP 3</div><div class="steptitle">Parse AdaM</div><div class="stepsub">Upload spec and parse JSON</div></div>
    <div class="stepcard"><div class="stepnum">STEP 4</div><div class="steptitle">Guardrails</div><div class="stepsub">Validate and auto-repair</div></div>
    <div class="stepcard"><div class="stepnum">STEP 5</div><div class="steptitle">Recipe + R/SAS</div><div class="stepsub">Generate validated recipe and both programs</div></div>
    <div class="stepcard"><div class="stepnum">STEP 6</div><div class="steptitle">Eval Harness</div><div class="stepsub">Run golden cases and score</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


for k, v in {
    "table_json": None,
    "adam_specs": None,
    "recipe_json": None,
    "r_code": "",
    "sas_code": "",
    "table_issues": [],
    "adam_issues": [],
    "recipe_issues": [],
    "repair_stats": {},
    "eval_result": None,
    "session_log": [],
    "session_id": "",
    "event_seq": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _log_event(event: str, status: str = "INFO", details: dict | None = None, run_id: str | None = None) -> None:
    st.session_state.event_seq += 1
    safe_details = _make_json_safe(copy.deepcopy(details) if details is not None else {})
    st.session_state.session_log.append(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": st.session_state.session_id,
            "event_seq": st.session_state.event_seq,
            "run_id": run_id,
            "event": event,
            "status": status,
            "details": safe_details,
        }
    )


def _make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return _make_json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _make_json_safe(vars(obj))
    return str(obj)


def _session_log_text() -> str:
    lines = []
    for rec in st.session_state.session_log:
        lines.append(
            f"[{rec['ts_utc']}] {rec['status']} {rec['event']} :: {json.dumps(rec['details'], ensure_ascii=True)}"
        )
    return "\n".join(lines)


if not st.session_state.session_id:
    st.session_state.session_id = str(uuid.uuid4())
if not st.session_state.session_log:
    _log_event("session_started", "INFO", {"app": "TLF Compiler V2", "session_id": st.session_state.session_id})


def render_issues(issues: list[ValidationIssue] | list[dict], title: str) -> None:
    st.markdown(f"**{title}**")
    if not issues:
        st.markdown('<span class="okchip">No issues</span>', unsafe_allow_html=True)
        return
    st.markdown(f'<span class="warnchip">{len(issues)} issue(s)</span>', unsafe_allow_html=True)
    for i in issues:
        ii = i if isinstance(i, dict) else i.__dict__
        st.write(f"- `{ii.get('code')}` at `{ii.get('path')}`: {ii.get('message')}")


with st.sidebar:
    st.markdown("### Step 1 · Configure")
    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0)
    model = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)
    api_key = st.text_input("API key", type="password")
    parse_temperature = st.slider("Shell parse temperature", 0.0, 1.0, 0.7, 0.05)
    adam_temperature = st.slider("AdaM parse temperature", 0.0, 1.0, 0.4, 0.05)
    recipe_temperature = st.slider("Recipe temperature", 0.0, 1.0, 0.2, 0.05)
    repair_retries = st.slider("Auto-repair retries", 0, 3, 2, 1)
    routing_mode = st.selectbox("Routing mode", ["heuristic", "llm", "consensus"], index=0)
    classifier_votes = st.slider("Consensus votes", 1, 7, 3, 2, disabled=(routing_mode != "consensus"))
    st.markdown("---")
    st.markdown("### Session Log")
    st.caption("Download full in-app session events (parse, guardrails, recipe, eval).")
    st.download_button(
        "Download Log (JSON)",
        data=json.dumps(_make_json_safe(st.session_state.session_log), indent=2),
        file_name="session_log.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download Log (TXT)",
        data=_session_log_text(),
        file_name="session_log.txt",
        mime="text/plain",
        use_container_width=True,
    )
    if st.button("Clear Session Log", use_container_width=True):
        st.session_state.session_log = []
        _log_event("session_log_cleared", "INFO")
        st.rerun()


col1, col2 = st.columns(2)

with col1:
    st.markdown("## Step 2 · Parse Shell")
    shell_file = st.file_uploader("Upload shell (PNG/JPG/PDF/DOCX)", type=["png", "jpg", "jpeg", "pdf", "docx"])
    if st.button("Run Shell Parse", use_container_width=True, type="primary"):
        _log_event("shell_parse_requested", "INFO", {"has_file": bool(shell_file)})
        if not api_key:
            st.error("API key is required.")
            _log_event("shell_parse_failed", "ERROR", {"reason": "missing_api_key"})
        elif not shell_file:
            st.error("Upload a shell file first.")
            _log_event("shell_parse_failed", "ERROR", {"reason": "missing_file"})
        else:
            with st.spinner("Parsing shell..."):
                ext = shell_file.name.rsplit(".", 1)[-1]
                try:
                    parsed = parse_shell(
                        file_bytes=shell_file.read(),
                        extension=ext,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        parse_temperature=parse_temperature,
                    )
                    st.session_state.table_json = parsed
                    issues = issue_dicts(validate_table_json(parsed))
                    st.session_state.table_issues = issues
                    _log_event(
                        "shell_parse_completed",
                        "SUCCESS",
                        {"extension": ext, "issue_count": len(issues)},
                    )
                except Exception as e:
                    st.error(f"Shell parse failed: {e}")
                    _log_event("shell_parse_failed", "ERROR", {"extension": ext, "error": str(e)})
    if st.session_state.table_json:
        st.code(json.dumps(st.session_state.table_json, indent=2), language="json")
    render_issues(st.session_state.table_issues, "Shell issues")

with col2:
    st.markdown("## Step 3 · Parse AdaM Specs")
    adam_file = st.file_uploader("Upload AdaM spec (XLSX/PDF/DOCX)", type=["xlsx", "xls", "xlsm", "pdf", "docx"])
    if st.button("Run AdaM Parse", use_container_width=True, type="primary"):
        _log_event("adam_parse_requested", "INFO", {"has_file": bool(adam_file)})
        if not api_key:
            st.error("API key is required.")
            _log_event("adam_parse_failed", "ERROR", {"reason": "missing_api_key"})
        elif not adam_file:
            st.error("Upload an AdaM spec file first.")
            _log_event("adam_parse_failed", "ERROR", {"reason": "missing_file"})
        else:
            with st.spinner("Parsing AdaM specs..."):
                ext = adam_file.name.rsplit(".", 1)[-1]
                try:
                    parsed = parse_adam_specs(
                        file_bytes=adam_file.read(),
                        extension=ext,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        parse_temperature=adam_temperature,
                    )
                    st.session_state.adam_specs = parsed
                    issues = issue_dicts(validate_adam_specs(parsed))
                    st.session_state.adam_issues = issues
                    _log_event(
                        "adam_parse_completed",
                        "SUCCESS",
                        {"extension": ext, "issue_count": len(issues)},
                    )
                except Exception as e:
                    st.error(f"AdaM parse failed: {e}")
                    _log_event("adam_parse_failed", "ERROR", {"extension": ext, "error": str(e)})
    if st.session_state.adam_specs:
        st.code(json.dumps(st.session_state.adam_specs, indent=2), language="json")
    render_issues(st.session_state.adam_issues, "AdaM issues")


st.markdown("## Step 4 · Guardrails (Validate + Auto-Repair)")
g1, g2 = st.columns([1, 2])
with g1:
    run_guardrails = st.button("Run Guardrails", use_container_width=True, type="primary")
with g2:
    st.caption("This validates shell/spec/recipe structures and can auto-repair JSON with constrained retries.")

if run_guardrails:
    _log_event("guardrails_requested", "INFO")
    if not api_key:
        st.error("API key is required for auto-repair.")
        _log_event("guardrails_failed", "ERROR", {"reason": "missing_api_key"})
    else:
        stats = {}
        if st.session_state.table_json:
            fixed, issues, retries = validate_and_repair(
                kind="table_json",
                candidate=st.session_state.table_json,
                validator=validate_table_json,
                context={"adam_specs": st.session_state.adam_specs or {}},
                provider=provider,
                model=model,
                api_key=api_key,
                max_retries=repair_retries,
            )
            st.session_state.table_json = fixed
            st.session_state.table_issues = issue_dicts(issues)
            stats["table_json_retries"] = retries

        if st.session_state.adam_specs:
            fixed, issues, retries = validate_and_repair(
                kind="adam_specs",
                candidate=st.session_state.adam_specs,
                validator=validate_adam_specs,
                context={"table_json": st.session_state.table_json or {}},
                provider=provider,
                model=model,
                api_key=api_key,
                max_retries=repair_retries,
            )
            st.session_state.adam_specs = fixed
            st.session_state.adam_issues = issue_dicts(issues)
            stats["adam_specs_retries"] = retries
        st.session_state.repair_stats = stats
        st.success("Guardrail pass complete.")
        _log_event("guardrails_completed", "SUCCESS", stats)

if st.session_state.repair_stats:
    st.json(st.session_state.repair_stats)


st.markdown("## Step 5 · Generate Recipe + Assemble R + SAS")
if st.session_state.table_json:
    cls = route_table(
        table_json=st.session_state.table_json,
        adam_specs=st.session_state.adam_specs,
        mode=routing_mode,
        provider=provider,
        model=model,
        api_key=api_key,
        votes=classifier_votes,
    )
    st.info(
        f"Routing target: `{cls.table_type}` (confidence {cls.confidence:.2f}, source={cls.source}) · {cls.rationale}"
    )
if st.button("Generate Recipe and R + SAS Code", type="primary", use_container_width=True):
    _log_event("recipe_generation_requested", "INFO")
    if not api_key:
        st.error("API key is required.")
        _log_event("recipe_generation_failed", "ERROR", {"reason": "missing_api_key"})
    elif not st.session_state.table_json:
        st.error("Parse shell first.")
        _log_event("recipe_generation_failed", "ERROR", {"reason": "missing_table_json"})
    else:
        with st.spinner("Generating recipe..."):
            try:
                recipe = generate_recipe(
                    table_json=st.session_state.table_json,
                    adam_specs=st.session_state.adam_specs,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=recipe_temperature,
                    routing_mode=routing_mode,
                    classifier_votes=classifier_votes,
                )
                fixed_recipe, recipe_issues, retries = validate_and_repair(
                    kind="recipe_json",
                    candidate=recipe,
                    validator=lambda r: validate_recipe(r, st.session_state.table_json, st.session_state.adam_specs),
                    context={
                        "table_json": st.session_state.table_json,
                        "adam_specs": st.session_state.adam_specs or {},
                    },
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    max_retries=repair_retries,
                )
                st.session_state.recipe_json = fixed_recipe
                st.session_state.recipe_issues = issue_dicts(recipe_issues)
                st.session_state.repair_stats["recipe_retries"] = retries
                if recipe_issues:
                    fallback_recipe = build_deterministic_recipe(st.session_state.table_json, st.session_state.adam_specs)
                    fallback_issues = validate_recipe(fallback_recipe, st.session_state.table_json, st.session_state.adam_specs)
                    if not fallback_issues:
                        st.session_state.recipe_json = fallback_recipe
                        st.session_state.recipe_issues = []
                        st.session_state.r_code = assemble_r_from_recipe(fallback_recipe)
                        st.session_state.sas_code = assemble_sas_from_recipe(fallback_recipe)
                        st.success("LLM recipe was incomplete, so a validated deterministic R + SAS recipe was assembled.")
                        _log_event(
                            "recipe_generation_completed",
                            "SUCCESS",
                            {
                                "issue_count": 0,
                                "repair_retries": retries,
                                "used_deterministic_fallback": True,
                                "llm_generation": (fixed_recipe.get("_generation") or {}),
                                "original_issues": recipe_issues,
                            },
                        )
                    else:
                        st.error("Recipe still has validation issues; R/SAS assembly blocked.")
                        st.session_state.recipe_issues = issue_dicts(fallback_issues)
                        _log_event(
                            "recipe_generation_completed",
                            "WARNING",
                            {
                                "issue_count": len(fallback_issues),
                                "repair_retries": retries,
                                "used_deterministic_fallback": True,
                                "original_issues": recipe_issues,
                                "fallback_issues": fallback_issues,
                            },
                        )
                else:
                    st.session_state.r_code = assemble_r_from_recipe(fixed_recipe)
                    st.session_state.sas_code = assemble_sas_from_recipe(fixed_recipe)
                    st.success("Recipe validated and R + SAS code assembled.")
                    _log_event(
                        "recipe_generation_completed",
                        "SUCCESS",
                        {
                            "issue_count": 0,
                            "repair_retries": retries,
                            "used_deterministic_fallback": False,
                            "llm_generation": (fixed_recipe.get("_generation") or {}),
                        },
                    )
            except Exception as e:
                st.error(f"Recipe generation failed: {e}")
                _log_event("recipe_generation_failed", "ERROR", {"error": str(e)})

if st.session_state.recipe_json:
    st.markdown("### Recipe JSON")
    st.code(json.dumps(st.session_state.recipe_json, indent=2), language="json")
render_issues(st.session_state.recipe_issues, "Recipe issues")

if st.session_state.r_code or st.session_state.sas_code or st.session_state.recipe_json or st.session_state.recipe_issues:
    st.markdown("### Assembled Programs")
    left, right = st.columns(2)
    with left:
        st.markdown("#### R Script")
        if st.session_state.r_code:
            st.code(st.session_state.r_code, language="r")
            st.download_button(
                "Download R Script",
                data=st.session_state.r_code,
                file_name="generated_table.R",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("R script not available.")
    with right:
        st.markdown("#### SAS Program")
        if st.session_state.sas_code:
            st.code(st.session_state.sas_code, language="sas")
            st.download_button(
                "Download SAS Program",
                data=st.session_state.sas_code,
                file_name="generated_table.sas",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("SAS program not available.")


st.markdown("## Step 6 · Evaluation Harness")
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    run_llm_recipe = st.checkbox("Run LLM recipe in eval", value=True)
with c2:
    benchmark_routing = st.checkbox("Benchmark routing", value=True)
with c3:
    run_eval_btn = st.button("Run Eval Suite", type="primary", use_container_width=True)
with c4:
    st.caption(f"Cases path: `{EVAL_CASES_DIR}`")

if run_eval_btn:
    eval_run_id = str(uuid.uuid4())
    eval_start = time.time()
    _log_event(
        "eval_requested",
        "INFO",
        {
            "run_id": eval_run_id,
            "run_llm_recipe": run_llm_recipe,
            "benchmark_routing": benchmark_routing,
            "routing_mode": routing_mode,
            "classifier_votes": classifier_votes,
        },
        run_id=eval_run_id,
    )
    if run_llm_recipe and not api_key:
        st.error("API key is required for LLM-backed eval.")
        _log_event("eval_failed", "ERROR", {"reason": "missing_api_key_for_llm_eval"}, run_id=eval_run_id)
    else:
        with st.spinner("Running evaluation harness..."):
            try:
                result = run_suite(
                    cases_dir=EVAL_CASES_DIR,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    recipe_temperature=recipe_temperature,
                    run_llm_recipe=run_llm_recipe,
                    repair_retries=repair_retries,
                    routing_mode=routing_mode,
                    classifier_votes=classifier_votes,
                    benchmark_routing=benchmark_routing,
                    event_logger=lambda ev, stt, det: _log_event(ev, stt, det, run_id=eval_run_id),
                    run_id=eval_run_id,
                )
                st.session_state.eval_result = result
                _log_event(
                    "eval_completed",
                    "SUCCESS",
                    {
                        "run_id": eval_run_id,
                        "total_cases": result.get("total_cases"),
                        "passed_cases": result.get("passed_cases"),
                        "pass_rate": result.get("pass_rate"),
                        "routing_accuracy": result.get("routing_accuracy"),
                    },
                    run_id=eval_run_id,
                )
            except Exception as e:
                st.error(f"Eval failed: {e}")
                _log_event(
                    "eval_failed",
                    "ERROR",
                    {"error": str(e), "traceback": traceback.format_exc()},
                    run_id=eval_run_id,
                )
            finally:
                elapsed_ms = int((time.time() - eval_start) * 1000)
                _log_event(
                    "eval_finished",
                    "INFO",
                    {"run_id": eval_run_id, "elapsed_ms": elapsed_ms},
                    run_id=eval_run_id,
                )

if st.session_state.eval_result:
    result = st.session_state.eval_result
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Cases", result["total_cases"])
    m2.metric("Passed Cases", result["passed_cases"])
    m3.metric("Pass Rate", f"{result['pass_rate']:.1f}%")
    ae_acc = result.get("ae_nested_accuracy")
    m4.metric("AE Nested Accuracy", "N/A" if ae_acc is None else f"{ae_acc:.1f}%")
    m5.metric("Unknown Var Issues", result.get("unknown_var_issues_total", 0))
    ra = result.get("routing_accuracy")
    m6.metric("Routing Accuracy", "N/A" if ra is None else f"{ra:.1f}%")
    st.caption(f"Flag misuse issues total: {result.get('flag_misuse_issues_total', 0)}")
    st.caption(f"Routing scored cases: {result.get('routing_scored_cases', 0)}")
    st.markdown("### Case Results")
    for r in result["results"]:
        label = "✅ PASS" if r["passed"] else "❌ FAIL"
        with st.expander(f"{label} · {r['id']}"):
            st.write(r["description"])
            st.write(f"Route: `{r['metrics']['table_type']}` (confidence {r['metrics']['route_confidence']:.2f})")
            if r["metrics"].get("expected_table_type"):
                st.write(
                    f"Expected route: `{r['metrics']['expected_table_type']}` · Correct: `{r['metrics']['route_correct']}`"
                )
            st.write(f"Recipe repair retries: {r.get('recipe_repair_retries', 0)}")
            if r["llm_error"]:
                st.error(r["llm_error"])
            if r["table_issues"]:
                st.write("Table issues:")
                st.json(r["table_issues"])
            if r["adam_issues"]:
                st.write("AdaM issues:")
                st.json(r["adam_issues"])
            if r.get("recipe_issues_before_repair"):
                st.write("Recipe issues (before repair):")
                st.json(r["recipe_issues_before_repair"])
            if r["recipe_issues"]:
                st.write("Recipe issues:")
                st.json(r["recipe_issues"])
            st.write("Metrics:")
            st.json(r["metrics"])
