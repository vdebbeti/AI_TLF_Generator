from dataclasses import dataclass
import json
from collections import Counter

from llm_client import call_llm
from prompts import CLASSIFIER_SYSTEM


@dataclass
class TableClassification:
    table_type: str
    confidence: float
    rationale: str
    source: str = "heuristic"


def classify_table(table_json: dict, adam_specs: dict | None = None) -> TableClassification:
    meta = table_json.get("table_metadata", {}) if isinstance(table_json, dict) else {}
    title = (meta.get("title") or "").lower()
    ds = (meta.get("dataset_source") or "").upper()
    rows = table_json.get("rows", []) if isinstance(table_json, dict) else []
    analysis_vars = {(r.get("analysis_var") or "").upper() for r in rows if isinstance(r, dict)}

    if ds == "ADAE" or "adverse event" in title or "teae" in title or {"AEBODSYS", "AEDECOD"} & analysis_vars:
        return TableClassification("ae", 0.95, "Detected ADAE/AE SOC-PT patterns.", "heuristic")
    if ds == "ADSL" or "demographic" in title or "baseline" in title or {"AGE", "SEX", "RACE"} & analysis_vars:
        return TableClassification("demog", 0.9, "Detected ADSL demographics/baseline patterns.", "heuristic")
    if ds == "ADRS" or "response" in title or "bor" in title or "orr" in title or "avalc" in {v.lower() for v in analysis_vars}:
        return TableClassification("response", 0.9, "Detected ADRS response/BOR patterns.", "heuristic")
    if ds == "ADTTE" or "survival" in title or "kaplan" in title or "time to" in title:
        return TableClassification("survival", 0.8, "Detected ADTTE/survival patterns.", "heuristic")
    if adam_specs and (adam_specs.get("dataset") or "").upper() in {"ADAE", "ADSL", "ADRS", "ADTTE"}:
        mapped = {"ADAE": "ae", "ADSL": "demog", "ADRS": "response", "ADTTE": "survival"}[
            (adam_specs.get("dataset") or "").upper()
        ]
        return TableClassification(mapped, 0.7, "Inferred from AdaM dataset.", "heuristic")
    return TableClassification("generic", 0.5, "No strong signature; using generic route.", "heuristic")


def _strip_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json\n"):
            raw = raw[5:]
    return raw.strip()


def classify_table_llm(
    table_json: dict,
    adam_specs: dict | None,
    provider: str,
    model: str,
    api_key: str,
) -> TableClassification:
    user = (
        f"Table JSON:\n{json.dumps(table_json, indent=2)}\n\n"
        f"AdaM Specs:\n{json.dumps(adam_specs or {}, indent=2)}"
    )
    raw = call_llm(
        system=CLASSIFIER_SYSTEM,
        user=user,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=700,
        temperature=0.2,
        json_mode=True,
    )
    obj = json.loads(_strip_fences(raw))
    t = str(obj.get("table_type", "generic")).lower().strip()
    if t not in {"ae", "demog", "response", "survival", "generic"}:
        t = "generic"
    conf = obj.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return TableClassification(t, conf, str(obj.get("rationale", "LLM classification")), "llm")


def classify_table_consensus(
    table_json: dict,
    adam_specs: dict | None,
    provider: str,
    model: str,
    api_key: str,
    votes: int = 3,
) -> TableClassification:
    heuristic = classify_table(table_json, adam_specs)
    llm_votes: list[TableClassification] = []
    vote_count = max(1, votes)
    for _ in range(vote_count):
        try:
            llm_votes.append(classify_table_llm(table_json, adam_specs, provider, model, api_key))
        except Exception:
            continue
    if not llm_votes:
        heuristic.source = "consensus_fallback_heuristic"
        return heuristic

    c = Counter(v.table_type for v in llm_votes)
    top_count = c.most_common(1)[0][1]
    top_types = [k for k, v in c.items() if v == top_count]
    if len(top_types) == 1:
        winner = top_types[0]
    else:
        by_conf = {}
        for t in top_types:
            confs = [v.confidence for v in llm_votes if v.table_type == t]
            by_conf[t] = sum(confs) / len(confs)
        winner = sorted(by_conf.items(), key=lambda x: x[1], reverse=True)[0][0]
    winner_votes = [v for v in llm_votes if v.table_type == winner]
    avg_conf = sum(v.confidence for v in winner_votes) / max(1, len(winner_votes))
    rationale = f"Consensus {len(winner_votes)}/{len(llm_votes)} LLM votes; heuristic={heuristic.table_type}"
    return TableClassification(winner, avg_conf, rationale, "consensus")


def route_table(
    table_json: dict,
    adam_specs: dict | None,
    mode: str = "heuristic",
    provider: str = "OpenAI",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    votes: int = 3,
) -> TableClassification:
    m = (mode or "heuristic").lower()
    if m == "llm" and api_key:
        return classify_table_llm(table_json, adam_specs, provider, model, api_key)
    if m == "consensus" and api_key:
        return classify_table_consensus(table_json, adam_specs, provider, model, api_key, votes=votes)
    return classify_table(table_json, adam_specs)
