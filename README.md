# TLF Compiler V2

This V2 workspace implements:

- Phase 1 guardrails:
  - JSON structure validation for shell/spec/recipe
  - semantic checks (AE nested SOC/PT, flag misuse, invalid variable references)
  - auto-repair loop with validator feedback
  - no silent fallback to free-form code generation
- Initial Phase 4 harness:
  - golden eval cases under `eval_cases/`
  - batch run + pass-rate reporting in-app
  - richer metrics: AE SOC/PT nested accuracy, unknown-variable issue counts, flag-misuse counts
  - pre/post repair issue tracking
- Table-type classifier + prompt routing:
  - routing modes: `heuristic`, `llm`, `consensus`
  - routes recipe generation to `ae`, `demog`, `response`, or `generic` prompts
  - consensus mode uses majority voting across multiple LLM classifier passes
  - eval harness can benchmark routing accuracy against `expected_table_type`

## Run

```bash
cd v2
streamlit run app.py
```

## Notes

- Use the sidebar to set temperatures separately:
  - shell parsing (can remain higher)
  - AdaM parsing
  - recipe generation (kept lower by default)
- Step numbering in UI matches execution order exactly (1 to 6).
