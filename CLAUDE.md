# CLAUDE.md

Working notes for Claude Code sessions in this repo.

## What this is
An equity-research MVP: SEC filings -> standardized three-statement model + DCF (deterministic) -> adversarial
debate over forecast assumptions with evidence citation -> robustness/ablation harness -> analyst decision log.

## Rules that must not be broken
- No LLM-generated numbers in the model. Numbers come from XBRL via edgartools and from explicit driver inputs.
- Evidence packets are frozen and hashed. Agents cite evidence ids; the validator drops anything uncited.
- Keep `model.forecast` and `excel.write_model` in lockstep. `tests/test_excel.py` recalculates the workbook
  with LibreOffice and asserts parity; run it after touching either file.
- Public SEC data only. Do not add scrapers for licensed data.

## Commands
- `python -m pytest -q` before every commit.
- `python -m crucible demo` runs the whole loop offline (synthetic data, mock LLM).
- `python -m crucible ingest TICKER --peers ...` needs internet and `EDGAR_IDENTITY`.

## When ingest produces unmapped labels
Add a concept or label pattern to `crucible/schema.py` (concept match preferred), or write
`data/<TICKER>/mapping_overrides.json`. Re-run `python -m crucible model TICKER` and check the residual sizes
in `mapping_report.txt`; residuals above ~5% of the relevant total mean the schema is missing a line.

## Style
Small functions, explicit units (USD millions unless meta.json says otherwise), no hidden state.
