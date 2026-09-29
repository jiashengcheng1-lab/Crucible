# Contributing

- `pip install -e ".[all]"` then `python -m pytest -q`; every change ships with a test that runs offline (the mock model and the synthetic company are there for that).
- The rules in `CLAUDE.md` are the project's invariants: no model-generated numbers in the statements, evidence packets frozen and hashed, every citation verbatim from a filing, analyst decisions recorded in the ledger and never overwritten by a rerun.
- New industry templates go in `crucible/templates.py` as data (drivers, KPI patterns, build); new decisions go in `crucible/dossier.py` as a `DecisionSpec`.
- Keep `model.forecast` and `excel.write_model` in lockstep; `tests/test_excel.py` recalculates the workbook with LibreOffice when it is installed.
