"""crucible CLI.

  crucible ingest VRT --peers ETN NVT MOD            pull filings (needs internet)
  crucible model VRT                                  Excel model from data/VRT
  crucible packet VRT --assumption revenue_growth_3y  freeze an evidence packet
  crucible debate VRT --assumption wacc --provider anthropic
  crucible harness VRT --assumption revenue_growth_3y --runs 5 --loo-k 2
  crucible decide VRT --assumption wacc --action edit --value 0.095 --reason "..."
  crucible demo                                       whole loop offline on synthetic data + mock LLM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .debate import review_markdown, run_debate, transcript
from .evidence import ASSUMPTIONS, EvidencePacket, build_packet
from .templates import MINER_ASSUMPTIONS, TEMPLATES, fit_kpis, load_kpis, save_kpis

ASSUMPTIONS.update(MINER_ASSUMPTIONS)
from .harness import DebateCache, attribution_summary, compare_arms, leave_one_out, permutation, pooled_noise_floor, stability, vote
from .llm import get_llm
from .log import RunLog
from .model import Drivers, WACCInputs, checks, dcf, forecast


def _load_company(args) -> tuple[str, pd.DataFrame, dict, dict[str, str], dict]:
    """Return (company_label, hist, peers, sections, meta). meta["releases"] holds the earnings releases in scope."""
    if getattr(args, "fixture", False) or args.ticker.upper() == "SYNTH":
        from .synthetic import SYNTHETIC_META, synthetic_history

        return SYNTHETIC_META["name"], synthetic_history(), {}, {}, {**SYNTHETIC_META, "releases": []}
    from .ingest import load_eightk, load_meta, load_sections
    from .mapping import load_history

    data = Path(args.data)
    as_of, hy = getattr(args, "as_of", None), getattr(args, "hist_years", 5)
    hist, reports = load_history(data, args.ticker, as_of=as_of, hist_years=hy)
    long_hist, _ = load_history(data, args.ticker, as_of=as_of, hist_years=20)
    meta = load_meta(data, args.ticker)
    meta = {**meta, "long_hist": long_hist}
    meta = {**meta, "units": "USD millions", "releases": load_eightk(data, args.ticker, as_of=as_of, n=getattr(args, "releases", 2)),
            "all_releases": load_eightk(data, args.ticker, as_of=as_of, n=100)}
    peers = {}
    for p in _peers(args):
        try:
            peers[p.upper()], _ = load_history(data, p, as_of=as_of, hist_years=hy)
        except FileNotFoundError as e:
            print(f"warning: {e}", file=sys.stderr)
    return meta.get("name") or args.ticker.upper(), hist, peers, load_sections(data, args.ticker, as_of=as_of), meta


def _role_llms(args, default) -> dict:
    """Per-role model overrides: --bull-model anthropic:claude-opus-5-5, --bear-model openai:gpt-... etc."""
    out = {}
    for role in ("bull", "bear", "judge"):
        spec = getattr(args, f"{role}_model", None)
        if spec:
            out[role] = get_llm(spec)
    return out


def _calibration(args) -> dict | None:
    p = Path(f"outputs/{str(getattr(args, 'ticker', '')).upper()}_{getattr(args, 'assumption', '')}_calibration.json")
    return json.loads(p.read_text()) if p.exists() else None


def _cache(args) -> DebateCache | None:
    return None if getattr(args, "no_cache", False) else DebateCache(Path(args.logs) / "cache")


def _analyst(args) -> dict:
    """Analyst inputs, in order: data/<TICKER>/analyst_inputs.json (derived, dated, cited) > --analyst <flat json> > placeholders (warned)."""
    from .inputs import flatten, load_inputs

    if not getattr(args, "fixture", False) and getattr(args, "ticker", None):
        inp = load_inputs(Path(getattr(args, "data", "data")), args.ticker)
        if inp:
            return flatten(inp)
    if getattr(args, "analyst", None):
        return json.loads(Path(args.analyst).read_text())
    print("warning: no derived analyst inputs (run `crucible inputs TICKER --peers ...`); using placeholders for rf/ERP/beta", file=sys.stderr)
    return {"risk_free": 0.042, "erp": 0.05, "beta": 1.2, "risk_free_source": "PLACEHOLDER (run crucible inputs)",
            "erp_source": "PLACEHOLDER (run crucible inputs)", "beta_source": "PLACEHOLDER (run crucible inputs)"}


def cmd_inputs(args) -> None:
    """Derive the cost-of-capital inputs from FRED, Yahoo Finance and the filings; cache by date; apply overrides and notes."""
    from .inputs import derive_inputs, flatten, is_fresh, load_inputs, save_inputs, summary
    from .mapping import load_history

    data = Path(args.data)
    hist, _ = load_history(data, args.ticker, as_of=args.as_of)
    existing = load_inputs(data, args.ticker)
    changed = False
    if existing is None or args.refresh or not is_fresh(existing, args.as_of, args.refresh_days):
        inp = derive_inputs(args.ticker, data, hist, peers=_peers(args), as_of=args.as_of, existing=existing, erp_method=args.erp_method)
        changed = True
    else:
        inp = existing
        print(f"reusing {inp['as_of']} inputs (within {args.refresh_days} days; --refresh to rebuild)", file=sys.stderr)
    if args.erp is not None:
        inp.setdefault("fields", {})["erp"] = {"value": args.erp, "source": args.erp_source or "analyst input (no source given)", "analyst_set": True}
        changed = True
    for kv in args.set or []:
        key, _, val = kv.partition("=")
        if key not in ("risk_free", "erp", "beta", "cost_of_debt", "debt_weight", "tax_rate"):
            raise SystemExit(f"--set key must be one of risk_free, erp, beta, cost_of_debt, debt_weight, tax_rate (got {key})")
        inp.setdefault("overrides", {})[key] = {"value": float(val), "reason": args.reason or "no reason given"}
        changed = True
    for n in args.note or []:
        inp.setdefault("notes", []).append(n)
        changed = True
    if args.clear_overrides:
        inp["overrides"] = {}
        changed = True
    if changed:
        p = save_inputs(data, args.ticker, inp)
        print(f"saved {p}", file=sys.stderr)
    print(summary(inp))
    flat = flatten(inp)
    if all(k in flat for k in ("risk_free", "erp", "beta")):
        ke = flat["risk_free"] + flat["beta"] * flat["erp"]
        w, kd, t = flat.get("debt_weight", 0.0), flat.get("cost_of_debt", 0.0), flat.get("tax_rate", 0.21)
        print(f"  cost of equity {ke:.2%}  |  mechanical WACC {(1 - w) * ke + w * kd * (1 - t):.2%}")


def cmd_ingest(args) -> None:
    from .ingest import ingest

    for t in [args.ticker] + (args.peers or []):
        d = ingest(t, Path(args.data), years=args.years, quarters=args.quarters, with_text=not args.no_text, with_pit=not args.no_pit,
                   with_eightk=not args.no_eightk, only_eightk=args.only_eightk, with_xbrl=not args.no_xbrl)
        print(f"wrote {d}")
    if args.peers:  # remember the peer set so later commands do not need --peers
        pf = Path(args.data) / args.ticker.upper() / "peers.json"
        pf.write_text(json.dumps([p.upper() for p in args.peers]))
        print(f"peer set saved to {pf}; later commands use it unless --peers is given")


def _peers(args) -> list[str]:
    """--peers if given, else the set saved by ingest."""
    given = getattr(args, "peers", None)
    if given:
        return [p.upper() for p in given]
    pf = Path(getattr(args, "data", "data")) / str(getattr(args, "ticker", "")).upper() / "peers.json"
    if pf.exists():
        try:
            return json.loads(pf.read_text())
        except Exception:
            return []
    return []


def _scenarios_from_debate(args, drv, hist) -> dict:
    """Bull/base/bear driver sets: growth shifts from the latest revenue_growth_3y verdict when one exists (high/low vs base),
    else +/-5pp; margin shifts +/-1pp; WACC -/+50bp; probabilities 25/50/25. The analyst edits the Scenarios sheet."""
    up, down = 0.05, -0.05
    p = Path(f"outputs/{args.ticker.upper()}_revenue_growth_3y_debate_seed1.json")
    src = "default +/-5pp growth"
    if p.exists():
        try:
            v = json.loads(p.read_text())["verdict"]
            base_g = sum(drv.revenue_growth) / len(drv.revenue_growth)
            up, down = float(v["high"]) - base_g, float(v["low"]) - base_g
            src = f"debate verdict low {v['low']:.1%} / high {v['high']:.1%} vs model base growth {base_g:.1%}"
        except Exception:
            pass
    return {"base": {"growth_delta": 0.0, "margin_delta": 0.0, "capex_delta": 0.0, "wacc_delta": 0.0, "tg_delta": 0.0, "prob": 0.5},
            "bull": {"growth_delta": up, "margin_delta": 0.01, "capex_delta": 0.0, "wacc_delta": -0.005, "tg_delta": 0.0, "prob": 0.25, "taper": 0.85},
            "bear": {"growth_delta": down, "margin_delta": -0.01, "capex_delta": 0.005, "wacc_delta": 0.005, "tg_delta": 0.0, "prob": 0.25, "taper": 0.85},
            "_source": src}


def cmd_model(args) -> None:
    from .excel import write_model
    from .model import elasticity, scenario_values

    name, hist, _, _, meta = _load_company(args)
    drv = Drivers.from_history(hist, years=args.years)
    drv.nwc_method = args.nwc_method
    if args.drivers:
        drv = Drivers.from_dict({**drv.to_dict(), **json.loads(Path(args.drivers).read_text())})
        drv.nwc_method = args.nwc_method
    if args.wacc:
        wacc_in = WACCInputs(**json.loads(Path(args.wacc).read_text()))
    else:
        flat = _analyst(args)
        wacc_in = WACCInputs(risk_free=flat.get("risk_free", 0.042), beta=flat.get("beta", 1.2), erp=flat.get("erp", 0.05),
                             pretax_cost_of_debt=flat.get("cost_of_debt", 0.055), tax_rate=flat.get("tax_rate", 0.21), debt_weight=flat.get("debt_weight", 0.2))
    if not args.wacc and flat.get("diluted_shares") and not args.no_dilution:
        drv.shares_override = flat["diluted_shares"] / 1e6
        drv.notes["shares"] = flat.get("diluted_shares_source", "")
    template = _template_for(args, name, hist, drv) if not getattr(args, "fixture", False) else None
    fc = forecast(hist, drv)
    chk = checks(fc, drv.years)
    res = dcf(fc, drv, wacc_in.wacc, args.g, mid_year=args.mid_year)
    scn = _scenarios_from_debate(args, drv, hist)
    src = scn.pop("_source")
    scenarios = scenario_values(hist, drv, wacc_in.wacc, args.g, scn, mid_year=args.mid_year)
    elas = elasticity(hist, drv, wacc_in.wacc, args.g, mid_year=args.mid_year)
    out = Path(args.out or f"outputs/{args.ticker.upper()}_model.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    mr = Path(args.data) / args.ticker.upper() / "mapping_report.txt"
    if mr.exists():
        notes = mr.read_text().splitlines()[:12]
    write_model(hist, drv, wacc_in, args.g, out, company=name, units=meta.get("units", "USD millions"),
                sources=meta.get("sources_by_fiscal_year") or meta.get("sources"), mapping_notes=notes, mid_year=args.mid_year,
                scenarios=scenarios, elasticity_rows=elas, audit=args.audit_links, template=template)
    Path(f"outputs/{args.ticker.upper()}_scenarios.json").write_text(json.dumps({"source": src, **scenarios}, indent=2))
    Path(f"outputs/{args.ticker.upper()}_elasticity.json").write_text(json.dumps(elas, indent=2))
    fc.to_csv(f"outputs/{args.ticker.upper()}_forecast.csv")  # the statements as built, history and forecast, for previews and questions
    warn = []
    gm = sum(drv.gross_margin) / len(drv.gross_margin)
    if scenarios.get("bull", {}).get("per_share", 0) < scenarios.get("base", {}).get("per_share", 0):
        warn.append(f"value falls as growth rises: the trailing-average driver set has negative unit economics (gross margin {gm:.0%}); growth destroys value "
                    f"until the industry driver template replaces these defaults (see the industry_template decision)")
    if res.per_share < 0:
        warn.append("negative equity value per share: the driver set never reaches positive free cash flow; this is the arithmetic of the history, not a valuation")
    if res.as_dict()["tv_share_of_ev"] > 0.85 or res.as_dict()["tv_share_of_ev"] < 0:
        warn.append(f"terminal value is {res.as_dict()['tv_share_of_ev']:.0%} of EV: the explicit forecast carries little of the value")
    summary = {"workbook": str(out), "checks": chk, "wacc": round(wacc_in.wacc, 4), "dcf": {k: round(v, 3) for k, v in res.as_dict().items()},
               "scenarios": {k: round(v["per_share"], 2) for k, v in scenarios.items()}, "scenario_source": src,
               "top_elasticities": [(r["driver"], r["delta"]) for r in elas[:4]], "nwc_method": drv.nwc_method,
               "driver_notes": {k: v for k, v in drv.notes.items() if k != "basis"}, "warnings": warn,
               "drivers": {k: [round(float(x), 4) for x in v] for k, v in drv.to_dict().items() if isinstance(v, list) and k in ("revenue_growth", "gross_margin", "sga_pct", "da_pct", "capex_pct", "tax_rate", "dso", "dio", "dpo")},
               "template": ({"key": template["key"], "revenue_path": [round(float(x), 0) for x in template["build"].revenue],
                             "btc_mined": [round(float(x), 0) for x in template["build"].btc_mined], "gross_margin": [round(float(x), 3) for x in template["build"].gross_margin],
                             "treasury_value_end": round(float(template["build"].treasury_value.iloc[-1]), 0)} if template else None)}
    Path(f"outputs/{args.ticker.upper()}_model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _kpis(args) -> pd.DataFrame:
    """Fitted KPI table for the company (fitted from the releases on first use, then cached as data/<T>/kpi_history.csv)."""
    if getattr(args, "fixture", False) or args.ticker.upper() == "SYNTH":
        return pd.DataFrame()
    data = Path(args.data)
    q = load_kpis(data, args.ticker)
    if not len(q):
        from .ingest import load_eightk

        q = fit_kpis(load_eightk(data, args.ticker, n=100))
        if len(q):
            save_kpis(data, args.ticker, q)
    return q


def cmd_kpis(args) -> None:
    """Fit the industry template's KPI table from the earnings releases and print it with its citations."""
    from .ingest import load_eightk
    from .templates import annual_kpis

    q = fit_kpis(load_eightk(Path(args.data), args.ticker, n=100), template=args.template)
    if not len(q):
        raise SystemExit("no KPI sentences matched; the releases may be missing (run ingest) or the template does not fit")
    p = save_kpis(Path(args.data), args.ticker, q)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    show = [c for c in q.columns if not c.endswith("_cite") and c not in ("accession", "filing_date")]
    print(q[show].to_string(index=False))
    print("\nfiscal-year drivers (sums for flows, means for rates):")
    print(annual_kpis(q).round(3).to_string())
    if args.cites:
        for _, r in q.tail(2).iterrows():
            for c in [c for c in q.columns if c.endswith("_cite")]:
                if isinstance(r[c], str) and r[c]:
                    print(f"  {r.quarter} {c[:-5]}: {r[c][:200]}")
    print(f"saved {p}")


def _template_for(args, name, hist, drv):
    """Resolve --template (auto = the accepted industry_template decision) and build the driver paths."""
    from .decisions import DecisionLedger
    from .templates import apply_build_to_drivers, miner_build, miner_defaults

    key = args.template
    if key == "auto":
        try:
            sel = DecisionLedger(Path(args.data)).selected(args.ticker, "industry_template")
            key = sel[0] if sel else "none"
        except Exception:
            key = "none"
    if key in ("none", None) or key not in TEMPLATES:
        return None
    q = _kpis(args)
    if not len(q):
        print(f"template {key}: no KPI history fitted from the releases; falling back to trailing-average drivers", file=sys.stderr)
        return None
    inputs = miner_defaults(hist, q, drv.years)
    if args.template_inputs:
        for k, v in json.loads(Path(args.template_inputs).read_text()).items():
            if k in ("fleet_life_years", "btc_held"):
                inputs[k] = float(v)
            elif k in ("hashrate_last",):
                inputs[k] = float(v)
            elif isinstance(v, (int, float)):
                inputs[k] = [float(v)] * len(drv.years)
            else:
                inputs[k] = [float(x) for x in v]
    # driver debates already run replace the default growth paths (base verdict), tapering as the defaults do
    for akey, dkey in (("hashrate_growth_3y", "hashrate_eh"), ("network_hashrate_growth_3y", "network_hashrate_eh"), ("btc_price_change_3y", "btc_price")):
        vp = Path(f"outputs/{args.ticker.upper()}_{akey}_debate_seed1.json")
        if vp.exists() and not (args.template_inputs and dkey in json.loads(Path(args.template_inputs).read_text())):
            try:
                base = float(json.loads(vp.read_text())["verdict"]["base"])
                from .templates import annual_kpis
                a = annual_kpis(q)
                full = a[a.quarters == 4] if len(a) else a
                col = {"hashrate_eh": "hashrate_eh", "network_hashrate_eh": "network_hashrate_eh", "btc_price": "avg_price_produced"}[dkey]
                start = float(full.iloc[-1][col]) if len(full) and col in full.columns and full.iloc[-1][col] == full.iloc[-1][col] else inputs[dkey][0] / (1 + base)
                path, v, g = [], start, base
                for _ in drv.years:
                    v = v * (1 + g)
                    path.append(round(v, 4))
                    g *= 0.85
                inputs[dkey] = path
                inputs["basis"] += f"; {dkey} path from the {akey} debate verdict (base {base:+.1%}, tapering)"
            except Exception as e:
                print(f"warning: {akey} verdict not applied: {e}", file=sys.stderr)
    build = miner_build(hist, drv.years, inputs)
    apply_build_to_drivers(drv, build)
    drv.notes["template"] = f"{key}: {inputs['basis']}"
    from .templates import annual_kpis
    a = annual_kpis(q)
    full = a[a.quarters == 4] if len(a) else a
    last_hr = float(full.iloc[-1].hashrate_eh) if len(full) and "hashrate_eh" in full.columns else None
    Path("outputs").mkdir(exist_ok=True)
    build.to_csv(f"outputs/{args.ticker.upper()}_driver_build.csv")
    Path(f"outputs/{args.ticker.upper()}_template_inputs.json").write_text(json.dumps(inputs, indent=2))
    return {"key": key, "label": TEMPLATES[key].label, "inputs": inputs, "build": build, "basis": inputs["basis"], "last_hashrate": inputs.get("hashrate_last") or last_hr}


def cmd_packet(args) -> EvidencePacket:
    name, hist, peers, sections, meta = _load_company(args)
    pk = build_packet(args.assumption, name, hist, peers=peers, sections=sections, analyst=_analyst(args),
                      as_of=getattr(args, "as_of", None) or (meta.get("tenk") or {}).get("filing_date"), releases=meta.get("releases") or [],
                      all_releases=meta.get("all_releases") or [], long_hist=meta.get("long_hist"), kpis=_kpis(args))
    print(f"history window FY{hist.columns[0]}-FY{hist.columns[-1]} ({len(hist.columns)} fiscal years; all years on disk: "
          f"FY{meta['long_hist'].columns[0]}-FY{meta['long_hist'].columns[-1]})" if meta.get("long_hist") is not None else "", file=sys.stderr)
    if not meta.get("releases") and not getattr(args, "fixture", False):
        print("note: no earnings releases found under data/<TICKER>/eightk (run `crucible ingest` with v0.3+); guidance and orders evidence will be missing", file=sys.stderr)
    out = Path(args.out or f"outputs/{args.ticker.upper()}_{args.assumption}_packet.json")
    pk.save(out)
    print(f"packet {pk.hash()} with {len(pk.items)} items -> {out}")
    print(pk.render())
    return pk


def _packet(args) -> EvidencePacket:
    """Rebuild the packet from data every time (cheap, no API calls) unless the user points at a specific packet file,
    so evidence-builder changes and new filings are never masked by a stale outputs/ file."""
    if getattr(args, "packet", None):
        return EvidencePacket.load(Path(args.packet))
    return cmd_packet(args)


def cmd_debate(args) -> None:
    pk = _packet(args)
    llm = get_llm(args.provider, getattr(args, "model", None))
    res = run_debate(pk, llm, rounds=args.rounds, seed=args.seed, temperature=args.temperature, policy=args.policy, role_llms=_role_llms(args, llm))
    RunLog(Path(args.logs)).record_run(res, tag="cli")
    print(transcript(res))
    out = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_debate_seed{args.seed}.json")
    out.write_text(res.model_dump_json(indent=2))
    review = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_review.md")
    review.write_text(review_markdown(res, calibration=_calibration(args)))
    print(f"\nsaved {out} and {review}")


def cmd_harness(args) -> None:
    pk = _packet(args)
    llm = get_llm(args.provider, getattr(args, "model", None))
    role_llms = _role_llms(args, llm)
    cache = _cache(args)
    log = RunLog(Path(args.logs))
    from .harness import group_key

    n_units = len({group_key(i) for i in pk.items}) if args.loo_mode == "groups" else len(pk.items)
    n_debates = 2 * args.runs + (n_units * args.loo_k if args.loo_k > 0 else 0)
    print(f"harness plan: {args.runs} stability + {args.runs} permutation + {n_units} x {args.loo_k} ablation ({args.loo_mode}) = {n_debates} debates, "
          f"~{n_debates * (2 * args.rounds + 1)} model calls, {args.workers} workers, cache {'on' if cache else 'off'} (~{n_debates * 75 / max(1, args.workers) / 60:.0f} min if nothing is cached)")
    kw = dict(rounds=args.rounds, temperature=args.temperature, policy=args.policy, workers=args.workers, cache=cache, role_llms=role_llms)
    st = stability(pk, llm, n=args.runs, **kw)
    pm = permutation(pk, llm, n=args.runs, **kw)
    for r in st.results + pm.results:
        log.record_run(r, tag="harness")
    print("STABILITY   ", json.dumps(st.summary(args.tol)))
    print("PERMUTATION ", json.dumps(pm.summary(args.tol)))
    arms = None
    if args.vote_n > 0:
        vt = vote(pk, llm, n=args.vote_n, temperature=args.temperature, workers=args.workers, cache=cache)
        print("VOTE (control: independent estimates, no debate)", json.dumps(vt.summary()))
        arms = compare_arms(st, vt, pooled_noise_floor(st, pm))
        print("DEBATE vs VOTE", json.dumps(arms))
    att_rows = None
    if args.ablation == "random":
        from .attribution import attribution_random, attribution_table

        rk = args.random_k
        print(f"random-subset attribution: {rk} debates (~{rk * (2 * args.rounds + 1)} calls), OLS of base on group inclusion")
        adf, fit, rres = attribution_random(pk, llm, k=rk, rounds=args.rounds, temperature=args.temperature, policy=args.policy, workers=args.workers,
                                            cache=cache, role_llms=role_llms)
        for r in rres:
            log.record_run(r, tag="harness random-subset")
        print("\nEVIDENCE ATTRIBUTION (regression on random subsets; effect = change in base when the group is included)")
        print(attribution_table(fit))
        Path(f"outputs/{args.ticker.upper()}_{args.assumption}_attribution_fit.json").write_text(json.dumps(fit, indent=2))
        adf.to_csv(f"outputs/{args.ticker.upper()}_{args.assumption}_attribution.csv", index=False)
        att_rows = [{"evidence_id": r["group"], "delta_base": (r["effect"] if r["effect"] == r["effect"] else 0.0), "delta_width": 0.0} for r in fit["rows"]]
    if args.ablation == "loo" and args.loo_k > 0:
        att = leave_one_out(pk, llm, k=args.loo_k, full=st, mode=args.loo_mode, **kw)
        out = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_attribution.csv")
        att.to_csv(out, index=False)
        pd.set_option("display.width", 200)
        print("\nEVIDENCE ATTRIBUTION (leave-one-out; delta_base = base without item minus base with full packet)")
        print(att[["evidence_id", "kind", "delta_base", "delta_width", "content"]].to_string(index=False))
        print("\n", json.dumps(attribution_summary(att, noise_floor=pooled_noise_floor(st, pm)), indent=2, default=str))
        print(f"saved {out}")
        att_rows = att.to_dict("records")
    review = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_review.md")
    review.write_text(review_markdown(st.results[0], stability=st.summary(args.tol), attribution_rows=att_rows, arms=arms, calibration=_calibration(args)))
    print(f"review sheet: {review}")


def cmd_decide(args) -> None:
    log = RunLog(Path(args.logs))
    runs = [r for r in log.read("runs.jsonl") if r["assumption"] == args.assumption]
    last = runs[-1] if runs else None
    v = (last or {}).get("verdict", {})
    rec = log.record_decision(args.ticker.upper(), args.assumption, (last or {}).get("packet_hash", "n/a"),
                              v.get("low", float("nan")), v.get("base", float("nan")), v.get("high", float("nan")),
                              args.action, args.value, args.reason, analyst=args.analyst_name)
    print(json.dumps(rec, indent=2))
    print(json.dumps(log.decision_stats(), indent=2))


def _tenk_dates(data_dir: Path, ticker: str) -> list[dict]:
    """10-K filings with their fiscal year and filing date (from meta.json), plus what a backtest could score today."""
    from .ingest import load_meta
    from .mapping import load_history

    meta = load_meta(data_dir, ticker)
    full, _ = load_history(data_dir, ticker, hist_years=15)
    latest = int(full.columns[-1])
    found: dict[int, str] = {}
    for f in meta.get("filings", []):  # filings index (recent filings only)
        if f.get("form") == "10-K" and str(f.get("period_of_report", ""))[:4].isdigit():
            found.setdefault(int(f["period_of_report"][:4]), f["filing_date"])
    for p in (Path(data_dir) / ticker.upper()).glob("tenk_sections_*.json"):  # per-year 10-K text carries its filing date
        try:
            fl = json.loads(p.read_text()).get("_filing") or {}
            if str(fl.get("period_of_report", ""))[:4].isdigit() and fl.get("filing_date"):
                found.setdefault(int(fl["period_of_report"][:4]), fl["filing_date"])
        except Exception:
            continue
    out = []
    for fy, fd in found.items():
        if fy not in full.columns:
            continue
        out.append({"fy": fy, "filing_date": fd, "as_of": fd, "years_realized": max(0, min(3, latest - fy))})
    return sorted(out, key=lambda x: x["fy"])


def _ledger(args):
    from .ledger import MappingLedger

    return MappingLedger(Path(args.data))


def cmd_map_status(args) -> None:
    from .mapping import load_history

    load_history(Path(args.data), args.ticker)  # records rule matches
    print(_ledger(args).summary(args.ticker))


def cmd_map_suggest(args) -> None:
    """LLM proposals for the unmapped residue, written to the ledger as pending."""
    from .facts import load_xbrl_facts
    from .mapping import load_history
    from .mapsuggest import MockMapper, candidates, suggest, write_pending

    data = Path(args.data)
    hist, reports = load_history(data, args.ticker)
    ledger = _ledger(args)
    xbrl = load_xbrl_facts(data, args.ticker)
    from .ingest import load_meta, load_sections
    from .mapsuggest import context_sentences, merge_runs, value_duplicates

    cands = candidates(data, args.ticker, reports, hist, xbrl, materiality=args.materiality, ledger=ledger)
    print(f"{len(cands)} unmapped sources above {args.materiality:.0%} materiality" + ("" if xbrl is not None else " (no full-filing XBRL facts on disk; run ingest with v0.7+ to include custom concepts)"))
    if not cands:
        return
    sections = load_sections(data, args.ticker)
    if not sections.get("Item 8 Notes"):
        print("note: no Item 8 notes text on disk; citations will come from Items 1/1A/7 only. Re-run `crucible ingest TICKER` (v0.8+) to add the notes.", file=sys.stderr)
    for c in cands:
        c["context"] = context_sentences(sections, c["source"], c.get("label", ""))
    print(f"{sum(1 for c in cands if c['context'])} of {len(cands)} candidates have filing sentences to cite", file=sys.stderr)
    accession = ((load_meta(data, args.ticker).get("sources_by_fiscal_year") or {}).get(str(hist.columns[-1])) or "latest 10-K").split(" filed")[0]
    llm = MockMapper() if (args.provider or "mock") == "mock" else get_llm(args.provider, getattr(args, "model", None))
    runs = [suggest(cands[: args.max_candidates], llm, args.ticker.upper(), seed=r * 1000, batch=args.batch) for r in range(max(1, args.runs))]
    props, agreement = merge_runs(runs)
    if not args.no_confirm:
        from .mapsuggest import confirm_citations

        cc = confirm_citations(props, sections, llm)
        print(f"citation check: {cc['checked']} proposals without a model-chosen citation re-offered wider context; {cc['confirmed']} confirmed by the model, "
              f"{cc['declined']} left without a citation ({cc['calls']} calls)", file=sys.stderr)
    vd = value_duplicates(cands, hist)
    n = write_pending(ledger, args.ticker, props, getattr(llm, "name", "?"), accession=accession, val_dups=vd)
    ledger.save()
    if args.runs > 1:
        print(f"agreement across {agreement['runs']} runs: {agreement['agree']}/{agreement['sources']} sources ({agreement['agreement_rate']:.0%}); "
              f"disagreements keep the stronger target and carry the other as an alternative")
    print(f"{n} proposals written as pending; review with `crucible map-export {args.ticker}` or decide with `crucible map-approve`")
    for p in props:
        alts = ", ".join(f"{a['target']} {a['confidence']:.2f}" for a in p["alternatives"]) or "none"
        flag = "  DUP" if (p.get("duplicates") or vd.get(p["source"])) else ""
        print(f"  {p['statement']} {p['source'][:58]:58s} -> {p['target']:20s} conf {p['confidence']:.2f}  alts: {alts}  ({p['share_of_base']:.1%}){flag}")


def cmd_map_approve(args) -> None:
    ledger = _ledger(args)
    n = 0
    if args.file:  # decisions made in the exported Excel
        n = ledger.from_excel(Path(args.file), decided_by=args.analyst_name)
    elif not args.source and args.accept_above is None:
        pend = len(ledger.rows(args.ticker, status="pending"))
        raise SystemExit(f"nothing to decide: {pend} pending row(s). Use one of:\n"
                         f"  crucible map-approve {args.ticker} --file outputs/{args.ticker.upper()}_mapping_ledger.xlsx   (after editing status/target in Excel)\n"
                         f"  crucible map-approve {args.ticker} --accept-above 0.8\n"
                         f"  crucible map-approve {args.ticker} --source \"<caption or tag>\" --statement IS --target cogs [--note '...']\n"
                         f"  crucible map-approve {args.ticker} --source \"<caption or tag>\" --statement IS --reject")
    if args.source:
        if args.target and args.target not in ("residual", "ignore"):
            from .schema import ITEM_BY_KEY

            if args.target not in ITEM_BY_KEY:
                raise SystemExit(f"unknown target {args.target}; valid: {', '.join(ITEM_BY_KEY)}, residual, ignore")
            if args.statement and ITEM_BY_KEY[args.target].statement != args.statement:
                raise SystemExit(f"{args.target} is a {ITEM_BY_KEY[args.target].statement} line; it cannot be the target of a {args.statement} source")
        n = ledger.decide(args.ticker, args.source, "rejected" if args.reject else "accepted", target=args.target, decided_by=args.analyst_name,
                          note=args.note or "", statement=args.statement)
    elif args.accept_above is not None:
        pend = ledger.rows(args.ticker, status="pending")
        for _, r in pend.iterrows():
            if float(r.confidence or 0) >= args.accept_above:
                n += ledger.decide(args.ticker, r.source, "accepted", decided_by=args.analyst_name, note=f"auto-accepted above {args.accept_above}", statement=r.statement)
    ledger.save()
    print(f"{n} ledger row(s) updated")
    print(ledger.summary(args.ticker))


def cmd_map_export(args) -> None:
    out = Path(args.out or f"outputs/{args.ticker.upper()}_mapping_ledger.xlsx")
    _ledger(args).to_excel(out, args.ticker)
    print(f"wrote {out} (edit target/status/note, then: crucible map-import {args.ticker} {out})")


def cmd_map_import(args) -> None:
    ledger = _ledger(args)
    path = Path(args.file) if args.file else Path(f"outputs/{args.ticker.upper()}_mapping_ledger.xlsx")
    n = ledger.from_excel(path, decided_by=args.analyst_name)
    args.file = str(path)
    ledger.save()
    print(f"{n} row(s) applied from {args.file}")
    print(ledger.summary(args.ticker))


# ----------------------------------------------------------------------------- decisions (steps 1-5 of the research process)

def _dossier(args):
    """Shared evidence packet for the company; cached as outputs/<T>_dossier.json keyed by the inputs."""
    from .dossier import build_dossier
    from .ingest import load_eightk, load_meta, load_sections
    from .mapping import load_history

    data = Path(args.data)
    hist, _ = load_history(data, args.ticker, as_of=getattr(args, "as_of", None))
    long_hist, _ = load_history(data, args.ticker, as_of=getattr(args, "as_of", None), hist_years=20)
    meta = load_meta(data, args.ticker)
    sections = load_sections(data, args.ticker, as_of=getattr(args, "as_of", None))
    peers = {}
    for p in _peers(args):
        try:
            peers[p], _ = load_history(data, p, as_of=getattr(args, "as_of", None))
        except Exception:
            pass
    pk = build_dossier(data, args.ticker, hist, long_hist, sections, load_eightk(data, args.ticker, as_of=getattr(args, "as_of", None), n=2),
                       load_eightk(data, args.ticker, as_of=getattr(args, "as_of", None), n=100), peers=peers, company=meta.get("name") or args.ticker.upper(),
                       as_of=getattr(args, "as_of", None), with_macro=not getattr(args, "no_macro", False))
    out = Path(f"outputs/{args.ticker.upper()}_dossier.json")
    pk.save(out)
    print(f"dossier {pk.hash()} with {len(pk.items)} items -> {out}", file=sys.stderr)
    args._meta = meta
    return pk, sections, hist


def _profile(args, pk, sections, llm=None):
    """The company profile: cached in data/<T>/profile.json, built with the model on first use (or --reprofile)."""
    from .profile import build_profile, load_profile, save_profile

    data = Path(args.data)
    prof = None if getattr(args, "reprofile", False) else load_profile(data, args.ticker)
    if prof is None:
        llm = llm or get_llm(args.provider, getattr(args, "model", None))
        prof = build_profile(pk, sections, getattr(args, "_meta", {}) or {}, llm, seed=getattr(args, "seed", 1))
        save_profile(data, args.ticker, prof)
    return prof


def cmd_profile(args) -> None:
    """Industry from the filing header, business model and drivers from the documents, with verified quotes."""
    from .profile import profile_markdown

    pk, sections, _ = _dossier(args)
    prof = _profile(args, pk, sections)
    md = profile_markdown(prof)
    Path(f"outputs/{args.ticker.upper()}_profile.md").write_text(md)
    print(md)


def cmd_dossier(args) -> None:
    pk, _, _ = _dossier(args)
    for it in pk.items:
        print(it.render()[:220])


def cmd_choose(args) -> None:
    """Run the choice debates (all, or --keys ...) and write every option to the decision ledger as pending."""
    from .decisions import DecisionLedger, decide
    from .dossier import decision_specs

    pk, sections, hist = _dossier(args)
    llm = get_llm(args.provider, getattr(args, "model", None))
    prof = _profile(args, pk, sections, llm)
    fam = prof.get("family") or {}
    print(f"industry: {prof.get('industry_filing') or 'n/a'} (SIC {prof.get('sic')}, filing header); family {fam.get('label')}; model reads: {prof.get('industry')}; "
          f"{len(prof.get('kpis') or [])} KPIs, {len(prof.get('template_candidates') or [])} template candidates, {prof.get('dropped', 0)} unverifiable quotes dropped", file=sys.stderr)
    specs = decision_specs(sections, _peers(args), pk.company, profile=prof)
    if args.keys:
        specs = [sp for sp in specs if sp.key in set(args.keys)]
    ledger = DecisionLedger(Path(args.data))
    cache = Path(args.logs) / "cache"
    calls = sum(((max(1, len(sp.options) - 1) if (sp.tournament or len(sp.options) <= 2) else 1) if args.cheap else 3 * max(1, len(sp.options) - 1)) for sp in specs)
    print(f"{len(specs)} decisions, ~{calls} model calls ({'cheap: judge ranks directly, tournaments judge-only' if args.cheap else 'advocates + judge per pairing'}), cached under {cache}", file=sys.stderr)
    log = RunLog(Path(args.logs))
    decided_before = ledger.rows(args.ticker)
    decided_before = set(decided_before[decided_before.decided_by.astype(str) != ""].statement) if len(decided_before) else set()
    for sp in specs:
        res = decide(pk, sp, llm, seed=args.seed, cheap=args.cheap, cache_dir=cache, progress=lambda m: print(m, file=sys.stderr))
        if sp.key in decided_before and not args.reset:
            print(f"{sp.key}: rows the analyst already decided are kept (pass --reset to replace them with this run)", file=sys.stderr)
        n = ledger.record(args.ticker, res, pk, sp, cheap=args.cheap, reset=args.reset)
        log._append("decisions_runs.jsonl", {"ticker": args.ticker.upper(), **res.model_dump()})
        top = res.ranking[0]
        print(f"{sp.key:22s} -> {top.label} (conf {top.confidence:.2f}); " + ", ".join(f"{v.label} {v.confidence:.2f}" for v in res.ranking[1:]))
        if top.questions:
            print(f"{'':22s}    settle: {top.questions[0]}")
        if all(v.confidence == 0 for v in res.ranking):
            print(f"{'':22s}    WARNING: {top.reasoning}", file=sys.stderr)
    ledger.save()
    print(f"\n{ledger.summary(args.ticker)}")
    print(f"next: crucible review {args.ticker} --what decisions   (or map-export style: crucible export {args.ticker})")


def _ask_fn(args, pk=None, sections=None):
    """Closure answering follow-up questions from the filings for this ticker (dossier packet plus sections)."""
    from .ask import ask

    if getattr(args, "provider", None) in (None, "", "none"):
        return None
    llm = get_llm(args.provider, getattr(args, "model", None))
    if pk is None or sections is None:
        try:
            pk, sections, _ = _dossier(args)
        except Exception:
            from .ingest import load_sections

            sections = load_sections(Path(args.data), args.ticker)
            pk = None
    from .state import state_items

    try:
        state = state_items(Path(args.data), args.ticker)
    except Exception:
        state = []
    return lambda row, q: ask(q, row, pk, sections, llm, log_dir=Path(args.logs), ticker=args.ticker.upper(), state=state)


def cmd_review(args) -> None:
    """Terminal review of pending rows: mapping proposals, decisions, or both. '?question' asks the filings."""
    from .decisions import DecisionLedger, review_terminal

    what = ["mapping", "decisions"] if args.what == "both" else [args.what]
    ask_fn = _ask_fn(args) if args.provider != "none" else None
    for w in what:
        ledger = _ledger(args) if w == "mapping" else DecisionLedger(Path(args.data))
        dec = None
        if args.decisions_file:
            dec = {k: tuple(v) for k, v in json.loads(Path(args.decisions_file).read_text()).items()}
        counts = review_terminal(ledger, args.ticker, statement=args.statement, analyst=args.analyst_name, decisions=dec, ask_fn=ask_fn)
        ledger.save()
        print(f"{w}: {counts}")
    if args.export:
        cmd_export(args)


def cmd_review_export(args) -> None:
    """Everything pending, with every field, in one workbook the analyst can decide in and bring back."""
    from .decisions import DecisionLedger, export_review

    out = export_review(Path(args.out or f"outputs/{args.ticker.upper()}_review.xlsx"), args.ticker, _ledger(args), DecisionLedger(Path(args.data)),
                        include_decided=args.include_decided)
    print(f"wrote {out}: fill DECISION (accepted / rejected / unsure), TARGET_OVERRIDE, NOTE and FOLLOW_UP_QUESTION, then `crucible review-import {args.ticker.upper()} {out}`")


def cmd_review_import(args) -> None:
    """Apply the decisions in a review workbook; answer its follow-up questions with citations and confidence."""
    from .decisions import DecisionLedger, import_review

    ask_fn = _ask_fn(args) if args.provider != "none" else None
    res = import_review(Path(args.file), args.ticker, _ledger(args), DecisionLedger(Path(args.data)), analyst=args.analyst_name, ask_fn=ask_fn)
    answers = res.pop("answers", [])
    print(json.dumps(res, indent=2))
    for a in answers:
        print(f"\n[{a['sheet']}] {a['source']}\n  Q: {a['question']}\n  A ({a['confidence']:.2f}): {a['answer']}")
        for c in a.get("citations", []):
            print(f"    - {c['source']} | {c['text'][:160]}")


def cmd_ask(args) -> None:
    """Ask the filings a question; the answer cites the sentences it rests on and carries a confidence."""
    from .ask import format_answer

    fn = _ask_fn(args)
    if fn is None:
        raise SystemExit("pass --provider anthropic (or mock)")
    row = {}
    if args.row:
        d = _ledger(args).rows(args.ticker)
        m = d[d.source == args.row]
        if len(m):
            row = m.iloc[0].to_dict()
    print(format_answer(fn(row, args.question)))


def cmd_brief(args) -> None:
    from .brief import brief_markdown, write_brief

    pk, _, _ = _dossier(args)
    llm = get_llm(args.provider, getattr(args, "model", None))
    b = write_brief(pk, llm, seed=args.seed)
    out = Path(f"outputs/{args.ticker.upper()}_brief.json")
    out.write_text(json.dumps(b, indent=2))
    Path(f"outputs/{args.ticker.upper()}_brief.md").write_text(brief_markdown(b))
    print(brief_markdown(b))
    print(f"saved {out}")


def cmd_export(args) -> None:
    """Final workbook (mapping ledger, decision ledger, brief) plus the unsure workbook."""
    from .decisions import DecisionLedger, export_workbook

    bp = Path(f"outputs/{args.ticker.upper()}_brief.json")
    brief = json.loads(bp.read_text()) if bp.exists() else None
    final, unsure = export_workbook(Path(getattr(args, "out", None) or f"outputs/{args.ticker.upper()}_final.xlsx"), args.ticker, _ledger(args),
                                    DecisionLedger(Path(args.data)), brief)
    print(f"wrote {final}" + (f" and {unsure}" if unsure else ""))


def cmd_set_peers(args) -> None:
    """Record the chosen peer tickers (after the peer_set decision) so ingest and every later command use them."""
    pf = Path(args.data) / args.ticker.upper() / "peers.json"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps([t.upper() for t in args.tickers]))
    print(f"peer set for {args.ticker.upper()}: {', '.join(t.upper() for t in args.tickers)} -> {pf}\nnext: python -m crucible ingest {args.ticker} --peers {' '.join(t.upper() for t in args.tickers)}")


def cmd_attribution_fit(args) -> None:
    """Regression on the runs already logged for this packet family: no API calls."""
    from .attribution import attribution_from_logs, attribution_table

    pk = _packet(args)
    runs = [r for r in RunLog(Path(args.logs)).read("runs.jsonl")
            if r.get("assumption") == args.assumption and r.get("company") == pk.company and set(r.get("evidence_ids", [])) <= set(pk.ids())]
    fit = attribution_from_logs(runs, pk)
    if "error" in fit:
        raise SystemExit(fit["error"])
    print(f"{fit['n_runs']} logged runs share this packet's evidence ids")
    print("\nOLS: base on group inclusion (observational)")
    print(attribution_table(fit["ols_base"]))
    if "logistic_hit" in fit:
        print("\nLogistic: range covered the realized outcome (backtest runs)")
        for r in fit["logistic_hit"]["rows"]:
            print(f"  {r['group'][:34]:34s} log-odds {r['log_odds']:+.3f}" if r["log_odds"] == r["log_odds"] else f"  {r['group'][:34]:34s} n/a")
        print(f"  hit rate {fit['logistic_hit']['hit_rate']}  McFadden R2 {fit['logistic_hit']['mcfadden_r2']}")
        print("\nOLS: realized minus base on group inclusion (positive = group pushed the base toward reality)")
        print(attribution_table(fit["ols_error"]))
    out = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_attribution_from_logs.json")
    out.write_text(json.dumps(fit, indent=2))
    print(f"saved {out}")


def cmd_quarterly(args) -> None:
    """Quarterly income statement history (Q4 = FY minus nine months) to outputs/<T>_quarterly.csv."""
    from .facts import load_facts, load_xbrl_facts, quarterly_from_facts
    from .schema import ITEM_BY_KEY

    facts = load_facts(Path(args.data), args.ticker)
    if facts is None:
        raise SystemExit("no facts on disk; run ingest")
    extra = load_xbrl_facts(Path(args.data), args.ticker)
    if extra is not None and len(extra):
        facts = pd.concat([facts, extra[~extra.concept.isin(set(facts.concept))]], ignore_index=True)
    ledger = _ledger(args)
    links, rejects = {}, set()
    for st in ("IS", "BS", "CF"):
        for src, tgt in ledger.decided_links(args.ticker, st, "concept").items():
            if tgt in ITEM_BY_KEY:
                links[src.lower()] = tgt
        rejects |= {x.lower() for x in ledger.decided_rejects(args.ticker, st, "concept")}
    q = quarterly_from_facts(facts, as_of=args.as_of, concept_links=links, concept_rejects=rejects)
    out = Path(f"outputs/{args.ticker.upper()}_quarterly.csv")
    q.to_csv(out)
    pd.set_option("display.width", 220)
    print(q.iloc[:, -8:].to_string())
    if len(q.columns):
        last_fy = q.columns[-1][:6]
        gaps = [k for k in q.index if q.loc[k, [c for c in q.columns if c.startswith(last_fy)]].isna().any()]
        if gaps:
            print(f"note: {', '.join(gaps)} incomplete for {last_fy}: the quarterly filings lack a rule concept and not every linked component is present; "
                  f"`ingest {args.ticker.upper()}` now saves 10-Q full-filing XBRL facts (custom concepts) which fills this")
    print(f"saved {out}")


def cmd_dates(args) -> None:
    """List filings on disk and the as-of dates a backtest can use."""
    from .ingest import load_eightk, load_meta

    data = Path(args.data)
    meta = load_meta(data, args.ticker)
    tenks = _tenk_dates(data, args.ticker)
    d = data / args.ticker.upper()
    rels = load_eightk(data, args.ticker, n=100)
    print(f"{args.ticker.upper()}: {meta.get('name')}  (fiscal year end {meta.get('fiscal_year_end')})")
    print(f"10-K text sections on disk: {sorted(p.name for p in d.glob('tenk_sections_*.json'))}")
    print(f"earnings releases on disk: {len(rels)}  ({rels[-1]['filing_date'] if rels else '-'} .. {rels[0]['filing_date'] if rels else '-'})")
    forms = {}
    for f in meta.get("filings", []):
        forms.setdefault(f.get("form"), []).append(f.get("filing_date"))
    for form, dates in forms.items():
        print(f"{form}: {len(dates)} filings, {min(dates)} .. {max(dates)}")
    print("\nbacktest as-of dates (10-K filing dates) and realized horizon available today:")
    for t in tenks:
        tag = "full 3-year" if t["years_realized"] >= 3 else (f"partial: {t['years_realized']} year(s)" if t["years_realized"] > 0 else "nothing realized yet")
        print(f"  FY{t['fy']} 10-K filed {t['filing_date']}  ->  --as-of {t['as_of']}   [{tag}]")


def _realized(full: pd.DataFrame, last: int, horizon: int = 3) -> tuple[float | None, int]:
    """Realized CAGR from FY``last`` over the years available (up to ``horizon``); returns (cagr, years_used)."""
    years = [y for y in range(last + 1, last + horizon + 1) if y in full.columns]
    if not years:
        return None, 0
    n = len(years)
    rev0, revn = float(full.loc["revenue", last]), float(full.loc["revenue", years[-1]])
    return ((revn / rev0) ** (1 / n) - 1) if rev0 > 0 else None, n


def _run_backtest_point(args, as_of: str, data: Path, full: pd.DataFrame, log: RunLog, llm, cache, role_llms) -> dict | None:
    """One as-of date: build the dated packet, run both arms, score against what was realized so far."""
    import statistics

    from .ingest import load_eightk, load_meta, load_sections
    from .mapping import load_history

    hist, _ = load_history(data, args.ticker, as_of=as_of, hist_years=args.hist_years)
    long_hist, _ = load_history(data, args.ticker, as_of=as_of, hist_years=20)
    last = int(hist.columns[-1])
    realized, n_years = _realized(full, last)
    if realized is None:
        print(f"as-of {as_of}: history through FY{last}, nothing realized yet; skipped", file=sys.stderr)
        return None
    meta = load_meta(data, args.ticker)
    peers = {}
    for p in _peers(args):
        try:
            peers[p.upper()], _ = load_history(data, p, as_of=as_of, hist_years=args.hist_years)
        except Exception as e:
            print(f"warning: peer {p} as of {as_of}: {e}", file=sys.stderr)
    sections = load_sections(data, args.ticker, as_of=as_of)
    releases = load_eightk(data, args.ticker, as_of=as_of, n=args.releases)
    pk = build_packet(args.assumption, meta.get("name") or args.ticker.upper(), hist, peers=peers, sections=sections,
                      analyst=_analyst(args), as_of=as_of, releases=releases, all_releases=load_eightk(data, args.ticker, as_of=as_of, n=100),
                      long_hist=long_hist, kpis=_kpis(args))
    pk.save(Path(f"outputs/{args.ticker.upper()}_{args.assumption}_asof{as_of}_packet.json"))
    growth = hist.loc["revenue"].pct_change().dropna()
    vol = float(growth.std(ddof=0)) if len(growth) > 1 else 0.0
    tag = "full 3-year" if n_years >= 3 else f"partial {n_years}-year"
    print(f"as-of {as_of}: modeling window FY{hist.columns[0]}-FY{hist.columns[-1]} (all years known then: FY{long_hist.columns[0]}-FY{last}); "
          f"{len(pk.items)} items ({len(releases)} releases, {sum(1 for i in pk.items if i.source.startswith('derived: guidance'))} guidance items); "
          f"scored on FY{last + 1}-FY{last + n_years}: realized {tag} CAGR {realized:.1%}; trailing growth std {vol:.1%}")
    st = stability(pk, llm, n=args.runs, rounds=args.rounds, temperature=args.temperature, policy=args.policy, seed0=args.seed,
                   workers=args.workers, cache=cache, role_llms=role_llms, label=f"backtest {as_of}")
    for res in st.results:
        log.record_run(res, tag=f"backtest asof {as_of}", extra={"realized": round(realized, 4), "as_of": as_of})
    hits = [r.verdict.low <= realized <= r.verdict.high for r in st.results]
    widened_hits = [(r.verdict.base - max((r.verdict.high - r.verdict.low) / 2, vol)) <= realized <= (r.verdict.base + max((r.verdict.high - r.verdict.low) / 2, vol)) for r in st.results]
    row = {"as_of": as_of, "last_fy": last, "years_realized": n_years, "realized": round(realized, 4),
           "debate_base": round(st.base_median, 4), "debate_low": round(statistics.median(st.lows), 4), "debate_high": round(statistics.median(st.highs), 4),
           "debate_error": round(st.base_median - realized, 4), "debate_hit_rate": round(sum(hits) / len(hits), 2),
           "debate_hit_rate_vol_floor": round(sum(widened_hits) / len(widened_hits), 2), "trailing_growth_std": round(vol, 4),
           "debate_width": round(st.mean_width, 4), "debate_confidence": round(statistics.mean(st.confidences), 2)}
    if args.vote_n > 0:
        vt = vote(pk, llm, n=args.vote_n, temperature=args.temperature, seed0=args.seed, workers=args.workers, cache=cache)
        vhits = [e.low <= realized <= e.high for e in vt.estimates]
        row.update({"vote_base": round(vt.base_median, 4), "vote_error": round(vt.base_median - realized, 4),
                    "vote_hit_rate": round(sum(vhits) / len(vhits), 2), "vote_width": round(vt.mean_width, 4),
                    "debate_beats_vote": abs(st.base_median - realized) < abs(vt.base_median - realized)})
    log.record_outcome(args.ticker.upper(), args.assumption, f"FY{last + 1}-FY{last + n_years}", realized, f"10-K revenue FY{last} and FY{last + n_years}")
    print("  " + json.dumps(row))
    return row


def cmd_backtest(args) -> None:
    """Packet as of a past date (or every 10-K date with --sweep), debates plus the vote control, scored against what the company then reported."""
    from .mapping import load_history

    data = Path(args.data)
    full, _ = load_history(data, args.ticker, hist_years=15)
    llm = get_llm(args.provider, getattr(args, "model", None))
    log = RunLog(Path(args.logs))
    cache, role_llms = _cache(args), _role_llms(args, llm)
    if args.sweep:
        points = [t for t in _tenk_dates(data, args.ticker) if t["years_realized"] > 0]
        dates = [t["as_of"] for t in points][-args.sweep:]
        print(f"sweep over {len(dates)} as-of dates: {dates}  (~{len(dates) * (args.runs * (2 * args.rounds + 1) + args.vote_n)} calls if nothing is cached)")
    elif args.as_of:
        dates = [args.as_of]
    else:
        raise SystemExit("give --as-of YYYY-MM-DD or --sweep N")
    rows = [r for d in dates if (r := _run_backtest_point(args, d, data, full, log, llm, cache, role_llms))]
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = Path(f"outputs/{args.ticker.upper()}_{args.assumption}_backtest.csv")
    df.to_csv(out, index=False)
    summ = {"points": len(rows), "debate_hit_rate": round(df.debate_hit_rate.mean(), 2), "debate_hit_rate_vol_floor": round(df.debate_hit_rate_vol_floor.mean(), 2),
            "debate_mean_abs_error": round(df.debate_error.abs().mean(), 4), "debate_mean_error": round(df.debate_error.mean(), 4)}
    if "vote_hit_rate" in df:
        summ.update({"vote_hit_rate": round(df.vote_hit_rate.mean(), 2), "vote_mean_abs_error": round(df.vote_error.abs().mean(), 4),
                     "debate_beats_vote_share": round(df.debate_beats_vote.mean(), 2)})
    errs = (df.realized - df.debate_base).tolist()  # realized minus base: positive = the tool was too low
    cal = {"ticker": args.ticker.upper(), "assumption": args.assumption, "n_points": len(errs), "as_of_dates": df.as_of.tolist(),
           "error_min": round(min(errs), 4), "error_median": round(float(pd.Series(errs).median()), 4), "error_max": round(max(errs), 4),
           "mean_debate_half_width": round(float((df.debate_high - df.debate_low).mean() / 2), 4),
           "note": "empirical adjustment = base + [error_min, error_max] from these backtests; shown beside the advisory range as a warning, not a replacement"}
    Path(f"outputs/{args.ticker.upper()}_{args.assumption}_calibration.json").write_text(json.dumps(cal, indent=2))
    summ["empirical_error_range"] = [cal["error_min"], cal["error_max"]]
    print("\nCALIBRATION", json.dumps(summ), f"\nsaved {out} and outputs/{args.ticker.upper()}_{args.assumption}_calibration.json")


def cmd_demo(args) -> None:
    """Whole loop offline: synthetic company, mock LLM. Proves the plumbing; numbers are invented."""
    args.ticker, args.fixture, args.data, args.years = "SYNTH", True, "data", 5
    args.as_of, args.hist_years = None, 5
    args.drivers, args.wacc, args.g, args.mid_year, args.out = None, None, 0.025, False, "outputs/SYNTH_model.xlsx"
    args.nwc_method, args.audit_links, args.no_dilution, args.template, args.template_inputs = "days", False, True, "none", None
    cmd_model(args)
    for a in ("revenue_growth_3y", "wacc"):
        args.assumption, args.analyst, args.packet, args.peers, args.releases = a, None, None, [], 2
        args.out = None
        cmd_packet(args)
        args.provider, args.model, args.rounds, args.seed, args.temperature, args.policy, args.logs = "mock", None, 2, 1, 0.7, "calibrated", "logs"
        args.bull_model = args.bear_model = args.judge_model = None; args.workers, args.no_cache = 2, True
        cmd_debate(args)
        args.runs, args.loo_k, args.tol, args.loo_mode, args.vote_n, args.ablation, args.random_k = 4, 2, 0.01, "items", 3, "loo", 8
        cmd_harness(args)
    args.action, args.value, args.reason, args.analyst_name = "edit", 0.14, "demo: haircut for backlog conversion risk", "demo"
    args.assumption = "revenue_growth_3y"
    cmd_decide(args)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="crucible", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="pull filings with edgartools (needs internet)")
    s.add_argument("ticker"); s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--data", default="data")
    s.add_argument("--years", type=int, default=5); s.add_argument("--quarters", type=int, default=8)
    s.add_argument("--no-text", action="store_true"); s.add_argument("--no-pit", action="store_true")
    s.add_argument("--no-eightk", action="store_true", help="skip 8-K earnings releases")
    s.add_argument("--no-xbrl", action="store_true", help="skip full-filing XBRL facts (custom concepts)")
    s.add_argument("--only-eightk", action="store_true", help="only fetch 8-K earnings releases (statements, text and facts untouched)")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("model", help="build the Excel model")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--fixture", action="store_true")
    s.add_argument("--as-of", help="build the history as known on this date (YYYY-MM-DD), for backtests")
    s.add_argument("--hist-years", type=int, default=5)
    s.add_argument("--years", type=int, default=5); s.add_argument("--drivers", help="JSON overrides for Drivers")
    s.add_argument("--wacc", help="JSON for WACCInputs"); s.add_argument("--g", type=float, default=0.025)
    s.add_argument("--mid-year", action="store_true"); s.add_argument("--out")
    s.add_argument("--nwc-method", choices=["days", "pct_revenue"], default="days", help="working capital forecast: DSO/DIO/DPO days or ratios to revenue")
    s.add_argument("--audit-links", action="store_true", help="add an Audit sheet listing every check row's cells and formulas (default off)")
    s.add_argument("--no-dilution", action="store_true", help="use the last 10-K diluted count instead of the treasury-stock-method count at the as-of price")
    s.add_argument("--template", default="auto", help="industry driver template: auto (the accepted industry_template decision), none, or a key such as bitcoin_miner")
    s.add_argument("--template-inputs", help="JSON overriding template driver paths (a number applies to every year, a list per year)")
    s.set_defaults(fn=cmd_model)

    s = sub.add_parser("kpis", help="fit the industry template's KPI table from the earnings releases (hashrate, blocks, BTC produced, cost per BTC, prices, holdings)")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--template", default="bitcoin_miner"); s.add_argument("--cites", action="store_true")
    s.set_defaults(fn=cmd_kpis)

    for name, fn in (("packet", cmd_packet), ("debate", cmd_debate), ("harness", cmd_harness)):
        s = sub.add_parser(name)
        s.add_argument("ticker"); s.add_argument("--assumption", choices=list(ASSUMPTIONS), required=True)
        s.add_argument("--data", default="data"); s.add_argument("--fixture", action="store_true")
        s.add_argument("--as-of", help="history as known on this date (YYYY-MM-DD)"); s.add_argument("--hist-years", type=int, default=5)
        s.add_argument("--releases", type=int, default=2, help="how many recent earnings releases to draw evidence from")
        s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--analyst", help="JSON with risk_free/erp/beta/notes")
        s.add_argument("--packet", help="existing packet JSON"); s.add_argument("--out")
        s.add_argument("--provider", default=None, help="mock | anthropic | openai (default: $CRUCIBLE_PROVIDER or mock)")
        s.add_argument("--model", default=None, help="model id for the provider (default: $CRUCIBLE_MODEL, else claude-sonnet-5 for anthropic)")
        s.add_argument("--rounds", type=int, default=2); s.add_argument("--seed", type=int, default=1)
        s.add_argument("--temperature", type=float, default=0.7)
        s.add_argument("--policy", choices=["stubborn", "calibrated", "agreeable"], default="calibrated")
        s.add_argument("--logs", default="logs")
        s.add_argument("--bull-model"); s.add_argument("--bear-model"); s.add_argument("--judge-model", help="provider:model specs for heterogeneity")
        s.add_argument("--workers", type=int, default=4, help="concurrent debates"); s.add_argument("--no-cache", action="store_true")
        if name == "harness":
            s.add_argument("--runs", type=int, default=5); s.add_argument("--loo-k", type=int, default=2)
            s.add_argument("--loo-mode", choices=["groups", "items"], default="groups", help="ablate evidence groups (default) or single items")
            s.add_argument("--vote-n", type=int, default=5, help="control arm: N independent estimates (1 call each); 0 disables")
            s.add_argument("--ablation", choices=["loo", "random", "none"], default="loo", help="loo = one debate per removed group; random = random subsets + regression")
            s.add_argument("--random-k", type=int, default=16, help="debates for the random-subset design")
            s.add_argument("--tol", type=float, default=0.01, help="agreement tolerance, absolute decimal (0.01 = 1pp)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("inputs", help="derive rf (FRED), beta (Yahoo regression), ERP, cost of debt, debt weight, tax rate; cache by date; overrides and notes")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--peers", nargs="*", default=[])
    s.add_argument("--as-of", help="derive as of this date (YYYY-MM-DD)"); s.add_argument("--refresh", action="store_true")
    s.add_argument("--refresh-days", type=int, default=7)
    s.add_argument("--erp", type=float, help="set the equity risk premium (decimal) when Damodaran is unreachable or you hold a house view")
    s.add_argument("--erp-source", help="citation for --erp, e.g. 'Damodaran implied ERP 2026-09-01'")
    s.add_argument("--erp-method", choices=["hist30y", "trend20y", "hist10y", "hist30d"], default="hist30y",
                   help="expected market return: trailing 30y total return (default), 20y log-trend drift, trailing 10y (recency-biased), trailing 30d (noisy)")
    s.add_argument("--set", nargs="*", help="override a derived field: --set beta=1.25 --reason '...'"); s.add_argument("--reason")
    s.add_argument("--note", nargs="*", help="analyst notes the debate will see as evidence"); s.add_argument("--clear-overrides", action="store_true")
    s.set_defaults(fn=cmd_inputs)

    s = sub.add_parser("map-status", help="record rule matches and show the mapping ledger for a company")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.set_defaults(fn=cmd_map_status)
    s = sub.add_parser("map-suggest", help="LLM proposals for unmapped lines above materiality -> ledger (pending)")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--provider", default=None); s.add_argument("--model", default=None)
    s.add_argument("--materiality", type=float, default=0.01, help="share of revenue (IS/CF) or total assets (BS)")
    s.add_argument("--max-candidates", type=int, default=200); s.add_argument("--batch", type=int, default=15)
    s.add_argument("--runs", type=int, default=1, help="repeat the proposer N times and measure agreement per source (default 1)")
    s.add_argument("--no-confirm", action="store_true", help="skip the model citation-confirmation pass (citations then stay keyword-matched)")
    s.set_defaults(fn=cmd_map_suggest)
    s = sub.add_parser("map-approve", help="accept/reject a ledger row, or accept all pending above a confidence")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--source"); s.add_argument("--target"); s.add_argument("--statement")
    s.add_argument("--reject", action="store_true"); s.add_argument("--accept-above", type=float); s.add_argument("--note"); s.add_argument("--analyst-name", default="analyst")
    s.add_argument("--file", help="apply decisions from an exported Excel ledger (same as map-import)")
    s.set_defaults(fn=cmd_map_approve)
    s = sub.add_parser("map-export", help="export the company's ledger to Excel for review")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--out"); s.set_defaults(fn=cmd_map_export)
    s = sub.add_parser("map-import", help="apply analyst edits from an exported Excel ledger")
    s.add_argument("ticker"); s.add_argument("file", nargs="?"); s.add_argument("--data", default="data"); s.add_argument("--analyst-name", default="analyst (excel)")
    s.set_defaults(fn=cmd_map_import)

    def _common(sp):
        sp.add_argument("ticker"); sp.add_argument("--data", default="data"); sp.add_argument("--logs", default="logs"); sp.add_argument("--peers", nargs="*", default=[])
        sp.add_argument("--as-of"); sp.add_argument("--no-macro", action="store_true", help="skip FRED macro features")
        return sp

    s = _common(sub.add_parser("dossier", help="build and print the company evidence dossier (shared by decisions and the brief)"))
    s.set_defaults(fn=cmd_dossier)
    s = _common(sub.add_parser("choose", help="run the choice debates (segments, KPIs, peers, pricing power, cycle, alignment, methods, valuation) -> decision ledger"))
    s.add_argument("--keys", nargs="*", help="subset of decision keys"); s.add_argument("--provider", default="mock"); s.add_argument("--model")
    s.add_argument("--seed", type=int, default=1); s.add_argument("--cheap", action="store_true", help="judge ranks directly, no advocates (1 call per decision)")
    s.add_argument("--reset", action="store_true", help="replace rows the analyst already decided for the chosen decisions (rerun after new evidence)")
    s.add_argument("--reprofile", action="store_true", help="rebuild the company profile instead of using data/<T>/profile.json")
    s.set_defaults(fn=cmd_choose)
    s = _common(sub.add_parser("profile", help="industry from the filing header (SIC) and the drivers, KPIs and templates the model reads from Item 1 and the releases, quote-verified"))
    s.add_argument("--provider", default="mock"); s.add_argument("--model"); s.add_argument("--seed", type=int, default=1); s.add_argument("--reprofile", action="store_true")
    s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("review", help="terminal review of pending rows: accept / reject / unsure / retarget, with notes")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--what", choices=["mapping", "decisions", "both"], default="both")
    s.add_argument("--statement", help="restrict to one statement or decision key"); s.add_argument("--analyst-name", default="analyst")
    s.add_argument("--decisions-file", help="JSON {source: [status, target, note]} for non-interactive review"); s.add_argument("--export", action="store_true")
    s.add_argument("--out"); s.add_argument("--provider", default="none", help="model for '?question' follow-ups during review (anthropic, mock, none)")
    s.add_argument("--model"); s.add_argument("--logs", default="logs"); s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--as-of")
    s.add_argument("--no-macro", action="store_true"); s.set_defaults(fn=cmd_review)
    s = sub.add_parser("review-export", help="workbook of everything pending with every field; decide offline, then review-import")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--out"); s.add_argument("--include-decided", action="store_true")
    s.set_defaults(fn=cmd_review_export)
    s = sub.add_parser("review-import", help="apply DECISION / TARGET_OVERRIDE / NOTE from a review workbook; answer its FOLLOW_UP_QUESTION column")
    s.add_argument("ticker"); s.add_argument("file"); s.add_argument("--data", default="data"); s.add_argument("--logs", default="logs"); s.add_argument("--analyst-name", default="analyst")
    s.add_argument("--provider", default="none"); s.add_argument("--model"); s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--as-of")
    s.add_argument("--no-macro", action="store_true"); s.set_defaults(fn=cmd_review_import)
    s = sub.add_parser("ask", help="ask the filings a question; answer with citations and confidence")
    s.add_argument("ticker"); s.add_argument("question"); s.add_argument("--row", help="ledger source the question is about"); s.add_argument("--data", default="data")
    s.add_argument("--logs", default="logs"); s.add_argument("--provider", default="anthropic"); s.add_argument("--model"); s.add_argument("--peers", nargs="*", default=[])
    s.add_argument("--as-of"); s.add_argument("--no-macro", action="store_true"); s.set_defaults(fn=cmd_ask)
    s = _common(sub.add_parser("brief", help="what the company does, why now, what the debate is; cited to the dossier"))
    s.add_argument("--provider", default="mock"); s.add_argument("--model"); s.add_argument("--seed", type=int, default=1); s.set_defaults(fn=cmd_brief)
    s = sub.add_parser("export", help="final workbook (mapping ledger, decision ledger, brief) and the unsure workbook")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--out"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("set-peers", help="record the chosen peer tickers after the peer_set decision")
    s.add_argument("ticker"); s.add_argument("tickers", nargs="+"); s.add_argument("--data", default="data"); s.set_defaults(fn=cmd_set_peers)

    s = sub.add_parser("attribution-fit", help="regression attribution from logged runs (no API calls)")
    s.add_argument("ticker"); s.add_argument("--assumption", choices=list(ASSUMPTIONS), required=True); s.add_argument("--data", default="data")
    s.add_argument("--logs", default="logs"); s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--analyst"); s.add_argument("--as-of")
    s.add_argument("--hist-years", type=int, default=5); s.add_argument("--releases", type=int, default=2); s.add_argument("--packet"); s.add_argument("--out")
    s.set_defaults(fn=cmd_attribution_fit)

    s = sub.add_parser("quarterly", help="quarterly income statement history from 10-Q/10-K facts (Q4 = FY minus nine months)")
    s.add_argument("ticker"); s.add_argument("--data", default="data"); s.add_argument("--as-of"); s.set_defaults(fn=cmd_quarterly)

    s = sub.add_parser("dates", help="list the filings on disk and the as-of dates a backtest can use")
    s.add_argument("ticker"); s.add_argument("--data", default="data")
    s.set_defaults(fn=cmd_dates)

    s = sub.add_parser("backtest", help="packet as of a past date (or --sweep N recent 10-K dates), debates + vote control, scored against realized")
    s.add_argument("ticker"); s.add_argument("--assumption", choices=list(ASSUMPTIONS), required=True)
    s.add_argument("--as-of", help="YYYY-MM-DD; use only filings on or before this date")
    s.add_argument("--sweep", type=int, default=0, help="run the N most recent scoreable 10-K dates instead of --as-of")
    s.add_argument("--vote-n", type=int, default=5, help="control arm size (0 disables)")
    s.add_argument("--data", default="data"); s.add_argument("--hist-years", type=int, default=5)
    s.add_argument("--peers", nargs="*", default=[]); s.add_argument("--analyst"); s.add_argument("--releases", type=int, default=2)
    s.add_argument("--provider", default=None); s.add_argument("--model", default=None)
    s.add_argument("--runs", type=int, default=3); s.add_argument("--rounds", type=int, default=2); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--temperature", type=float, default=0.7); s.add_argument("--policy", choices=["stubborn", "calibrated", "agreeable"], default="calibrated")
    s.add_argument("--logs", default="logs"); s.add_argument("--workers", type=int, default=4); s.add_argument("--no-cache", action="store_true")
    s.add_argument("--bull-model"); s.add_argument("--bear-model"); s.add_argument("--judge-model")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("decide", help="log an analyst decision on the latest verdict")
    s.add_argument("ticker"); s.add_argument("--assumption", choices=list(ASSUMPTIONS), required=True)
    s.add_argument("--action", choices=["accept", "edit", "reject"], required=True); s.add_argument("--value", type=float)
    s.add_argument("--reason", required=True); s.add_argument("--analyst-name", default="analyst"); s.add_argument("--logs", default="logs")
    s.set_defaults(fn=cmd_decide)

    s = sub.add_parser("demo", help="run the whole loop offline (synthetic data, mock LLM)")
    s.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
