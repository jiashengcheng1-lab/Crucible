"""Crucible UI. Run with `streamlit run crucible/ui/app.py` (or `crucible-ui`). Thin layer over crucible.ui.service."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crucible.ui import service as svc

st.set_page_config(page_title="crucible", layout="wide")

with st.sidebar:
    st.title("crucible")
    st.caption("filings in, a model out, every number traceable, every judgment recorded")
    data_dir = Path(st.text_input("data directory", value=st.session_state.get("data_dir", "data")))
    st.session_state["data_dir"] = str(data_dir)
    known = svc.tickers(data_dir)
    ticker = st.text_input("ticker", value=st.session_state.get("ticker", known[0] if known else "")).upper().strip()
    st.session_state["ticker"] = ticker
    if known:
        st.caption("on disk: " + ", ".join(known))
    provider = st.selectbox("model provider", ["anthropic", "mock"], index=0 if os.environ.get("ANTHROPIC_API_KEY") else 1)
    st.caption("anthropic needs ANTHROPIC_API_KEY in the environment; mock runs offline with placeholder answers")
    analyst = st.text_input("your name (recorded on decisions)", value=st.session_state.get("analyst", "analyst"))
    st.session_state["analyst"] = analyst
    if ticker:
        c = svc.counts(data_dir, ticker)
        st.write({"mapping": c.get("mapping", {}), "decisions": c.get("decisions", {})})

if not ticker:
    st.info("Enter a ticker in the sidebar to begin (for example VRT, MARA, AAPL).")
    st.stop()

tab_run, tab_review, tab_stmt, tab_brief, tab_model, tab_debate, tab_qa = st.tabs(["Run", "Review", "Statements", "Brief & profile", "Model", "Debates", "Q&A log"])

# --------------------------------------------------------------------------- Run
with tab_run:
    st.subheader(f"Run the pipeline for {ticker}")
    st.caption("Each step is the CLI command shown; output appears below. Steps that call the model use the provider in the sidebar.")
    cols = st.columns(2)
    for i, (key, label, _) in enumerate(svc.STEPS):
        args = svc.step_args(key, ticker, provider)
        with cols[i % 2]:
            if st.button(f"{label}", key=f"run_{key}", help="python -m crucible " + " ".join(args)):
                with st.spinner(f"running {key}..."):
                    res = svc.run_cli(args)
                st.session_state[f"out_{key}"] = res
    for key, _, _ in svc.STEPS:
        res = st.session_state.get(f"out_{key}")
        if res:
            with st.expander(f"{res['cmd']}  (exit {res['code']})", expanded=(res["code"] != 0)):
                if res["stdout"]:
                    st.code(res["stdout"][-8000:])
                if res["stderr"]:
                    st.code(res["stderr"][-4000:])
    st.markdown("**Custom command**")
    custom = st.text_input("arguments after `python -m crucible`", value=f"model {ticker} --template auto")
    if st.button("run custom"):
        with st.spinner("running..."):
            res = svc.run_cli(custom.split())
        st.code((res["stdout"] + "\n" + res["stderr"])[-10000:])

# --------------------------------------------------------------------------- Review
with tab_review:
    st.subheader("Review what the model proposed")
    area = st.radio("what", ["mapping", "decisions"], horizontal=True)
    include_decided = st.checkbox("include decided rows", value=False)
    df = svc.pending(data_dir, ticker, area, include_decided)
    if df.empty:
        st.info("Nothing pending. Run map-suggest (mapping) or choose (decisions) on the Run tab.")
    else:
        show = df[["statement_or_decision", "source_or_option", "proposed_target", "confidence", "status", "proposed_by"]].reset_index(drop=True)
        st.dataframe(show, width='stretch', height=min(400, 40 + 28 * len(show)))
        idx = st.number_input("row to review (index in the table above)", min_value=0, max_value=len(df) - 1, value=0, step=1)
        r = df.iloc[int(idx)]
        st.markdown(f"### [{r['statement_or_decision']}] {r['source_or_option']}")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**proposed target:** `{r['proposed_target']}`  **confidence:** {r['confidence']}  **by:** {r['proposed_by']}")
            st.markdown(f"**alternatives:** {r['alternatives']}")
            st.markdown(f"**evidence:** {r['evidence']}")
            if r["target_now_mapped_from"]:
                st.markdown(f"**target now mapped from:** {r['target_now_mapped_from']}")
            st.markdown(f"**reasoning:** {r['reasoning']}")
            st.markdown(f"**citation:** {r['citation']}")
            st.markdown(f"**questions that would settle it:** {r['questions']}")
            if r["considerations_duplicates"]:
                st.markdown(f"**considerations / duplicates:** {r['considerations_duplicates']}")
        with c2:
            note = st.text_area("note", value="", key=f"note_{area}_{idx}")
            target = st.text_input("retarget (optional schema key or option value)", value="", key=f"tgt_{area}_{idx}")
            b1, b2, b3 = st.columns(3)
            done = None
            if b1.button("accept", key=f"acc_{area}_{idx}"):
                done = svc.decide(data_dir, ticker, area, r["statement_or_decision"], r["source_or_option"], "accepted", target or None, note, analyst)
            if b2.button("reject", key=f"rej_{area}_{idx}"):
                done = svc.decide(data_dir, ticker, area, r["statement_or_decision"], r["source_or_option"], "rejected", None, note, analyst)
            if b3.button("unsure", key=f"uns_{area}_{idx}"):
                done = svc.decide(data_dir, ticker, area, r["statement_or_decision"], r["source_or_option"], "unsure", None, note or "marked unsure", analyst)
            if done:
                st.success(f"recorded ({done} row)")
                st.rerun()
        st.markdown("**Ask the filings about this row**")
        q = st.text_input("follow-up question", key=f"q_{area}_{idx}")
        if st.button("ask", key=f"ask_{area}_{idx}") and q.strip():
            with st.spinner("reading the filings..."):
                ans = svc.ask_question(data_dir, ticker, provider, r.to_dict(), q.strip())
            st.markdown(f"**answer** (confidence {ans['confidence']:.2f}{', unsupported by the filings or the model state' if ans.get('unsupported') else ''}): {ans['answer']}")
            for c in ans.get("citations", []):
                st.caption(f"[{c['id']}] {c['source']} | {c['text']}")
        st.caption("Questions see the filings, the dossier, and the model's own state: the three statements as mapped now and the forecast, DCF and sheets as last built.")
        with st.expander("the statement this row belongs to, as mapped now"):
            code = r["statement_or_decision"] if r["statement_or_decision"] in ("IS", "BS", "CF") else None
            if code:
                try:
                    tables = svc.statements(data_dir, ticker)
                    name = {"IS": "Income statement", "BS": "Balance sheet", "CF": "Cash flow"}[code]
                    st.dataframe(tables[name], width='stretch')
                except Exception as e:
                    st.warning(f"statements not available: {e}")
            else:
                st.caption("decision rows are not statement lines")
    st.divider()
    st.markdown("**Offline review**: export everything pending with every field to a workbook, decide in Excel, import it back (follow-up questions in the workbook are answered on import).")
    e1, e2 = st.columns(2)
    if e1.button("export review workbook"):
        res = svc.run_cli(["review-export", ticker])
        st.code(res["stdout"] + res["stderr"])
    up = e2.file_uploader("import a filled review workbook", type=["xlsx"])
    if up is not None and st.button("import"):
        p = Path("outputs") / f"{ticker}_review_upload.xlsx"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(up.getbuffer())
        res = svc.run_cli(["review-import", ticker, str(p), "--provider", provider, "--analyst-name", analyst])
        st.code(res["stdout"] + res["stderr"])

# --------------------------------------------------------------------------- Statements
with tab_stmt:
    st.subheader("The three statements as mapped now")
    st.caption("USD millions. 'source' is where each line comes from (XBRL concept, statement label, or a derivation). Residual lines absorb what maps to nothing. "
               "The last table is the model as last built, history and forecast, when a model exists.")
    try:
        tables = svc.statements(data_dir, ticker)
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, width='stretch', height=min(600, 40 + 28 * len(df)))
    except Exception as e:
        st.info(f"No mapped statements yet ({e}). Run ingest on the Run tab.")

# --------------------------------------------------------------------------- Brief & profile
with tab_brief:
    arts = svc.artifacts(ticker)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Brief")
        if "brief_md" in arts:
            st.markdown(arts["brief_md"].read_text())
        else:
            st.info("No brief yet: run 'Write the brief' on the Run tab.")
    with c2:
        st.subheader("Profile")
        if "profile_md" in arts:
            st.markdown(arts["profile_md"].read_text())
        else:
            st.info("No profile yet: run 'Profile the company' on the Run tab.")

# --------------------------------------------------------------------------- Model
with tab_model:
    st.subheader("Model")
    arts = svc.artifacts(ticker)
    res = st.session_state.get("out_model")
    if res and res["stdout"].strip().startswith("{"):
        try:
            st.json(json.loads(res["stdout"]))
        except Exception:
            st.code(res["stdout"][-6000:])
    for key, label in (("model", "model workbook"), ("final", "final workbook (ledgers + brief)"), ("unsure", "unsure workbook"), ("review", "review workbook")):
        if key in arts:
            st.download_button(f"download {label}", data=arts[key].read_bytes(), file_name=arts[key].name, key=f"dl_{key}")
    if "quarterly" in arts:
        st.markdown("**Quarterly history**")
        st.dataframe(pd.read_csv(arts["quarterly"], index_col=0), width='stretch')
    if "driver_build" in arts:
        st.markdown("**Driver build (industry template)**")
        st.dataframe(pd.read_csv(arts["driver_build"], index_col=0).round(2), width='stretch')
    if "elasticity" in arts:
        st.markdown("**Per-driver elasticity**")
        st.dataframe(pd.DataFrame(json.loads(arts["elasticity"].read_text())), width='stretch')
    st.divider()
    st.markdown("**Preview an Excel workbook**")
    wbs = svc.workbooks(ticker)
    if not wbs:
        st.info("No workbook yet: build the model or export on the Run tab.")
    else:
        which = st.selectbox("workbook", list(wbs.keys()), format_func=lambda k: f"{k}: {wbs[k].name}")
        names = svc.sheet_names(wbs[which])
        sheet = st.selectbox("sheet", names)
        frame = svc.sheet_frame(wbs[which], sheet)
        st.caption("Values as stored; formula cells show the formula text (open the file in Excel for computed values).")
        st.dataframe(frame, width='stretch', height=min(700, 40 + 28 * len(frame)))

# --------------------------------------------------------------------------- Debates
with tab_debate:
    st.subheader("Assumption debates")
    assumption = st.selectbox("assumption", ["revenue_growth_3y", "wacc", "hashrate_growth_3y", "network_hashrate_growth_3y", "btc_price_change_3y"])
    if st.button("run debate"):
        with st.spinner("debating..."):
            res = svc.run_cli(["debate", ticker, "--assumption", assumption, "--provider", provider])
        st.code((res["stdout"] + "\n" + res["stderr"])[-6000:])
    for p in svc.debate_reviews(ticker):
        with st.expander(p.name):
            st.markdown(p.read_text())

# --------------------------------------------------------------------------- Q&A log
with tab_qa:
    st.subheader("Questions asked and answers given")
    rows = svc.qa_log(Path("logs"), ticker)
    if not rows:
        st.info("No questions yet.")
    for a in reversed(rows):
        st.markdown(f"**Q:** {a['question']}  \n**A** (confidence {a['confidence']:.2f}{', unsupported' if a.get('unsupported') else ''}): {a['answer']}")
        for c in a.get("citations", []):
            st.caption(f"{c['source']} | {c['text'][:200]}")
        st.divider()
