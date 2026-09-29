"""Pull filings data for a ticker with edgartools and write it to data/<TICKER>/.

Runs on a machine with internet access to sec.gov (EDGAR requires a User-Agent
identity: set EDGAR_IDENTITY="Your Name your@email"). Raw statement frames are
saved untouched so the mapping step (``mapping.py``) is reproducible and any
unmapped label can be inspected.

Outputs:
  annual_is.csv / annual_bs.csv / annual_cf.csv        standardized statements, N annual periods
  quarterly_is.csv / quarterly_bs.csv / quarterly_cf.csv N quarterly periods
  facts_pit.parquet (or .csv)                          all XBRL facts with filing_date (point-in-time, for backtests)
  tenk_sections.json                                   Item 1 business, Item 1A risk factors, Item 7 MD&A, competition excerpt
  meta.json                                            name, CIK, SIC, industry, filing list with accession numbers
  mapping_report.txt                                   what mapped, what did not, residual sizes
"""
from __future__ import annotations

import json
import re
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _df_out(df, path: Path) -> bool:
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_dataframe()
        except Exception:
            return False
    if "label" not in [str(c).lower() for c in df.columns]:
        df = df.reset_index()
        if "index" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"index": "label"})
    df.to_csv(path, index=False)
    return True


def _competition_excerpt(business: str, window: int = 1500, max_chars: int = 4000) -> str:
    if not business:
        return ""
    text = re.sub(r"\s+", " ", business)
    out, used = [], 0
    for m in re.finditer(r"compet", text, re.I):
        s, e = max(0, m.start() - window // 3), min(len(text), m.end() + window)
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    parts = []
    for s, e in out:
        parts.append(text[s:e])
        used += e - s
        if used > max_chars:
            break
    return "\n...\n".join(parts)[:max_chars]


def _retry(fn, attempts: int = 4, first_wait: float = 2.0):
    """Retry on SEC 429/5xx and transient network errors with exponential backoff."""
    import time

    wait = first_wait
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(code in msg for code in ("503", "502", "504", "429", "timed out", "Timeout", "Connection"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(wait)
            wait *= 2.5


def ingest_eightk(co, d: Path, years: int = 5, max_releases: int | None = None) -> int:
    """Save earnings-release exhibits (8-K Item 2.02, EX-99.1) as data/<T>/eightk/<filing_date>_<accession>.json.

    Guidance, orders, backlog and book-to-bill live here, not in the 10-K. Public SEC data, dated, so as-of
    packets can use only what was known at the time. The SEC submissions index lists each 8-K's items, so
    only Item 2.02 filings are fetched (about four a year), and every fetch retries on 503/429.
    """
    from datetime import date

    out = d / "eightk"
    out.mkdir(exist_ok=True)
    max_releases = max_releases or (4 * years + 8)
    cutoff = date.today().replace(year=date.today().year - years - 1)
    n = 0
    try:
        filings = co.get_filings(form="8-K")
    except Exception as e:
        warnings.warn(f"8-K list: {e}")
        return 0
    for f in filings:
        try:
            fd = f.filing_date if hasattr(f.filing_date, "year") else date.fromisoformat(str(f.filing_date)[:10])
            if fd < cutoff:
                break
            idx_items = str(getattr(f, "items", "") or "")
            if idx_items and "2.02" not in idx_items:
                continue  # not an earnings 8-K, no document fetch needed
            path = out / f"{fd.isoformat()}_{f.accession_number}.json"
            if path.exists():
                n += 1
                continue
            ek = _retry(f.obj)
            items = [str(i) for i in (getattr(ek, "items", None) or [])] or [idx_items]
            if not any("2.02" in i for i in items):
                continue
            ex = ek.get_exhibit("EX-99.1")
            text = ""
            if ex is not None:
                try:
                    text = _retry(ex.text) or ""
                except Exception:
                    text = ""
            if not text:
                try:
                    text = _retry(ek.text) or ""
                except Exception:
                    text = ""
            if len(text) < 500:
                continue
            path.write_text(json.dumps({"accession": f.accession_number, "filing_date": fd.isoformat(),
                                        "date_of_report": str(getattr(ek, "date_of_report", "") or getattr(f, "report_date", "") or ""),
                                        "items": items, "text": re.sub(r"[ \t]+", " ", text)[:250_000]}, indent=1))
            n += 1
            if n >= max_releases:
                break
        except Exception as e:
            warnings.warn(f"8-K {getattr(f, 'accession_number', '?')}: {e}")
    return n


def _xbrl_facts_frame(f, d: Path, form: str, fy: int, fiscal_period: str, save_segments: bool = False) -> pd.DataFrame | None:
    """Every fact in one filing, custom extension concepts included, normalized to the companyfacts columns.
    Dimensional facts are dropped (segment members are saved separately when asked)."""
    xb = _retry(f.xbrl)
    if xb is None:
        return None
    df = xb.facts.to_dataframe()
    dim_col = next((c for c in df.columns if c.lower() in ("dimension", "dim_key", "dimensions", "axis")), None)
    mem_col = next((c for c in df.columns if c.lower() in ("member", "dim_value", "dimension_value")), None)
    if save_segments and dim_col and mem_col:
        seg = df[df[dim_col].notna() & (df[dim_col].astype(str) != "")]
        seg = seg[seg[dim_col].astype(str).str.contains("Segment|Product|Geograph", case=False, regex=True)]
        if len(seg):
            pe_c = next((c for c in ("period_end", "end_date") if c in seg.columns), None)
            pd.DataFrame({"concept": seg["concept"].astype(str), "axis": seg[dim_col].astype(str), "member": seg[mem_col].astype(str),
                          "numeric_value": pd.to_numeric(seg.get("numeric_value", seg.get("value")), errors="coerce"),
                          "period_end": pd.to_datetime(seg[pe_c], errors="coerce") if pe_c else pd.NaT,
                          "fiscal_year": fy}).dropna(subset=["numeric_value"]).to_parquet(d / f"xbrl_segments_{fy}.parquet", index=False)
    if dim_col:
        df = df[df[dim_col].isna() | (df[dim_col].astype(str) == "")]
    cols = {c: c for c in df.columns}
    ps = next((c for c in ("period_start", "start_date", "period_start_date") if c in cols), None)
    pe = next((c for c in ("period_end", "end_date", "period_end_date", "period_instant") if c in cols), None)
    norm = pd.DataFrame({
        "concept": df["concept"].astype(str),
        "numeric_value": pd.to_numeric(df.get("numeric_value", df.get("value")), errors="coerce"),
        "period_start": pd.to_datetime(df[ps], errors="coerce") if ps else pd.NaT,
        "period_end": pd.to_datetime(df[pe], errors="coerce") if pe else pd.NaT,
        "period_type": df.get("period_type", pd.Series(["duration" if ps else "instant"] * len(df))),
        "statement_type": df.get("statement_type", ""),
        "label": df.get("label", ""),
    })
    if ps and pe:
        norm["period_type"] = np.where(norm.period_start.isna(), "instant", "duration")
    norm["filing_date"] = pd.to_datetime(str(f.filing_date)[:10])
    norm["accession"] = f.accession_number
    norm["form_type"] = form
    norm["fiscal_year"] = fy
    norm["fiscal_period"] = fiscal_period
    return norm.dropna(subset=["numeric_value", "period_end"])


def ingest_xbrl_facts(co, d: Path, years: int = 5) -> int:
    """Full-filing facts for the last ``years`` 10-Ks -> xbrl_facts_<FY>.parquet (segment members -> xbrl_segments_<FY>.parquet)."""
    n = 0
    try:
        filings = co.get_filings(form="10-K")
    except Exception as e:
        warnings.warn(f"10-K list for xbrl: {e}")
        return 0
    for f in filings:
        try:
            por = str(getattr(f, "period_of_report", "") or "")
            if not por[:4].isdigit():
                continue
            out = d / f"xbrl_facts_{por[:4]}.parquet"
            if out.exists():
                n += 1
            else:
                norm = _xbrl_facts_frame(f, d, "10-K", int(por[:4]), "FY", save_segments=True)
                if norm is None:
                    continue
                norm.to_parquet(out, index=False)
                n += 1
            if n >= years:
                break
        except Exception as e:
            warnings.warn(f"xbrl facts {getattr(f, 'accession_number', '?')}: {e}")
    return n


def ingest_xbrl_facts_quarterly(co, d: Path, quarters: int = 8) -> int:
    """Full-filing facts for the last ``quarters`` 10-Qs -> xbrl_facts_q_<period>.parquet, so custom cost concepts that only
    appear in the quarterly filings (a miner's energy and hosting lines) can feed the quarterly history."""
    n = 0
    try:
        filings = co.get_filings(form="10-Q")
    except Exception as e:
        warnings.warn(f"10-Q list for xbrl: {e}")
        return 0
    for f in filings:
        try:
            por = str(getattr(f, "period_of_report", "") or "")
            if not por[:4].isdigit():
                continue
            out = d / f"xbrl_facts_q_{por[:10]}.parquet"
            if out.exists():
                n += 1
            else:
                norm = _xbrl_facts_frame(f, d, "10-Q", int(por[:4]), "Q")
                if norm is None:
                    continue
                norm.to_parquet(out, index=False)
                n += 1
            if n >= quarters:
                break
        except Exception as e:
            warnings.warn(f"xbrl facts {getattr(f, 'accession_number', '?')}: {e}")
    return n


def ingest_proxy(co, d: Path) -> bool:
    """Latest proxy statement (DEF 14A) text: the compensation discussion carries the incentive metrics."""
    out = d / "proxy_text.json"
    if out.exists():
        return True
    try:
        filings = co.get_filings(form="DEF 14A")
        f = filings[0] if filings else None
        if f is None:
            return False
        text = _retry(f.text) if callable(getattr(f, "text", None)) else str(getattr(f, "text", ""))
        text = re.sub(r"[ \t]+", " ", str(text or ""))
        out.write_text(json.dumps({"filing_date": str(f.filing_date)[:10], "accession": f.accession_number, "text": text[:900_000]}))
        return True
    except Exception as e:
        warnings.warn(f"proxy: {e}")
        return False


def ingest(ticker: str, out_dir: Path, years: int = 5, quarters: int = 8, with_text: bool = True, with_pit: bool = True,
           identity: str | None = None, with_eightk: bool = True, only_eightk: bool = False, with_xbrl: bool = True) -> Path:
    from edgar import Company, set_identity  # lazy: needs network + edgartools

    set_identity(identity or os.environ.get("EDGAR_IDENTITY", "crucible-mvp research@example.com"))
    ticker = ticker.upper()
    d = Path(out_dir) / ticker
    d.mkdir(parents=True, exist_ok=True)
    co = Company(ticker)
    if getattr(co, "not_found", False):
        raise ValueError(f"EDGAR could not find ticker {ticker}")
    if only_eightk:
        n = ingest_eightk(co, d, years=years)
        print(f"{ticker}: {n} earnings releases within the {years + 1}-year window ({len(list((d / 'eightk').glob('*.json')))} on disk)")
        mp = d / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            m["eightk_releases"] = n
            mp.write_text(json.dumps(m, indent=2, default=str))
        return d
    meta: dict = {"ticker": ticker, "name": getattr(co, "name", None), "cik": getattr(co, "cik", None),
                  "sic": getattr(co, "sic", None), "industry": getattr(co, "industry", None),
                  "fiscal_year_end": getattr(co, "fiscal_year_end", None),
                  "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "edgartools": None, "files": []}
    try:
        import importlib.metadata as m
        meta["edgartools"] = m.version("edgartools")
    except Exception:
        pass

    # ---- statements (standardized labels come from edgartools' XBRL standardization)
    pulls = [
        ("annual_is.csv", lambda: co.income_statement(periods=years, period="annual", as_dataframe=True)),
        ("annual_bs.csv", lambda: co.balance_sheet(periods=years, period="annual", as_dataframe=True)),
        ("annual_cf.csv", lambda: co.cash_flow_statement(periods=years, period="annual", as_dataframe=True)),
        ("quarterly_is.csv", lambda: co.income_statement(periods=quarters, period="quarterly", as_dataframe=True)),
        ("quarterly_bs.csv", lambda: co.balance_sheet(periods=quarters, period="quarterly", as_dataframe=True)),
        ("quarterly_cf.csv", lambda: co.cash_flow_statement(periods=quarters, period="quarterly", as_dataframe=True)),
    ]
    for name, fn in pulls:
        try:
            ok = _df_out(fn(), d / name)
            meta["files"].append({"file": name, "ok": ok})
        except Exception as e:  # keep going: partial data is still useful for the mapping report
            warnings.warn(f"{ticker} {name}: {e}")
            meta["files"].append({"file": name, "ok": False, "error": str(e)[:200]})

    # ---- filing provenance
    filings = []
    try:
        for f in co.get_filings(form=["10-K", "10-Q"]):
            filings.append({"form": f.form, "accession": f.accession_number, "filing_date": str(f.filing_date),
                            "period_of_report": str(getattr(f, "period_of_report", "")), "url": getattr(f, "filing_url", None) or getattr(f, "url", None)})
            if len(filings) >= (years + quarters + 4):
                break
    except Exception as e:
        warnings.warn(f"{ticker} filings list: {e}")
    meta["filings"] = filings
    sources = {}
    for f in filings:
        if f["form"] == "10-K" and f["period_of_report"][:4].isdigit():
            sources.setdefault(f["period_of_report"][:4], f"10-K {f['accession']} filed {f['filing_date']}")
    meta["sources_by_fiscal_year"] = sources

    # ---- text sections from the latest 10-K
    if with_text:
        sections = {}
        try:
            tenk = co.latest_tenk
            for key, attr in (("Item 1 Business", "business"), ("Item 1A Risk Factors", "risk_factors"), ("Item 7 MD&A", "management_discussion")):
                try:
                    val = getattr(tenk, attr, None)
                    sections[key] = str(val) if val else ""
                except Exception as e:
                    sections[key] = ""
                    warnings.warn(f"{ticker} {key}: {e}")
            try:
                sections["Item 8 Notes"] = str(tenk.get("Item 8", "") or "")[:600_000]  # accounting policies and notes: what each line contains
            except Exception:
                sections["Item 8 Notes"] = ""
            sections["Competition excerpt"] = _competition_excerpt(sections.get("Item 1 Business", ""))
            meta["tenk"] = {"filing_date": str(getattr(tenk, "filing_date", "")), "period_of_report": str(getattr(tenk, "period_of_report", ""))}
        except Exception as e:
            warnings.warn(f"{ticker} 10-K text: {e}")
        (d / "tenk_sections.json").write_text(json.dumps(sections, indent=2))
        # earlier 10-Ks too, so an as-of packet uses the text the analyst had at the time
        try:
            for f in co.get_filings(form="10-K"):
                por = str(getattr(f, "period_of_report", ""))
                if not por[:4].isdigit():
                    continue
                out = d / f"tenk_sections_{por[:4]}.json"
                if out.exists():
                    try:
                        if "Item 8 Notes" in json.loads(out.read_text()):
                            continue  # complete; files from before the notes were added get refreshed
                    except Exception:
                        pass
                if len(list(d.glob("tenk_sections_*.json"))) >= years:
                    break
                tk = f.obj()
                sec = {}
                for key, attr in (("Item 1 Business", "business"), ("Item 1A Risk Factors", "risk_factors"), ("Item 7 MD&A", "management_discussion")):
                    try:
                        val = getattr(tk, attr, None)
                        sec[key] = str(val) if val else ""
                    except Exception:
                        sec[key] = ""
                try:
                    sec["Item 8 Notes"] = str(tk.get("Item 8", "") or "")[:600_000]
                except Exception:
                    sec["Item 8 Notes"] = ""
                sec["Competition excerpt"] = _competition_excerpt(sec.get("Item 1 Business", ""))
                sec["_filing"] = {"accession": f.accession_number, "filing_date": str(f.filing_date), "period_of_report": por}
                out.write_text(json.dumps(sec, indent=2))
        except Exception as e:
            warnings.warn(f"{ticker} historical 10-K text: {e}")

    # ---- point-in-time facts for backtesting
    if with_pit:
        try:
            facts = co.get_facts()
            df = facts.to_dataframe(include_metadata=True, pit_mode=True)
            try:
                df.to_parquet(d / "facts_pit.parquet", index=False)
            except Exception:
                df.to_csv(d / "facts_pit.csv", index=False)
            meta["facts_pit_rows"] = int(len(df))
        except Exception as e:
            warnings.warn(f"{ticker} facts: {e}")

    if with_xbrl:
        meta["xbrl_facts_years"] = ingest_xbrl_facts(co, d, years=years)
        meta["xbrl_facts_quarters"] = ingest_xbrl_facts_quarterly(co, d, quarters=quarters)
        print(f"{ticker}: full-filing XBRL facts for {meta['xbrl_facts_years']} fiscal years and {meta['xbrl_facts_quarters']} quarters (custom concepts and segment members included)")
    if with_text:
        meta["proxy"] = ingest_proxy(co, d)
        print(f"{ticker}: proxy statement text {'saved' if meta['proxy'] else 'not available'}")
    if with_eightk:
        meta["eightk_releases"] = ingest_eightk(co, d, years=years)
        on_disk = len(list((d / "eightk").glob("*.json")))
        print(f"{ticker}: {meta['eightk_releases']} earnings releases within the {years + 1}-year window ({on_disk} on disk; older ones are kept), "
              f"{len(list(d.glob('tenk_sections_*.json')))} 10-K text years under {d}")

    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---- mapping report (offline step, run here so the analyst sees problems immediately)
    try:
        from .mapping import load_history

        hist, reports = load_history(Path(out_dir), ticker)
        text = "\n".join(r.summary() for r in reports.values())
        text += f"\n\nyears: {list(hist.columns)} (USD millions)\nrevenue: {hist.loc['revenue'].round(1).to_dict()}\n"
        text += "\nsource per line:\n" + "\n".join(f"  {k}: {v}" for r in reports.values() for k, v in r.mapped.items())
        (d / "mapping_report.txt").write_text(text)
        print(text)
    except Exception as e:
        (d / "mapping_report.txt").write_text(f"mapping failed: {e}")
        print(f"mapping failed: {e}")
    return d


def load_sections(data_dir: Path, ticker: str, as_of: str | None = None) -> dict[str, str]:
    """Latest 10-K sections, or the newest 10-K filed on or before ``as_of`` when per-year files exist."""
    d = Path(data_dir) / ticker.upper()
    if as_of:
        cands = []
        for p in d.glob("tenk_sections_*.json"):
            sec = json.loads(p.read_text())
            fd = (sec.get("_filing") or {}).get("filing_date", "")
            if fd and fd <= as_of:
                cands.append((fd, sec))
        if cands:
            sec = max(cands, key=lambda x: x[0])[1]
            return {k: v for k, v in sec.items() if not k.startswith("_")}
        return {}  # no dated text available: numbers-only packet rather than leaking future text
    p = d / "tenk_sections.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_meta(data_dir: Path, ticker: str) -> dict:
    p = Path(data_dir) / ticker.upper() / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_eightk(data_dir: Path, ticker: str, as_of: str | None = None, n: int = 2) -> list[dict]:
    """The ``n`` most recent earnings releases (filed on or before ``as_of`` when given), newest first."""
    d = Path(data_dir) / ticker.upper() / "eightk"
    if not d.exists():
        return []
    rels = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if as_of and r.get("filing_date", "") > as_of:
            continue
        rels.append(r)
        if len(rels) >= n:
            break
    return rels
