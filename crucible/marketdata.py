"""Market data for the cost-of-capital inputs, with disk caches so a run never refetches the same day twice.

Sources (all public):
  * FRED (St. Louis Fed API): daily Treasury yields, e.g. DGS10. Key from FRED_API_KEY.
  * Yahoo Finance via yfinance: weekly adjusted closes for the company, its peers and the S&P 500 (^GSPC).
  * Damodaran (NYU Stern): implied equity risk premium table (best effort; falls back to the analyst's value).

Caches live under data/market/. Every fetcher is injectable so tests run offline.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
DAMODARAN_HISTIMPL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"


def _d(s: str | date | None) -> date:
    if s is None:
        return date.today()
    return s if isinstance(s, date) else date.fromisoformat(str(s)[:10])


# ----------------------------------------------------------------------------- FRED

def fetch_fred(series_id: str, start: date, end: date, api_key: str | None = None) -> pd.DataFrame:
    """Daily observations (date, value) from the FRED API. Missing values ('.') are dropped."""
    import httpx

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set (get a free key at https://fredaccount.stlouisfed.org/apikey; keep it in .env, never in chat)")
    r = httpx.get(FRED_URL, params={"series_id": series_id, "api_key": key, "file_type": "json",
                                    "observation_start": start.isoformat(), "observation_end": end.isoformat()}, timeout=30)
    r.raise_for_status()
    rows = [(o["date"], float(o["value"])) for o in r.json().get("observations", []) if o.get("value") not in (None, ".")]
    return pd.DataFrame(rows, columns=["date", "value"])


def fred_series(series_id: str, as_of: str | date | None = None, years: int = 3, data_dir: Path = Path("data"),
                fetch: Callable[..., pd.DataFrame] = fetch_fred, refresh_days: int = 1) -> pd.DataFrame:
    """Cached daily series covering [as_of - years, as_of]; refetches only when the cache does not reach as_of."""
    as_of_d = _d(as_of)
    start = as_of_d - timedelta(days=365 * years)
    cache = Path(data_dir) / "market"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"fred_{series_id}.csv"
    df = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["date", "value"])
    have_end = df["date"].max().date() if len(df) else None
    need = have_end is None or (have_end < as_of_d - timedelta(days=refresh_days) and as_of_d <= date.today()) or (len(df) and df["date"].min().date() > start)
    if need:
        new = fetch(series_id, start, as_of_d if as_of_d <= date.today() else date.today())
        new["date"] = pd.to_datetime(new["date"])
        df = pd.concat([df, new]).drop_duplicates("date").sort_values("date")
        df.to_csv(p, index=False)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(as_of_d))].reset_index(drop=True)


def risk_free_rate(as_of: str | date | None = None, series_id: str = "DGS10", data_dir: Path = Path("data"),
                   fetch: Callable[..., pd.DataFrame] = fetch_fred) -> dict:
    """Latest observation on or before as_of, as a decimal, with its citation."""
    df = fred_series(series_id, as_of, years=1, data_dir=data_dir, fetch=fetch)
    if not len(df):
        raise RuntimeError(f"no {series_id} observations on or before {as_of}")
    last = df.iloc[-1]
    return {"value": round(float(last["value"]) / 100, 5), "series": series_id, "date": last["date"].strftime("%Y-%m-%d"),
            "source": f"FRED {series_id} ({'10-year Treasury constant maturity' if series_id == 'DGS10' else series_id}) observation {last['date'].strftime('%Y-%m-%d')}: {float(last['value']):.2f}%"}


# ----------------------------------------------------------------------------- Yahoo prices

def fetch_yahoo_weekly(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Weekly adjusted closes (date, close) via yfinance."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1wk", auto_adjust=True)
    if hist is None or not len(hist):
        raise RuntimeError(f"no Yahoo prices for {symbol}")
    out = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.dropna()


def weekly_prices(symbol: str, as_of: str | date | None = None, years: int = 2, data_dir: Path = Path("data"),
                  fetch: Callable[..., pd.DataFrame] = fetch_yahoo_weekly, refresh_days: int = 7) -> pd.DataFrame:
    as_of_d = _d(as_of)
    start = as_of_d - timedelta(days=int(365.25 * years) + 14)
    cache = Path(data_dir) / "market"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"prices_{symbol.replace('^', 'IDX_')}.csv"
    df = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["date", "close"])
    have_end = df["date"].max().date() if len(df) else None
    have_start = df["date"].min().date() if len(df) else None
    need = have_end is None or have_start > start or (have_end < as_of_d - timedelta(days=refresh_days) and as_of_d <= date.today())
    if need:
        new = fetch(symbol, start, min(as_of_d, date.today()))
        df = pd.concat([df, new]).drop_duplicates("date").sort_values("date")
        df.to_csv(p, index=False)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(as_of_d))].reset_index(drop=True)


def price_on_or_before(symbol: str, as_of: str | date | None = None, data_dir: Path = Path("data"),
                       fetch: Callable[..., pd.DataFrame] = fetch_yahoo_weekly) -> dict:
    df = weekly_prices(symbol, as_of, years=1, data_dir=data_dir, fetch=fetch)
    if not len(df):
        raise RuntimeError(f"no price for {symbol} on or before {as_of}")
    last = df.iloc[-1]
    return {"value": float(last["close"]), "date": last["date"].strftime("%Y-%m-%d"), "source": f"Yahoo Finance {symbol} weekly close {last['date'].strftime('%Y-%m-%d')}"}


def weekly_beta(symbol: str, benchmark: str = "^GSPC", as_of: str | date | None = None, years: int = 2, data_dir: Path = Path("data"),
                fetch: Callable[..., pd.DataFrame] = fetch_yahoo_weekly) -> dict:
    """OLS beta of weekly log returns vs the benchmark over the trailing window, with r2, n and the Blume-adjusted beta."""
    a = weekly_prices(symbol, as_of, years, data_dir, fetch).set_index("date")["close"]
    b = weekly_prices(benchmark, as_of, years, data_dir, fetch).set_index("date")["close"]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j.columns = ["a", "b"]
    r = np.log(j).diff().dropna()
    if len(r) < 30:
        raise RuntimeError(f"only {len(r)} overlapping weekly returns for {symbol} vs {benchmark}")
    x, y = r["b"].values, r["a"].values
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return {"value": round(float(beta), 3), "adjusted": round(float(2 / 3 * beta + 1 / 3), 3), "r2": round(float(r2), 3), "n_weeks": int(len(r)),
            "window": f"{r.index.min().date()}..{r.index.max().date()}", "benchmark": benchmark,
            "source": f"OLS of {symbol} weekly log returns on {benchmark}, {len(r)} weeks {r.index.min().date()}..{r.index.max().date()} (Yahoo Finance adjusted closes); Blume-adjusted {2 / 3 * beta + 1 / 3:.2f}"}


# ----------------------------------------------------------------------------- ERP

def fetch_damodaran_histimpl() -> str:
    import httpx

    r = httpx.get(DAMODARAN_HISTIMPL, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.text


def damodaran_implied_erp(fetch: Callable[[], str] = fetch_damodaran_histimpl) -> dict:
    """Latest annual implied ERP (FCFE-based) from Damodaran's historical implied premium table. Best effort."""
    html = fetch()
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.S | re.I)
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
    headers = [i for i, c in enumerate(cells) if "implied" in c.lower() and "fcfe" in c.lower()]
    if not headers:
        headers = [i for i, c in enumerate(cells) if "implied premium" in c.lower()]
    if not headers:
        raise RuntimeError("could not find the implied premium column")
    # rows are runs of cells that start with a year; the column offset is the header's offset within the header row
    year_pos = [i for i, c in enumerate(cells) if re.fullmatch(r"(19|20)\d\d", c)]
    header_row_start = max(i for i in range(headers[0]) if cells[i].lower().startswith("year")) if any(c.lower().startswith("year") for c in cells[:headers[0]]) else 0
    offset = headers[0] - header_row_start
    best = None
    for yp in year_pos:
        if yp + offset < len(cells):
            v = cells[yp + offset].replace("%", "").strip()
            try:
                val = float(v) / 100
            except ValueError:
                continue
            if 0.01 < val < 0.12:
                best = (int(cells[yp]), val)
    if not best:
        raise RuntimeError("no parsable implied premium row")
    return {"value": round(best[1], 4), "year": best[0], "source": f"Damodaran implied equity risk premium (FCFE), {best[0]} row of {DAMODARAN_HISTIMPL}"}


# ----------------------------------------------------------------------------- expected market return and ERP series

def total_return_index(as_of: str | date | None = None, years: int = 40, data_dir: Path = Path("data"), fetch: Callable[..., pd.DataFrame] = fetch_yahoo_weekly) -> tuple[pd.Series, str]:
    """S&P 500 total return index (^SP500TR) when Yahoo serves it, else the price index (^GSPC) plus an assumed 1.8% dividend yield."""
    try:
        s = weekly_prices("^SP500TR", as_of, years, data_dir, fetch).set_index("date")["close"]
        if len(s) > 100:
            return s, "^SP500TR total return"
    except Exception:
        pass
    s = weekly_prices("^GSPC", as_of, years, data_dir, fetch).set_index("date")["close"]
    return s, "^GSPC price index (+1.8% assumed dividend yield)"


ERP_METHODS = ("hist30y", "trend20y", "hist10y", "hist30d")


def expected_market_return(as_of: str | date | None = None, method: str = "hist30y", data_dir: Path = Path("data"),
                           fetch: Callable[..., pd.DataFrame] = fetch_yahoo_weekly) -> dict:
    """Expected market return Rm by method:
      hist30y  trailing 30-year annualized total return (the default: long enough to span several cycles; the longest
               available window is used and named when the index history is shorter)
      trend20y drift of an OLS fit to log total return over 20 years (annualized slope)
      hist10y  trailing 10-year annualized total return (recency-biased: the last decade was unusually strong)
      hist30d  trailing 30-day return annualized (a sentiment gauge, not an expectation; standard error is enormous)"""
    s, src = total_return_index(as_of, 40, data_dir, fetch)
    s = s.dropna()
    end = s.index[-1]
    add = 0.018 if "GSPC" in src else 0.0
    if method == "hist30y":
        span = 30
        prior = s[s.index <= end - timedelta(days=int(365.25 * span))]
        if not len(prior):
            span = max(5, int((end - s.index[0]).days / 365.25))
            prior = s[s.index <= end - timedelta(days=int(365.25 * span))]
        r = (float(s.iloc[-1]) / float(prior.iloc[-1])) ** (1 / span) - 1 if len(prior) else float("nan")
        note = f"trailing {span}-year annualized total return" + ("" if span == 30 else " (index history shorter than 30 years)")
    elif method == "hist30d":
        prior = s[s.index <= end - timedelta(days=30)]
        r = (float(s.iloc[-1]) / float(prior.iloc[-1])) ** (365.25 / 30) - 1 if len(prior) else float("nan")
        note = "trailing 30-day return annualized: a sentiment gauge, not an expectation; standard error is enormous"
    elif method == "trend20y":
        w = s[s.index >= end - timedelta(days=int(365.25 * 20))]
        t = np.arange(len(w)) / 52.18
        slope = np.polyfit(t, np.log(w.values), 1)[0]
        r = float(np.exp(slope) - 1)
        note = f"annualized drift of a log-linear OLS over {len(w)} weeks"
    else:
        prior = s[s.index <= end - timedelta(days=int(365.25 * 10))]
        r = (float(s.iloc[-1]) / float(prior.iloc[-1])) ** (1 / 10) - 1 if len(prior) else float("nan")
        note = "trailing 10-year annualized total return"
    return {"value": round(float(r) + add, 5), "method": method, "date": end.strftime("%Y-%m-%d"), "index": src,
            "source": f"expected market return ({method}): {note}; {src}; as of {end.date()}"}


def erp_series(as_of: str | date | None = None, data_dir: Path = Path("data"), rf_series_id: str = "DGS10", methods: tuple[str, ...] = ERP_METHODS,
               fetch_rf: Callable[..., pd.DataFrame] = fetch_fred, fetch_px: Callable[..., pd.DataFrame] = fetch_yahoo_weekly, months: int = 36) -> pd.DataFrame:
    """Monthly ERP history: at each month end, rf = latest DGS10 on or before the date, Rm by each method, ERP = Rm - rf.
    Saved to data/market/erp_series.csv so an ERP can be looked up at any as-of date (backtests) instead of typed in."""
    as_of_d = _d(as_of)
    rf = fred_series(rf_series_id, as_of_d, years=4, data_dir=data_dir, fetch=fetch_rf).set_index("date")["value"].astype(float) / 100
    s, src = total_return_index(as_of_d, 40, data_dir, fetch_px)
    rows = []
    dates = pd.date_range(end=pd.Timestamp(as_of_d), periods=months, freq="ME")
    for dt in dates:
        sub = s[s.index <= dt]
        rf_sub = rf[rf.index <= dt]
        if len(sub) < 60 or not len(rf_sub):
            continue
        row = {"date": dt.strftime("%Y-%m-%d"), "rf": round(float(rf_sub.iloc[-1]), 5), "rf_date": rf_sub.index[-1].strftime("%Y-%m-%d")}
        end = sub.index[-1]
        add = 0.018 if "GSPC" in src else 0.0
        for m in methods:
            if m == "hist30y":
                prior = sub[sub.index <= end - timedelta(days=int(365.25 * 30))]
                r = (float(sub.iloc[-1]) / float(prior.iloc[-1])) ** (1 / 30) - 1 if len(prior) else float("nan")
            elif m == "hist10y":
                prior = sub[sub.index <= end - timedelta(days=int(365.25 * 10))]
                r = (float(sub.iloc[-1]) / float(prior.iloc[-1])) ** 0.1 - 1 if len(prior) else float("nan")
            elif m == "trend20y":
                w = sub[sub.index >= end - timedelta(days=int(365.25 * 20))]
                r = float(np.exp(np.polyfit(np.arange(len(w)) / 52.18, np.log(w.values), 1)[0]) - 1) if len(w) > 100 else float("nan")
            else:
                prior = sub[sub.index <= end - timedelta(days=30)]
                r = (float(sub.iloc[-1]) / float(prior.iloc[-1])) ** (365.25 / 30) - 1 if len(prior) else float("nan")
            row[f"rm_{m}"] = round(r + add, 5) if r == r else None
            row[f"erp_{m}"] = round(r + add - row["rf"], 5) if r == r else None
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(data_dir) / "market" / "erp_series.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def erp_at(as_of: str | date | None, method: str = "hist30y", data_dir: Path = Path("data")) -> dict | None:
    """ERP from the saved series at the latest month end on or before as_of."""
    p = Path(data_dir) / "market" / "erp_series.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(_d(as_of))]
    if not len(df):
        return None
    r = df.iloc[-1]
    v = r.get(f"erp_{method}")
    if v is None or v != v:
        return None
    return {"value": float(v), "method": method, "date": r["date"].strftime("%Y-%m-%d"), "rf": float(r["rf"]), "rf_date": r["rf_date"],
            "source": f"ERP series ({method}) at {r['date'].strftime('%Y-%m-%d')}: Rm {float(r[f'rm_{method}']):.2%} minus DGS10 {float(r['rf']):.2%} (rf observation {r['rf_date']}); data/market/erp_series.csv"}
