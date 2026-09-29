"""Industry driver templates as data, KPI series fitted to them, driver-level debates.

A template names the operating drivers a business actually runs on and how revenue, cash costs, capex and depreciation
follow from them. The KPI fitter reads the drivers' history out of the earnings releases (verbatim sentences kept as
citations), the build turns driver paths into the per-year growth, margin, D&A and capex ratios the three-statement
model consumes (so every downstream formula still ties), and each driver can be debated with the numeric engine.

bitcoin_miner
    btc_mined      = own hashrate / network hashrate x network issuance (blocks x subsidy x (1 + fee share))
    revenue        = btc_mined x realized price + other revenue (hosting, energy)
    energy cost    = hashrate x efficiency (J/TH) x hours x power price ($/MWh)
    capex          = new hashrate x $/EH + replacement of the retiring fleet; D&A straight-line over the fleet life
    equity value   = EV - net debt + bitcoin held x price (the treasury is a non-operating asset)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import AssumptionSpec, EvidenceItem, EvidencePacket

BLOCKS_PER_DAY = 144.0
HALVINGS = [("2020-05-11", 6.25), ("2024-04-20", 3.125), ("2028-04-15", 1.5625), ("2032-04-01", 0.78125)]  # subsidy from each date
J_PER_MWH = 3.6e9


@dataclass
class DriverSpec:
    key: str
    label: str
    unit: str
    role: str            # "input" (analyst path) | "derived"
    note: str = ""


@dataclass
class Template:
    key: str
    label: str
    drivers: list[DriverSpec]
    kpi_columns: tuple[str, ...]
    assumptions: dict[str, AssumptionSpec] = field(default_factory=dict)


MINER_DRIVERS = [
    DriverSpec("hashrate_eh", "average energized hashrate", "EH/s", "input", "the fleet actually mining; releases give quarter-end energized EH/s"),
    DriverSpec("network_hashrate_eh", "effective network hashrate", "EH/s", "input", "energized EH/s divided by the realized block share; includes the company's own uptime and curtailment losses, so it sits above the published network figure and reproduces realized production"),
    DriverSpec("btc_price", "realized bitcoin price", "USD", "input", "average price of bitcoin produced in the period"),
    DriverSpec("fee_share", "transaction fees as a share of the subsidy", "pct", "input", "BTC produced / (blocks won x subsidy) - 1"),
    DriverSpec("efficiency_j_th", "fleet efficiency", "J/TH", "input", "energy per terahash; falls with fleet upgrades"),
    DriverSpec("power_price_mwh", "blended power and hosting cost", "USD/MWh", "input", "owned-site energy plus third-party hosting per MWh consumed"),
    DriverSpec("other_cash_cogs_pct", "other cash cost of revenue", "pct of revenue", "input", "operations and maintenance and other non-energy cost of revenue"),
    DriverSpec("other_revenue_pct", "non-mining revenue share", "pct of revenue", "input", "hosting, energy sales, other"),
    DriverSpec("capex_per_eh", "capex per EH/s added", "USD m", "input", "miners plus infrastructure per unit of new hashrate"),
    DriverSpec("fleet_life_years", "fleet life", "years", "input", "depreciation and replacement horizon for miners"),
    DriverSpec("btc_sold_pct", "share of production sold", "pct", "input", "the rest accrues to the treasury"),
    DriverSpec("btc_held", "bitcoin held", "BTC", "input", "treasury at the last balance date; valued at the price path as a non-operating asset"),
    DriverSpec("btc_mined", "bitcoin produced", "BTC", "derived"),
    DriverSpec("energy_cost_per_btc", "energy and hosting cost per bitcoin", "USD", "derived"),
]

MINER_ASSUMPTIONS = {
    "hashrate_growth_3y": AssumptionSpec(key="hashrate_growth_3y", description="Average annual growth of the company's energized hashrate over the next three fiscal years (decimal)",
                                         unit="pct", bull_direction="up", lower_bound=-0.5, upper_bound=2.0),
    "network_hashrate_growth_3y": AssumptionSpec(key="network_hashrate_growth_3y", description="Average annual growth of the bitcoin network hashrate over the next three fiscal years (decimal); faster network growth dilutes the company's share",
                                                 unit="pct", bull_direction="down", lower_bound=-0.3, upper_bound=1.5),
    "btc_price_change_3y": AssumptionSpec(key="btc_price_change_3y", description="Average annual change in the realized bitcoin price over the next three fiscal years (decimal)",
                                          unit="pct", bull_direction="up", lower_bound=-0.6, upper_bound=2.0),
}

TEMPLATES = {
    "bitcoin_miner": Template("bitcoin_miner", "bitcoin miner: hashrate x network share x price", MINER_DRIVERS,
                              ("hashrate_eh", "blocks_won", "btc_produced", "energy_cost_per_btc", "avg_price_produced", "btc_sold", "avg_price_sold", "btc_held"),
                              MINER_ASSUMPTIONS),
}


def subsidy_on(day: pd.Timestamp) -> float:
    s = 6.25
    for d, v in HALVINGS:
        if day >= pd.Timestamp(d):
            s = v
    return s


def issuance_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    """(blocks, subsidy BTC) mined network-wide between two dates, halving-aware."""
    blocks = btc = 0.0
    day = pd.Timestamp(start)
    while day < end:
        nxt = min(end, day + pd.Timedelta(days=1))
        frac = (nxt - day).total_seconds() / 86400
        blocks += BLOCKS_PER_DAY * frac
        btc += BLOCKS_PER_DAY * frac * subsidy_on(day)
        day = nxt
    return blocks, btc


# ----------------------------------------------------------------------------- KPI fitting from releases

_Q = re.compile(r"Q\s?([1-4])\s?(20\d\d)")
_NUM = r"([\d,]+(?:\.\d+)?)"
MINER_PATTERNS = {
    "hashrate_eh": re.compile(r"(?:energized\s+)?hash\s?rate[^.]{0,60}?(?:to|of|reached|was)\s*" + _NUM + r"\s*EH", re.I),
    "blocks_won": re.compile(r"(?:number of\s+)?blocks won\s+" + _NUM + r"\s+" + _NUM, re.I),
    "btc_produced": re.compile(r"BTC produced\s+" + _NUM + r"\s+" + _NUM + r"|produced\s+" + _NUM + r"\s+(?:BTC|bitcoin)\s+at an average price", re.I),
    "energy_cost_per_btc": re.compile(r"(?:purchased\s+)?energy cost per (?:bitcoin|BTC)(?: for our owned(?: and operated)? sites)? (?:was|of)\s*\$\s?" + _NUM, re.I),
    "avg_price_produced": re.compile(r"produced\s+[\d,]+\s+(?:BTC|bitcoin)\s+at an average price of\s*\$\s?" + _NUM, re.I),
    "btc_sold": re.compile(r"sold\s+" + _NUM + r"\s+(?:BTC|bitcoin)\s+at an average price of\s*\$\s?" + _NUM, re.I),
    "btc_held": re.compile(r"(?:we\s+)?held\s+" + _NUM + r"\s+(?:BTC|bitcoin)", re.I),
}


def _f(x: str) -> float:
    return float(str(x).replace(",", ""))


_BOUND = re.compile(r"[.!?]\s+(?=[A-Z\u2013\u2014-])")


def _sentence_at(text: str, pos: int) -> str:
    """The sentence around ``pos``; boundaries are sentence ends followed by a capital, so decimals do not split it."""
    starts = [m.end() for m in _BOUND.finditer(text[:pos + 1]) if m.end() <= pos]
    s = starts[-1] if starts else 0
    m = _BOUND.search(text, pos)
    e = m.start() + 1 if m else min(len(text), pos + 200)
    return text[s:e].strip()[:300]


def fit_kpis(releases: list[dict], template: str = "bitcoin_miner") -> pd.DataFrame:
    """One row per quarter from the earnings releases (newest release wins for a quarter), with the verbatim sentence and
    filing for every value. Columns: quarter, period_end, <kpi>..., <kpi>_cite..., filing_date, accession."""
    if template != "bitcoin_miner":
        return pd.DataFrame()
    rows: dict[str, dict] = {}
    for rel in sorted(releases, key=lambda r: r.get("filing_date", "")):
        text = re.sub(r"\s+", " ", rel.get("text", ""))
        head = text[:6000]
        mq = _Q.search(head) or _Q.search(text)
        if not mq:
            continue
        q, y = int(mq.group(1)), int(mq.group(2))
        label = f"{y}Q{q}"
        end = pd.Timestamp(f"{y}-{3 * q:02d}-01") + pd.offsets.MonthEnd(0)
        row = rows.get(label, {"quarter": label, "period_end": end.strftime("%Y-%m-%d"), "filing_date": rel.get("filing_date", ""), "accession": rel.get("accession", "")})
        cite = f"8-K EX-99.1 filed {rel.get('filing_date', '?')}"
        for key, pat in MINER_PATTERNS.items():
            m = pat.search(text)
            if not m:
                continue
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            try:
                val = _f(groups[0])
            except ValueError:
                continue
            if key == "hashrate_eh" and not (1 <= val <= 2000):
                continue
            row[key] = val
            row[f"{key}_cite"] = f"{cite} | \"{_sentence_at(text, m.start())}\""
            if key == "btc_sold" and len(groups) > 1:
                row["avg_price_sold"] = _f(groups[1])
                row["avg_price_sold_cite"] = row[f"{key}_cite"]
        rows[label] = row
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["quarter"]))
    if "blocks_won" in df.columns:
        blocks, subs = zip(*(issuance_between(pd.Timestamp(e) - pd.offsets.QuarterEnd(1) + pd.Timedelta(days=1), pd.Timestamp(e) + pd.Timedelta(days=1)) for e in df.period_end))
        df["network_blocks"] = np.round(blocks, 0)
        df["network_share"] = (df.blocks_won / df.network_blocks).round(5)
        if "hashrate_eh" in df.columns:
            df["network_hashrate_eh"] = (df.hashrate_eh / df.network_share).round(1)
        if "btc_produced" in df.columns:
            df["subsidy_btc"] = [round(b * subsidy_on(pd.Timestamp(e)), 1) for b, e in zip(df.blocks_won, df.period_end)]
            df["fee_share"] = (df.btc_produced / df.subsidy_btc - 1).round(4)
    return df


def annual_kpis(q: pd.DataFrame) -> pd.DataFrame:
    """Fiscal-year drivers from the quarterly table (calendar fiscal years): sums for flows, means for rates."""
    if not len(q):
        return pd.DataFrame()
    x = q.copy()
    x["fy"] = x.quarter.str[:4].astype(int)
    agg = {}
    for c in ("blocks_won", "btc_produced", "btc_sold", "network_blocks", "subsidy_btc"):
        if c in x.columns:
            agg[c] = "sum"
    for c in ("hashrate_eh", "network_hashrate_eh", "energy_cost_per_btc", "avg_price_produced", "avg_price_sold", "network_share", "fee_share"):
        if c in x.columns:
            agg[c] = "mean"
    a = x.groupby("fy").agg(agg)
    a["quarters"] = x.groupby("fy").size()
    if "btc_held" in x.columns:
        a["btc_held"] = x.groupby("fy").btc_held.last()
    return a


# ----------------------------------------------------------------------------- the driver build

def miner_defaults(hist: pd.DataFrame, q: pd.DataFrame, years: list[int]) -> dict:
    """Driver paths from the fitted history: hashrate grows at its trailing rate (tapering), the network at its own,
    price flat at the last realized level, efficiency and power price at the last year's implied values."""
    a = annual_kpis(q)
    last_fy = int(hist.columns[-1])
    rev_last = float(hist.loc["revenue"].iloc[-1])
    full = a[a.quarters == 4] if len(a) else a
    base = full.iloc[-1] if len(full) else (a.iloc[-1] if len(a) else None)
    n = len(years)

    def path(start: float, g: float, taper: float = 0.7) -> list[float]:
        out, v, gg = [], start, g
        for _ in range(n):
            v = v * (1 + gg)
            out.append(round(v, 4))
            gg *= taper
        return out

    if base is None:
        raise ValueError("no KPI history fitted; run `kpis` after ingest")
    hr = float(base.get("hashrate_eh", np.nan))                       # last full fiscal year average (the capex base)
    hr_prev = float(full.iloc[-2].hashrate_eh) if len(full) > 1 and "hashrate_eh" in full.columns else hr / 1.3
    net = float(base.get("network_hashrate_eh", np.nan))
    # growth from the latest quarter against the same quarter a year earlier (the current run-rate, not the FY average
    # which mixes acquisition timing); the first forecast year averages the latest level and a half year of that growth
    def _yoy(col: str, fallback: float) -> tuple[float, float]:
        s_ = q.dropna(subset=[col]) if col in q.columns else q.iloc[0:0]
        if len(s_) >= 5:
            cur, prev = float(s_[col].iloc[-1]), float(s_[col].iloc[-5])
            return cur, (cur / prev - 1) if prev else fallback
        return (float(s_[col].iloc[-1]) if len(s_) else np.nan), fallback
    hr_now, hr_g = _yoy("hashrate_eh", 0.25)
    net_now, net_g = _yoy("network_hashrate_eh", 0.30)
    btc = float(base.get("btc_produced", np.nan))
    price = float(base.get("avg_price_produced", np.nan))
    if price != price:  # the realized-price sentence only appears in newer releases: take the latest quarter that has it
        px = q.dropna(subset=["avg_price_produced"]) if "avg_price_produced" in q.columns else q.iloc[0:0]
        price = float(px.avg_price_produced.iloc[-1]) if len(px) else (rev_last * 1e6 * 0.9 / btc if btc == btc and btc else np.nan)
    fee = float(base.get("fee_share", 0.05)) if base.get("fee_share") == base.get("fee_share") else 0.05
    energy_pb = float(base.get("energy_cost_per_btc", np.nan))
    # energy + hosting from the mapped cost lines when the ledger split them, else the disclosed per-BTC energy cost
    # non-mining revenue share: FY revenue minus BTC produced x the realized price of that year (falls back to zero when
    # the realized price of the base year is unknown, because a later quarter's price would misstate the split)
    price_base = float(base.get("avg_price_produced", np.nan))
    mining_rev_m = btc * price_base / 1e6 if btc == btc and price_base == price_base else np.nan
    other_rev_pct = float(np.clip(1 - mining_rev_m / rev_last, 0.0, 0.6)) if rev_last and mining_rev_m == mining_rev_m else 0.0
    eff = 20.0  # J/TH assumed for the fleet when the filings do not state it; the analyst overrides
    energy_total = energy_pb * btc if energy_pb == energy_pb and btc == btc else np.nan  # USD, energy only
    mwh_year = hr * 1e6 * eff * 8760 / 1e6 if hr == hr else np.nan  # W = TH x J/TH; MWh = W x hours / 1e6
    power_price = float(energy_total / mwh_year) if energy_total == energy_total and mwh_year else 45.0
    cogs_last = float(hist.loc["cogs"].iloc[-1])
    da_last = float(hist.loc["d_and_a"].iloc[-1])
    other_cash = max(0.0, cogs_last - da_last - (energy_total / 1e6 if energy_total == energy_total else 0.0))
    other_cash_pct = other_cash / rev_last if rev_last else 0.05
    capex_last = float(hist.loc["capex"].iloc[-1])
    d_hr = hr - hr_prev if hr_prev else max(hr * 0.2, 1.0)
    capex_per_eh = max(5.0, min(60.0, capex_last / d_hr)) if d_hr > 0 else 25.0
    hr_path = path(hr_now * (1 + hr_g / 2) / (1 + hr_g), hr_g) if hr_now == hr_now else path(hr, hr_g)
    net_path = path(net_now * (1 + net_g / 2) / (1 + net_g), net_g) if net_now == net_now else path(net, net_g)
    return {
        "hashrate_eh": hr_path, "network_hashrate_eh": net_path, "btc_price": [round(price, 0)] * n,
        "fee_share": [round(fee, 4)] * n, "efficiency_j_th": [round(eff * (0.93 ** i), 2) for i in range(1, n + 1)],
        "power_price_mwh": [round(power_price, 1)] * n, "other_cash_cogs_pct": [round(other_cash_pct, 4)] * n,
        "other_revenue_pct": [round(other_rev_pct, 4)] * n, "capex_per_eh": [round(capex_per_eh, 1)] * n,
        "fleet_life_years": 3.0, "btc_sold_pct": [1.0] * n, "hashrate_last": round(hr, 3),
        "btc_held": float(q.dropna(subset=["btc_held"]).btc_held.iloc[-1]) if "btc_held" in q.columns and q.btc_held.notna().any() else 0.0,
        "basis": (f"FY{int(full.index[-1]) if len(full) else last_fy} KPIs: hashrate {hr:,.1f} EH/s avg, latest {hr_now:,.1f} ({hr_g:+.0%} y/y), network {net:,.0f} EH/s avg, latest {net_now:,.0f} ({net_g:+.0%} y/y), "
                  f"price {price:,.0f}, energy {energy_pb:,.0f}/BTC -> {power_price:,.0f}/MWh at {eff} J/TH, other cash cogs {other_cash_pct:.1%}, "
                  f"capex {capex_per_eh:,.0f}m per EH/s added, {btc:,.0f} BTC produced"),
    }


def miner_build(hist: pd.DataFrame, years: list[int], d: dict) -> pd.DataFrame:
    """Per-year driver outputs and the ratios the three-statement model consumes."""
    n = len(years)
    rows = []
    last_fy = int(hist.columns[-1])
    ppe0 = float(hist.loc["ppe_net"].iloc[-1])
    life = float(d.get("fleet_life_years", 3.0))
    capex_hist: list[float] = []
    hr_prev = float(d["hashrate_last"]) if d.get("hashrate_last") else None
    held = float(d.get("btc_held", 0.0))
    for i, y in enumerate(years):
        hr, net, price = d["hashrate_eh"][i], d["network_hashrate_eh"][i], d["btc_price"][i]
        blocks, subs = issuance_between(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01"))
        share = hr / net if net else 0.0
        btc = share * subs * (1 + d["fee_share"][i])
        mining_rev = btc * price / 1e6
        revenue = mining_rev / (1 - d["other_revenue_pct"][i]) if d["other_revenue_pct"][i] < 1 else mining_rev
        mwh = hr * 1e6 * d["efficiency_j_th"][i] * 8760 / 1e6      # W = TH x J/TH; MWh = W x h / 1e6
        energy = mwh * d["power_price_mwh"][i] / 1e6
        other_cash = revenue * d["other_cash_cogs_pct"][i]
        # capex: new hashrate plus replacement of the fleet retiring this year
        prev_hr = hr_prev if hr_prev is not None else hr
        added = max(hr - prev_hr, 0.0)
        replacement = prev_hr / life
        capex = (added + replacement) * d["capex_per_eh"][i]
        capex_hist.append(capex)
        # depreciation: legacy PP&E straight-line over the remaining life, new capex over the fleet life
        legacy = ppe0 / life if i < int(round(life)) else 0.0
        new_da = sum(c / life for c in capex_hist[-int(round(life)):])
        da = legacy + new_da
        cogs = energy + other_cash + da
        held = held + btc * (1 - d["btc_sold_pct"][i])
        rows.append({"year": y, "hashrate_eh": hr, "network_hashrate_eh": net, "network_share": share, "issuance_btc": subs, "btc_mined": btc,
                     "btc_price": price, "mining_revenue": mining_rev, "revenue": revenue, "energy_mwh": mwh, "energy_cost": energy,
                     "energy_cost_per_btc": energy * 1e6 / btc if btc else np.nan, "other_cash_cogs": other_cash, "d_and_a": da, "cogs": cogs,
                     "gross_margin": (revenue - cogs) / revenue if revenue else np.nan, "capex": capex, "capex_pct": capex / revenue if revenue else np.nan,
                     "da_pct": da / revenue if revenue else np.nan, "btc_held_end": held, "treasury_value": held * price / 1e6})
        hr_prev = hr
    b = pd.DataFrame(rows).set_index("year")
    rev_prev = float(hist.loc["revenue"].iloc[-1])
    g = []
    for y in years:
        g.append(float(b.at[y, "revenue"]) / rev_prev - 1 if rev_prev else 0.0)
        rev_prev = float(b.at[y, "revenue"])
    b["revenue_growth"] = g
    return b


def apply_build_to_drivers(drv, build: pd.DataFrame):
    """Overwrite the per-year ratios the model uses with the template build (the formulas downstream stay the same)."""
    drv.revenue_growth = [float(x) for x in build.revenue_growth]
    drv.gross_margin = [float(x) for x in build.gross_margin]
    drv.da_pct = [float(x) for x in build.da_pct]
    drv.capex_pct = [float(x) for x in build.capex_pct]
    drv.non_operating_assets = float(build.treasury_value.iloc[-1] if "treasury_value" in build.columns else 0.0)
    return drv


# ----------------------------------------------------------------------------- driver-level evidence

def driver_evidence(key: str, q: pd.DataFrame, company: str, start_id: int = 1) -> list[EvidenceItem]:
    """KPI history as evidence for one driver debate: the quarterly series, year-over-year changes, and the sentences."""
    items: list[EvidenceItem] = []
    if not len(q):
        return items
    k = start_id
    col = {"hashrate_growth_3y": "hashrate_eh", "network_hashrate_growth_3y": "network_hashrate_eh", "btc_price_change_3y": "avg_price_produced"}[key]
    label = {"hashrate_growth_3y": "energized hashrate (EH/s)", "network_hashrate_growth_3y": "implied network hashrate (EH/s)", "btc_price_change_3y": "realized bitcoin price (USD)"}[key]
    if col not in q.columns:
        return items
    s = q.dropna(subset=[col])
    series = ", ".join(f"{r.quarter} {r[col]:,.1f}" for _, r in s.tail(10).iterrows())
    items.append(EvidenceItem(id=f"E{k}", kind="historical_metric", source="8-K EX-99.1 releases (fitted KPI table)" + (" via blocks won / network blocks" if col == "network_hashrate_eh" else ""),
                              content=f"{company} {label} by quarter: {series}"))
    k += 1
    a = annual_kpis(s)
    a = a[a.quarters >= 3] if "quarters" in a.columns else a
    if len(a) > 1 and col in a.columns:
        for y in a.index[1:]:
            prev, cur = float(a.loc[y - 1, col]) if (y - 1) in a.index else np.nan, float(a.loc[y, col])
            if prev and prev == prev:
                g = cur / prev - 1
                items.append(EvidenceItem(id=f"E{k}", kind="historical_metric", source=f"8-K EX-99.1 releases, FY{y} vs FY{y - 1} averages", direct=True,
                                          content=f"{company} {label} FY{y}: {g:+.1%} ({cur:,.1f} vs {prev:,.1f})", value=g, unit="pct", period=f"FY{y}"))
                k += 1
    cite_col = f"{col}_cite" if f"{col}_cite" in q.columns else ("blocks_won_cite" if col == "network_hashrate_eh" else None)
    if cite_col:
        for _, r in s.tail(3).iterrows():
            c = r.get(cite_col)
            if isinstance(c, str) and c:
                src, _, quote = c.partition(" | ")
                items.append(EvidenceItem(id=f"E{k}", kind="filing_text", source=src + f" [kpi:{col}]", content=quote))
                k += 1
    return items


def load_kpis(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = Path(data_dir) / ticker.upper() / "kpi_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def save_kpis(data_dir: Path, ticker: str, q: pd.DataFrame) -> Path:
    p = Path(data_dir) / ticker.upper() / "kpi_history.csv"
    q.to_csv(p, index=False)
    return p
