# Roadmap

v0.9 closed most of this: decisions (segments, unit economics, KPIs, peers, pricing power, cycle, alignment, normalization, methods, valuation, DCF variant, scenarios) run through `choose` / `review` / `export`; ERP is a series; scenarios, elasticity, NWC options, TSM dilution and the audit sheet are in the model; quarterly history exists. v0.10 added the industry driver template (bitcoin miner) with KPI fitting and driver-level debates. Still open: a second template (orders-to-backlog for the data-center suppliers), organic vs M&A split, lease/SBC identification by the model merged into the statements, multiples / SOTP / reverse DCF sheets, roll-forward.: from SEC filings to an analyst's model

Each translation below is broken into steps. "Does" says what the step produces; "Done" says what the finished
state looks like. Status reflects v0.7.

Design rule for every step: deterministic code produces numbers; the LLM proposes links and locates text; a validator
checks proposals against the filing; an analyst accepts; the acceptance is stored in the mapping ledger and reused.

## 1. Line items (statement captions and tags into one schema) — done with residuals

| Step | Does | Done looks like |
|---|---|---|
| 1.1 Rules as data | Mapping rules live in `crucible/rules/schema_rules.csv` (concepts, labels, regexes, signs), editable without code | An analyst adds a concept for a new company by editing a row, not Python |
| 1.2 Candidate matching | Every rule that fires is a candidate with a confidence (concept 0.95, exact label 0.85, regex 0.60, sum 0.70); the best wins, the rest are alternatives | Every mapped line shows how it was mapped and what else it could have been |
| 1.3 Ledger | `data/mapping_ledger.csv` records every link per company: source, target, confidence, alternatives, evidence, who decided | The ledger is the complete audit trail; rules are only the first guess |
| 1.4 Residuals | Unmapped amounts fall into residual lines so totals tie; residual size per statement is the quality metric | Residuals under 5% of revenue or assets, or explained |
| 1.5 LLM proposals | `map-suggest`: for unmapped lines above materiality, the model proposes a target with confidence and alternatives; sources are constrained to what we offered | Nothing enters the model without an accepted ledger row |
| 1.6 Analyst decisions | `map-approve`, or Excel export and import; accepted rows override rules on every later run, rejected rows are never used | One decision per line per company, ever |
| 1.7 Custom concepts | Full-filing XBRL facts (`xbrl_facts_<FY>.parquet`) expose company extension tags so they can be linked like any other | MARA's energy cost and bitcoin holdings map through the ledger |

## 2. Periods — partly done

| Step | Does | Done looks like |
|---|---|---|
| 2.1 Fiscal year labels | Facts are labeled by the 10-K whose period they close (Modine March, Powell September) | Peer growth series line up by fiscal year |
| 2.2 Consecutive runs | Gap years and shell years (no revenue) are dropped before growth math | No zero-revenue years, no growth across a gap |
| 2.3 Restatements | Latest filing wins per concept and period (`--as-of` freezes what was known then) | Backtests never see later restatements |
| 2.4 Quarterly build | 10-Q facts into a quarterly history; Q4 = FY minus nine months; TTM | Quarterly model rows and trailing-four-quarter metrics |
| 2.5 Discontinued operations and M&A | Flag periods with discontinued ops or acquisitions from the CF and notes; organic vs reported growth from the 8-K bridge | Growth evidence split organic / M&A / FX where the company discloses it |

## 3. Segments and KPIs — backlog and guidance only

| Step | Does | Done looks like |
|---|---|---|
| 3.1 Segment facts | Dimensional XBRL facts (segment axis) into a segment table per period | Segment revenue and margin history per company |
| 3.2 KPI extraction | LLM locates KPIs in MD&A and releases (backlog, orders, units, ARR, hashrate) and proposes verbatim spans; validator checks the number exists | KPI series with citations, analyst-approved once |
| 3.3 KPI ledger | Same ledger mechanics as line items: source phrase, target KPI, confidence, alternatives | A KPI found once is found the same way next quarter |
| 3.4 Driver evidence | KPI series enter the packet as direct evidence with units | The debate argues from backlog conversion, not from the trend alone |

## 4. Non-GAAP — not done

| Step | Does | Done looks like |
|---|---|---|
| 4.1 Reconciliation capture | Parse the 8-K reconciliation table (GAAP to adjusted EBITDA / EPS) into adjustment lines | Every adjustment named and sized per quarter |
| 4.2 Adjustment policy | Analyst marks each adjustment accept / reject (stock comp, restructuring, amortization) in a policy file | The model's "adjusted" numbers reflect the analyst's policy, applied consistently |
| 4.3 Evidence | Adjustment history enters the packet (how often, how large) | The bear can cite recurring "one-offs" |

## 5. Normalization — not done

| Step | Does | Done looks like |
|---|---|---|
| 5.1 Stock comp | Choice: expense as reported or add back in FCF; recorded in the analyst inputs | One consistent treatment in DCF and comparables |
| 5.2 Leases | Operating lease liabilities as debt-like, with the matching expense adjustment | Net debt and EV consistent across peers |
| 5.3 One-offs | Impairments, litigation, restructuring flagged from facts; excluded from margin trends when material | Margin evidence is on a clean basis |
| 5.4 Pension and FX | Pension items and FX effects identified and isolated | Residuals shrink; evidence stays clean |

## 6. Share count — basic

| Step | Does | Done looks like |
|---|---|---|
| 6.1 Cover-page shares | Latest dei shares outstanding for market cap (done) | Market cap at the as-of date |
| 6.2 Dilution | RSUs, options and convertibles from the notes; treasury stock method at the as-of price | Diluted share count that moves with the price |
| 6.3 Buyback path | Buyback history as a driver in the model | Share count forecast, not a constant |

## 7. Capital structure — basic

| Step | Does | Done looks like |
|---|---|---|
| 7.1 Debt schedule | Maturities and rates from the debt note (LLM locates, validator checks) | Refinancing years and cost of debt by tranche |
| 7.2 Cash-like assets | Policy per company for short-term investments, restricted cash, digital assets | Net debt definition stated and consistent |
| 7.3 Market weights | Market-value debt weight from cover-page shares and price (done) | WACC on market weights with the book weight shown |

## 8. Driver decomposition — not done

| Step | Does | Done looks like |
|---|---|---|
| 8.1 Templates | Industry driver templates as data (generic, miner, data-center supplier): which KPIs multiply into revenue and cost | A new company picks a template; the analyst overrides any line |
| 8.2 Driver history | KPI series (step 3) fitted to the template historically | Revenue = price × volume reconciled to reported revenue with a residual |
| 8.3 Driver debates | Assumptions become KPI-level (hashrate, hashprice, backlog conversion) instead of a revenue growth rate | The crux is about the driver, which is what a PM asks about |

## 9. Roll-forward (model maintenance) — not done

| Step | Does | Done looks like |
|---|---|---|
| 9.1 New-filing diff | On a new 10-Q/10-K, refresh facts, re-map through the ledger, diff every mapped line and every unmapped residual against the last run | A change report: what moved, what is new, what no longer maps |
| 9.2 Definition checks | Detect concept changes, restatements, new custom tags | Breaks surface as pending ledger rows, never as silent zeros |
| 9.3 Workbook refresh | Rewrite historical columns; forecast formulas untouched; analyst overrides kept | The model updates in one command and the formulas survive |
| 9.4 Decision log | Every accept/edit/reject kept with the filing it was made on | The maintenance history is itself data |

## Attribution without one debate per removal

Two estimators run beside leave-one-out, and all three record their metrics:

- Random-subset design (`harness --ablation random --random-k 16`): each debate gets a random subset of evidence groups;
  OLS of the base on the inclusion indicators gives every group's effect at once with bootstrap intervals and an R².
- Fit on logs (`attribution-fit`): OLS on all cached runs for a packet family, no API calls; where backtest runs carry a
  realized outcome, a logistic fit of hit-or-miss and an OLS of realized-minus-base on group inclusion say which
  evidence moved the base toward reality. Observational, so read with the R² and the intervals.

Done looks like: for each company and assumption, a table of evidence groups with effect, interval, and whether the
effect is toward or away from realized outcomes, refreshed from logs as they accumulate.
