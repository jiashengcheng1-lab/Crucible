# Changelog

## 0.11.1 (2026-09-29) - questions see the model's own state; Excel preview
- Follow-up questions now have three kinds of context: the model's own state (the three statements as mapped now with the source of every line, residual lines, mapping notes, the forecast and DCF as last built, the workbook's sheets), the filing sentences, and the dossier. A question that names a statement ("what's under the balance sheet now?") gets every line of that statement; "unsupported" now means unsupported by the filings or the model state.
- `model` persists `outputs/<T>_forecast.csv` and `outputs/<T>_model_summary.json` (drivers, DCF, scenarios, checks, warnings), which both the questions and the UI read.
- UI: a Statements tab (income statement, balance sheet, cash flow as mapped, residual lines, model as last built), the statement of the row under review inside the Review tab, and an Excel preview of any workbook and sheet in the Model tab (values as stored; formula cells show their text).

## 0.11.0 (2026-09-29) - industry from the filings, model-confirmed citations, offline review, follow-up questions, UI, public repo
- Company profile (`profile`): industry from the filing header (SIC code and EDGAR description) placed in a family with a generic template; business model, revenue and cost drivers, KPIs, unit-economics and KPI-scheme candidates, company-specific driver templates and named competitors read by the model from Item 1 and the release KPIs, every element carrying a verbatim quote verified against the filing (unverifiable ones dropped and counted). The decision specs now take their options from the profile: a phone maker no longer sees hashrate options, a miner no longer sees ARR.
- Citations in map-suggest are confirmed by the model: proposals without a model-chosen sentence are re-offered a wider set (the line's words plus the target's) and the model picks or declines; ledger citations read "model-selected", "model-confirmed (conf)" or "no supporting sentence found by the model among N candidates". The keyword-only "auto" label remains only with `--no-confirm`.
- `review-export` / `review-import`: one workbook with every pending proposal and decision and all fields (proposed target, confidence, alternatives, evidence, target now mapped from, reasoning, citation, questions, considerations) plus DECISION, TARGET_OVERRIDE, NOTE and FOLLOW_UP_QUESTION columns; import applies the decisions and answers the questions.
- Follow-up questions: `?question` in the terminal review, `ask` on the command line, a box in the UI; answers come from the filings only (deterministic retrieval over Items 1, 1A, 7, 8 and the dossier), cite the sentences they rest on, carry a confidence, and are logged to `logs/qa.jsonl` and into the ledger note of the row under review.
- Streamlit UI (`crucible-ui`): run steps, review rows with all fields and the question box, brief and profile, model summary and downloads, debates, question log. `crucible/ui/service.py` keeps the logic testable without Streamlit.
- Fixes: option labels returned with a description ("reverse DCF: back out...") match the right option; the "target now mapped from" line is shown for mapping rows only; a zero interest-expense fact no longer yields a 0% cost of debt (proxy at the risk-free rate plus 100bp, flagged).
- Public repository packaging: MIT license, .gitignore for data, outputs and secrets, GitHub Actions CI (pytest and the offline demo on 3.11 and 3.12), .env.example, CONTRIBUTING, README rewritten.

## 0.10.0 (2026-09-25) - industry driver templates: the bitcoin miner runs on hashrate, network share and price
- `crucible/templates.py`: templates as data (drivers, units, roles), a KPI fitter that reads each driver's history out of the earnings releases with the verbatim sentence kept as the citation (energized hashrate, blocks won, BTC produced, energy cost per BTC, realized price of BTC produced and sold, BTC held), halving-aware network issuance, an effective network hashrate implied from blocks won over total blocks (it includes the company's own uptime and curtailment losses, so realized production is reproduced), and the miner build: BTC mined = own / network hashrate x issuance x (1 + fee share); revenue = BTC mined x realized price plus the non-mining share; energy MWh = EH/s x J/TH x 8760 at a blended $/MWh; capex = new hashrate plus the retiring fleet x $m per EH/s; depreciation straight-line over the fleet life (legacy PP&E and new capex); the bitcoin treasury as a non-operating asset in the equity bridge.
- `kpis TICKER`: fit and print the KPI table (with `--cites`); `model TICKER --template auto|bitcoin_miner|none` (auto = the accepted industry_template decision) applies the build through the same per-year ratios the three statements consume, so every downstream formula still ties; `--template-inputs` overrides any driver path.
- Workbook: a Drivers sheet with the driver paths (blue) and the build as formulas; the model's growth, gross margin, D&A and capex inputs link to it (through the Scenarios base rows when scenarios are on, with bull and bear as base plus the tapered shift); the DCF equity value adds the treasury row.
- Driver-level debates: `hashrate_growth_3y`, `network_hashrate_growth_3y` (bull direction down) and `btc_price_change_3y` run on the existing numeric engine with the fitted KPI series, year-over-year changes and the release sentences as evidence; a verdict on disk replaces the default path for that driver at the next `model`.
- Defaults come from the latest quarter's year-over-year run-rate, not the fiscal-year average (which mixes acquisition timing); prices flat at the last realized level; efficiency assumed 20 J/TH and improving 7% a year until the analyst sets it.
- Fixes: sentences quoted from releases no longer split at decimals; peer and driver CAGR items no longer reuse evidence ids.

## 0.9.1 (2026-09-25) - fixes from the first MARA run of v0.9
- Accepted links are components, never overrides. Before: an accepted concept link ranked above the schema total, so accepting `CostOfGoodsAndServicesSoldDepreciationAndAmortization -> cogs` made MARA's cost of revenue equal to depreciation in every year, `PaymentsToInvestInDecommissioningFund -> cfi` replaced net investing cash flow with bitcoin purchases (and zeroed 2022-2023), and the model built on that. Now the schema total wins wherever it has a value; linked concepts fill the years the total lacks (MARA FY2025, where the 10-K stopped tagging a total) and are summed when several are linked (payment-style cash-flow concepts negated; twins carrying the same fact counted once); a linked zero never counts; where both exist the components are checked against the total and one note per line reports the differences. Same semantics for label links in the statement path. Rejecting the schema total hands the line to the components. `unmapped facts` no longer lists linked concepts.
- Gross profit is the identity revenue - cogs whenever both are on the statement; a reported figure that disagrees (edgartools' "Gross Profit (Calculated)") is noted and replaced.
- Review: rows come most material first; each shows what the proposed target is currently mapped from; a decision is copied to twin rows carrying the same values and target (a label and its concept) with a note; rows the model failed to rank are flagged.
- `choose --cheap`: the ranker's labels are matched leniently (case, punctuation, value, prefix, fuzzy) with one retry using exact labels; unmatched options are marked as such instead of accepted at 0.00. `market_cycle` stays a pairwise tournament under `--cheap` (judge only). Call estimate corrected. `--reset` replaces rows the analyst already decided (rerun after new evidence); otherwise they are kept and said so.
- ERP: default method is now the trailing 30-year total return (`hist30y`), computed against the risk-free rate on the same as-of date; every method's ERP is listed beside the chosen one; a warning fires outside 2-8%. The 10-year method stays available and is labeled recency-biased. Cost of debt below the risk-free rate is flagged (convertible or zero-coupon notes).
- Dilution: only award facts within 15 months of the as-of date count; with none, the cover-page count is used and said so; convertible carrying values are reported when tagged.
- Default drivers: net gains inside operating income (fair-value gains on digital assets) are not forecast as recurring; capex is floored at D&A while revenue grows. Both are noted in the summary and on the Inputs sheet. The model summary warns when bull < base (negative unit economics), when value is negative, and when terminal value dominates.
- Quarterly: uses the analyst's links with the same component semantics; a linked sum is accepted only when every component present in the last 10-K is present in the quarter, so a 10-Q lacking the custom lines shows a gap instead of a wrong number. `ingest` now also saves full-filing 10-Q XBRL facts (`xbrl_facts_q_<period>.parquet`) so custom quarterly lines exist.
- Brief header says "latest filings on disk" when no as-of date is given.

## 0.9.0 (2026-09-26) - steps 1-5 of the research process run through one decision layer
- `choose`: every modeling decision (segment scheme, unit economics, KPI scheme, peer set, pricing power, market cycle, management alignment, normalization policy, revenue / cost / capex method, industry template, valuation method, DCF variant, scenario framing) is debated the same way: one advocate per option with verified quotes, a judge that returns confidence, reasoning, citations, settling questions and considerations per option; two options debate directly, more run a pairwise single-elimination tournament (winner advances). `--cheap` ranks with one call; results cache by packet hash.
- Decision ledger (`data/decision_ledger.csv`) with the mapping ledger's columns: statement = decision, source = option, target = value, relation = reasoning, citation = verbatim filing text, questions, duplicates = considerations. Analysts accept, reject, retarget or mark unsure with a note.
- `review`: terminal walk of pending mapping proposals and decisions (a / r / u / t<target> / n<note> / s / q), non-interactive with `--decisions-file`, `--export` at the end.
- `export`: final workbook (mapping_ledger, decision_ledger, brief sheets) plus `<T>_final_unsure.xlsx` with every unsure row from both ledgers.
- `brief`: what the company does, why now, what the debate is; each section carries evidence ids that resolve to filing sentences or computed metrics, written beside the text in the brief sheet.
- Dossier (`dossier`): one evidence packet per company shared by all decisions and the brief: growth, long-run history, costs, guidance and track record, KPI sentences from releases (hashrate, bitcoin mined, cost per bitcoin, backlog, orders, book-to-bill, MW, J/TH), segment revenue by member from dimensional XBRL, five-year capital allocation record and ROIC proxy, proxy-statement incentive metrics (DEF 14A mention counts and CD&A sentences), keyword-family filing sentences for each decision, the competition excerpt, peer CAGRs, FRED macro features (INDPRO y/y, UNRATE change, T10Y2Y, BAA10Y, DGS10).
- Ingest also saves dimensional segment facts (`xbrl_segments_<FY>.parquet`) and the latest proxy statement text (`proxy_text.json`); `set-peers` records the chosen peer tickers for the next `ingest --peers`.
- ERP is a dated series, not a number: `data/market/erp_series.csv` holds month-end rf (DGS10 on that date), expected market return by method (trailing 10-year total return [default], 20-year log-trend drift, trailing 30-day annualized [flagged noisy]) and ERP = Rm - rf; `inputs --erp-method` picks the method; Damodaran stays as a cross-check item.
- Model: bull / base / bear driver sets (growth shifts from the growth debate's low/high when a verdict exists, else +/-5pp; margin +/-1pp; WACC -/+50bp; 25/50/25) with a scenario selector cell (Inputs!B1) that switches the whole formula chain; Scenarios sheet with engine values and probability-weighted value; Elasticity sheet (each driver shocked alone); working capital by `--nwc-method days|pct_revenue`; diluted shares by the treasury stock method at the as-of price (options, RSUs; convertibles flagged) via `inputs`; `--audit-links` adds an Audit sheet listing every check row's cells and formulas (default off).
- `quarterly`: quarterly income statement history from the facts, Q4 = fiscal year minus nine months, as-of aware.

## 0.8.1 (2026-09-26)
- Proposer: relation is mandatory (falls back to the rationale, labeled); when the model cites nothing but the tool found a filing sentence, it is stored labeled 'auto (keyword match, not confirmed by the model)'; batches of 15 so every field gets filled; repeated runs pool relations and confirmed citations.
- Ingest refreshes per-year 10-K text files that predate the Item 8 notes; map-suggest warns when notes are missing and reports how many candidates have sentences to cite.

## 0.8.0 (2026-09-26)
- Ledger columns: relation (what the line is and what drives it), citation (accession | section | verbatim filing sentence, chosen from sentences the tool offered so it is verbatim by construction), questions (what must be settled to categorize), duplicates (lines this would double count: model-flagged plus deterministic same-value detection).
- Ingest saves Item 8 (financial statement notes) text per 10-K; the proposer feeds each candidate the notes/MD&A sentences that mention it.
- `map-suggest --runs N` repeats the proposer and reports the agreement rate per source; disagreements lower confidence and carry the other target as an alternative.

## 0.7.2 (2026-09-26)
- Excel export writes confidence as a number and version as an integer.
- `map-approve` with no decision given explains the three ways to decide instead of reporting zero rows; `--file` applies an edited Excel ledger; `map-import` defaults to outputs/<TICKER>_mapping_ledger.xlsx.

## 0.7.1 (2026-09-26)
- Only analyst-decided ledger rows feed back into mapping. Rule-recorded rows were being fed back as overrides, which let the ledger's row order reorder rule priority (MARA's FY2025 revenue flipped from the 907m Revenues tag to the 59m contract-revenue tag) and relabeled rule matches as analyst decisions.
- Excel export: validation lists moved to a hidden sheet (Excel strips inline lists over 255 characters, which the repair dialog reported).
- map-suggest: all candidates are proposed in batches of 30 (the default cap silently dropped 45 of 85 for MARA); candidates duplicating an already-mapped value are skipped; a caption and a tag with the same value count once.

## 0.7.0 (2026-09-26)
- Mapping rules moved out of Python into crucible/rules/schema_rules.csv (CRUCIBLE_SCHEMA_RULES points at a custom copy).
- Candidate matching with confidence and alternatives; mapping ledger (data/mapping_ledger.csv) records every link per company with evidence and decisions; accepted rows override rules, rejected rows are excluded; Excel export/import for analysts.
- `map-suggest`: LLM proposals for the unmapped residue above materiality, statement-consistent targets only, written as pending; `map-approve`, `map-export`, `map-import`, `map-status`.
- Ingest saves full-filing XBRL facts per 10-K (custom concepts) as xbrl_facts_<FY>.parquet; the mapper links them through the ledger.
- Attribution by regression: random-subset design with OLS and bootstrap intervals (`harness --ablation random`), and `attribution-fit` on logged runs (OLS on base, logistic on backtest hits, OLS on realized minus base); backtest runs now log the realized outcome.
- docs/ROADMAP.md: every translation from filings to model broken into steps with what done looks like.

## 0.6.4 (2026-09-25)
- Ingest reports releases within the window vs on disk, and the number of 10-K text years, so a shorter --years run no longer looks like data loss.

## 0.6.3 (2026-09-25)
- Shell fiscal years (no revenue or no total assets: pre-merger SPAC years, deprecated concepts) are dropped before the consecutive-run trim; the 10-year ingest had introduced FY2017 for Vertiv and FY2013-14 for Eaton with zero revenue.
- Text evidence excludes safe-harbor boilerplate, keeps near-duplicate sentences once (bullet and body), and skips the competition excerpt when Item 1 is present.
- Judge prompt (v0.5): the range must be at least twice the company's own long-run growth volatility unless argued otherwise.
- Harness reports judge lean (position of the base between the final bear and bull, 0 to 1); 8-K ingestion keeps 4 x years + 8 releases; "Long Term Debt" label matched.

## 0.6.2 (2026-09-25)
- Growth packets carry a long-run history item built from every fiscal year on disk as of the date (growth by year, CAGR, volatility, down years), on top of the 5-year modeling window; messages now print both windows.
- Backtest sweeps write outputs/<T>_<assumption>_calibration.json (realized minus base across the as-of dates); debate and harness review sheets show that empirical adjustment beside the advisory range.

## 0.6.1 (2026-09-25)
- `ingest --peers` saves the peer set to data/<TICKER>/peers.json; packet, debate, harness, backtest and inputs default to it, so a forgotten --peers no longer silently produces a peerless packet.

## 0.6.0 (2026-09-25)
- `crucible inputs`: derives risk-free (FRED DGS10), beta and peer betas (2-year weekly OLS vs ^GSPC via Yahoo Finance, with r2 and Blume adjustment), ERP (Damodaran implied, best effort, else analyst-set with a source), cost of debt, effective tax rate and market-value debt weight (dei cover-page shares x price) with citations, into data/<TICKER>/analyst_inputs.json; reused within 7 days; overrides with reasons; notes.
- Market data caches under data/market/ (FRED series, weekly prices) with date-aware refresh; all fetchers injectable and tested offline.
- WACC packet items carry the citation of each input; packet and model read the derived file automatically; the legacy flat template is marked PLACEHOLDER.

## 0.5.0 (2026-09-25)
- Control arm: `estimate()` single-analyst estimates (validated claims, quotes); `vote()` runs N of them concurrently with caching; harness prints VOTE and DEBATE vs VOTE (base delta vs pooled noise floor, range width, evidence cited); review.md carries the comparison.
- Backtest: `--sweep N` runs the N most recent scoreable 10-K dates, scores debate and vote arms, partial horizons (1 or 2 years realized) included, trailing growth volatility and a volatility-floored hit rate reported; calibration summary and CSV. `dates` command lists filings on disk and scoreable as-of dates.
- Guidance: column-aware guidance table parsing (quarter/full-year side by side), stated growth sentences ranked first, and a guidance track record item (initial guide vs actual for each realized year: for Vertiv, beat in 4 of 4 years, mean +6.2pp).
- History with a gap year (facts missing a fiscal year as of an old date) is trimmed to the consecutive run; peer growth evidence uses consecutive years only (fixes the KeyError at --as-of 2022-03-01).

## 0.4.1 (2026-09-25)
- Packets are rebuilt from data on every debate/harness/backtest run (a stale outputs/ packet masked v0.4's evidence changes).
- Guidance extraction is paragraph and bullet aware, weights sentences with a percentage and a guide verb, parses guidance table rows ("Full Year 2023 Guidance Net sales $6,450M - $6,600M Organic net sales growth 14% - 17%") and stated growth guides ("Expect 2023 net sales growth of 15%"). The February 2023 release now yields three direct items where v0.4 yielded none.
- Validator ignores year indices ("year 2", "Y1") and small bare counts ("3 years", "2 peers").
- Attribution uses a pooled noise floor (stability and permutation runs) and flags systematic shift when every removal moves the base the same way.

## 0.4.0 (2026-09-24)
- Verified-quote gate: every claim carries a verbatim span that must string-match a cited item; otherwise it is dropped. Numbers must exist in the packet; citations that omit the item holding a number are completed rather than trusted.
- Heterogeneity: role-specific evidence briefs (bull: forward indicators, trajectory, capacity; bear: risks, cost lines, peers), per-role model overrides (`--bull-model`, `--bear-model`, `--judge-model` as provider:model), advocate confidence.
- Crux-first synthesis: crux list (question, both positions, evidence, what settles it), questions for management and for internal discussion, implied path, advisory range; `outputs/<T>_<assumption>_review.md` review sheet; the analyst decides via `decide`.
- Harness: concurrent debates (`--workers`), progress lines, on-disk cache keyed by inputs (resume after interruption), call statistics per debate.
- JSON parsing repairs unescaped inner quotes (the cause of a bear round losing all but one claim); `CRUCIBLE_DEBUG=1` dumps raw responses.
- Guidance derivation uses the latest full-year guide (ranges or point guides), skips boilerplate sentences; cost-line evidence added; 8-K history extended to seven years.

## 0.3.1 (2026-09-24)
- 8-K ingestion pre-filters on the SEC index's item list (only Item 2.02 filings are fetched), retries 503/429 with backoff, and `ingest --only-eightk` adds releases without re-pulling everything else.

## 0.3.0 (2026-09-24)
- 8-K earnings releases (Item 2.02, EX-99.1) ingested per filing date; packets draw guidance, orders, backlog and book-to-bill sentences from the two most recent releases and derive the implied growth from a full-year net sales guide.
- Judge and advocates get an explicit evidence hierarchy (company forward indicators > company trajectory > peers > boilerplate), an arithmetic-discipline step (state what the base implies for the remaining years) and confidence calibration guidance.
- `backtest` command: as-of packet (numbers, 10-K text and 8-Ks all dated), N debates, hit rate and base error against the realized 3-year CAGR; outcomes logged.
- As-of history no longer back-fills from latest-known statements (no look-ahead leakage).
- Validator accepts differences of cited numbers and ignores "10-Ks"; harness groups 8-K text as its own ablation unit; analyst example notes emptied (placeholders were moving verdicts by 3pp).

## 0.2.4 (2026-09-23)
- Validator: ranges (12-18 months), form names (10-K) and 3yr-style tokens no longer parse as numbers (a false drop killed the bull's backlog claim in the first live debate).
- Harness: ablation by evidence group (target history, derived metrics, each peer, each filing section, analyst notes) is the default; prints the call budget before running.
- Ingest: saves 10-K text sections per fiscal year so `--as-of` packets use the text known at the time.

## 0.2.3 (2026-09-23)
- String-aware JSON extraction with truncation repair; max_tokens 6000 with one retry on truncation; raw responses saved to logs/raw/ on parse failure; one automatic retry per call on malformed JSON; advocate output bounded (6 claims, 4 rebuttals, 3 questions).

## 0.2.2 (2026-09-22)
- Anthropic SDK 1.x compatibility: sampling parameters are sent via extra_body (the SDK removed the temperature keyword in 1.0); automatic retry without them if a model rejects them.

## 0.2.1 (2026-09-22)
- Clear errors when ANTHROPIC_API_KEY is missing; anthropic model defaults to claude-sonnet-5; `--model` flag on packet/debate/harness.

## 0.2.0 (2026-09-22)
- Mapping rewritten against real edgartools output: concept-level facts are the primary source (exact us-gaap
  concepts, per-year coalescing, sum-of-components for debt and D&A, sign conventions), standardized statement
  labels fill gaps, "Additional" rows rank below main rows.
- Per-cell fallbacks: gross profit, pretax income and operating income are reconstructed only in the years they
  are missing (Eaton reports no operating income line; it is rebuilt from pretax + interest - other non-operating).
- Reconciliation-driven R&D handling: a negative other-opex residual flags R&D embedded in SG&A/COGS.
- History in USD millions, last 5 complete fiscal years, `--as-of` for point-in-time rebuilds.
- Mapping report now lists the source of every line and the largest unmapped facts absorbed by each residual.
- Vertiv and the four peers map with residuals in the low single digits of revenue; miners are flagged (custom concepts).

## 0.1.0 (2026-09-22)
- First runnable loop: synthetic data, deterministic model + Excel parity, debate, harness, decision log.
