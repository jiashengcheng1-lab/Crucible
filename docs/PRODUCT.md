# Product framing (working draft for the D. E. Shaw case)

## Problem
Fundamental analysts already own models they trust. Two things eat their time and judgment:
1. Maintenance: rolling actuals in when a 10-Q drops, reconciling restatements, keeping peers comparable.
2. Assumptions: the forecast rates and discount rate that drive the valuation are argued in meetings from
   scattered evidence, and it is hard to know which evidence actually moved the number.

## MVP loop (one company, three peers, two assumptions)
filings -> standardized model with provenance -> evidence packet -> bull/bear/judge -> range + questions for management
-> stability, order permutation, leave-one-out attribution -> analyst accept/edit/reject logged -> backtest later.

## Three design rules the debate follows
1. Verified quotes as a hard gate. Debate helps a non-expert judge when debaters must ground claims in verified quotes
   from the source (Khan et al., ICML 2024; the original proposal is Irving, Christiano and Amodei 2018; CRITIC, Gou et al.
   2023, is the tool-verification analogue). Every claim carries a span that must string-match a cited evidence item.
   Unverified claims never reach the analyst.
2. Heterogeneity. The bull starts from forward indicators, growth history and capacity evidence; the bear from risk
   factors, cost lines and peer decelerations. Each side can run on a different model family. Both state a confidence.
3. Output the crux and the questions, not a verdict. The synthesis pass maps the disagreement path and what would
   settle it, and produces questions for internal discussion and for management. The range is advisory. The analyst
   is the judge, and the accept/edit/reject log is the dataset that later trains evidence weighting.

## Why adversarial, and why that is not enough
Debate surfaces the strongest case on each side, which is what a management meeting needs. But debate alone is
not robustness: multi-agent debate does not reliably beat self-consistency and is sensitive to agreement
intensity (Smit et al., ICML 2024). So the product measures robustness instead of assuming it:
- seed stability: does the base case move when nothing changes?
- order permutation: does evidence position bias the verdict?
- leave-one-out attribution: which evidence items move the answer, which are noise?
- outcome backtest (next): build as of an old 10-K, compare the range to what happened.

## Why run a control
Recent evidence says debate is not free: Choi et al. (NeurIPS 2025) find that majority voting explains most of the gains
attributed to multi-agent debate; Zhu et al. (ACL 2026 Findings) show that homogeneous agents with uniform belief
updates cannot reliably beat their own starting distribution, and that diverse starting answers and explicit confidence
sharing are what help; Du et al.'s symmetric "society" debate fails mostly through mutual reinforcement of wrong answers.
Debate earns its tokens only with heterogeneity, a separate judge and grounding. Crucible has all three, and it runs the
control anyway: N independent estimates on the same evidence, and the harness states whether the debate moved the
assumption, tightened the range or surfaced evidence beyond the vote. Backtests score both arms.

## Metrics
- Model: mapping coverage, residual size vs reported totals, balance/cash checks, time-to-refresh on a new filing.
- Assumptions: agreement rate across seeds, range width, share of evidence that is noise, analyst edit distance
  from the tool's base, share of analyst edits inside the tool's range, realized value inside the range (backtest).
- Adoption: decisions logged per analyst per week; questions-for-management actually used.

## Data loop
decisions.jsonl (accept/edit/reject with reason) + outcomes.jsonl (realized) are the assets that compound.
They train the evidence weighting and calibrate the ranges. Nothing else in the stack is proprietary.

## Compliance
Public SEC data. Every number carries a filing accession. Attribution doubles as an audit trail of which
information entered an estimate (information barriers, data licensing).

## Roadmap after the MVP
1. Model maintenance mode: diff a new 10-Q against the model, flag breaks, propose the roll-forward.
2. More assumptions: margins, capex, terminal growth; sector driver templates (e.g. hashrate x hashprice for miners).
3. Analyst surface as a Claude Code skill over the edgartools MCP server; review page for decisions.
4. Backtest harness on point-in-time facts (`facts_pit.parquet` is already pulled by `ingest`).
