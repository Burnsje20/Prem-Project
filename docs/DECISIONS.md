# Engineering decisions and iteration history

This log explains how the project evolved and why choices were made. It separates
implemented behavior, evidence in saved reports, and exploratory findings. It is
not a claim that every early design intention is enforced by every later extension.
Reviewed against the repository on November 9, 2026.

## 1. Start with a measurable baseline

**Decision:** retain the original independent Poisson predictor and backtest when
adding advanced models. This baseline is not a correlated bivariate-Poisson model.

**Why:** complexity needs a consistent comparison point. Keeping the original
workflow makes regressions and model tradeoffs easier to inspect.

Evidence: [poisson_distribution.py](../poisson_distribution.py),
[backtest_poisson.py](../backtest_poisson.py), and saved model comparisons.

## 2. Adapt ingestion to API constraints

**Decision:** cap the API-Football script at three seasons and use supplied CSVs
for 2019–20 through 2025–26. Keep 2026–27 as the target/live season; completed
current-season results can inform later fixtures, never themselves.

**Why:** initial empty seasons reflected the user's free-plan access limit, not
missing matches. The selected historical range emphasizes the VAR era. It spans
seven seasons, despite an early request describing it as the “last five years.”

**Tradeoff:** original publication/revision histories are not reconstructed. Final
season standings and same-match statistics must not become pre-match inputs.

## 3. Learn promotion priors rather than hand-pick discounts

**Decision:** estimate division-relative Championship attack/defence, then fit a
ridge conversion from completed earlier promotion cohorts. Use those strengths
as regularization priors that Premier League evidence subsequently updates.

**Why:** directly transferring Championship averages ignores competition strength;
a universal discount ignores differences between promoted clubs.

**Tradeoff:** the saved experiment has only 18 transition samples and 15 later-cohort
error records. Sparse evidence needs shrinkage/neutral fallbacks, not claims of an
optimal universal conversion. See [promotion_validation.json](../reports/promotion_validation.json).

## 4. Separate tuning, blending, calibration, selection, and testing

**Decision:** compare time-decayed Dixon–Coles, multiclass XGBoost, probability
blends, and optional temperature calibration. Do not force a hybrid into deployment.

| Phase | Period | Purpose |
| --- | --- | --- |
| Warm-up / expanding fits | 2019–20 onward | Build lagged history; fit on earlier observations |
| Tuning | 2022–23 | Select decay/prior strength and booster settings |
| Ensemble | Aug–Dec 2023 | Fit out-of-sample blend weights |
| Calibration | Jan–May 2024 | Fit optional temperature scaling |
| Selection | 2024–25 | Choose model/feature/calibration configuration |
| Test | 2025–26 | Evaluate the frozen choice, 380 fixtures |
| Current replay | Cached 2026–27 results | Supplementary, already-examined evidence |

Fits use monthly origins; rolling features use earlier UTC dates. Earlier test
results may enter later expanding-window fits, as they would in deployment.

**Outcome:** uncalibrated Dixon–Coles won selection. Its test log loss is 1.037000
versus 1.084433 for original Poisson. Hybrid full reached 1.035614 on test, but
reselecting it on that result would reuse the final test for selection. Paired
date-block bootstrap intervals quantify uncertainty; they are not formal p-values
or a guarantee against every form of dependence across dates.

Evidence: [model_comparison.csv](../reports/model_comparison.csv),
[model_uncertainty.csv](../reports/model_uncertainty.csv),
[hybrid_config.json](../hybrid_config.json).

## 5. Keep probabilities mathematically coherent

**Decision:** constrain the low-score Dixon–Coles correction, maintain positive
rates and identified team effects, normalize score matrices, and derive markets
from a shared distribution. Reconcile H/D/A regions when blending outcome models.

**Why:** independently changing win percentages and scoreline probabilities can
produce contradictory outputs. Tests assert mathematical properties rather than
just checking that prediction functions return values.

## 6. Separate talent, form, and identity confidence

**Decision:** keep EA positional talent and observed FotMob form distinct; reject
an arbitrary 60/40 blend. Resolve latest eligible snapshots before club filtering
to handle transfers. Never backfill FC27 ratings into pre-release games.

**Identity:** club-scoped normalized names create candidates, not verified identities.
Only reviewed one-to-one mappings contribute to EA XI talent. Export requires
approval and a verification note. Full XI summaries need adequate mapped coverage;
the depth baseline is not a known injury-free starting lineup.

**Scope:** exclude tactical matchups without reliable inputs. Label league-only
rest as such; do not invent missing cup schedules or injury history.

## 7. Build live collection outside offline inference

**Decision:** use an explicit provider-client/enrichment boundary. Standard inference
stays offline. Collection filters upcoming fixtures, uses a configurable 90-minute
window and ten-minute cooldown, and now has a process lock. Missing starters,
schedule changes and provider failures are visible states.

**Why:** inference should not require a website request to succeed. Injury lists
are not lineups, and eleven names do not prove confirmation. FotMob's endpoint is
unofficial and can change.

**Iteration:** initial integration was context-only. Appearance ingestion and
optional form adjustment came later. Earlier handoffs saying there was no appearance
dataset describe historical checkpoints, not the current repository.

## 8. Preserve evidence before measuring future performance

**Decision:** create unique archives at actual capture time; hash inputs/models/code;
reject post-kickoff captures and changed inputs. Later evaluation chooses the latest
valid pre-kickoff forecast once per fixture, regardless of outcome. Missing results
remain pending; exact ID/club/kickoff checks expose stale schedules.

**Why:** running a model after observing outcomes is not prospective evaluation.

**Tradeoff:** local timestamps/checksums are not authenticated proof against deliberate
rewriting. Preserve archives as evidence, not training inputs. The initial report
has 40 pending forecasts and cannot establish live predictive performance.

## 9. Investigate player signal before expanding identity engineering

The later brief describes an “Option B before Option A” experiment: investigate EA
squad/rotation and FotMob form signal before expanding a comprehensive ID bridge.
It reports weak EA value, a harmful rotation delta and a small form improvement
across 100 current-season games.

**Evidence boundary:** no corresponding reproducible fold-generation/fitting script
or prediction-level experiment report was found in the reviewed tree. Do not use
those narrative values as verified performance metrics, assume the folds were
chronological, or equate a small gain with established generalization. Preserve
the finding as exploratory work to reproduce.

**Implemented:** each starter's minutes-weighted last five available rated appearances
are averaged across starters. Home/away rates scale by exp(+/- beta × form difference).
Beta defaults to 0.1675; goal rates are clipped to 0.05–10.0. Missing history uses
6.80 and is flagged. Selected original probabilities/goals/score are retained in
baseline_forecast—not a full immutable snapshot of every original market.

**Next validation:** persist chronological folds and out-of-sample predictions;
justify defaults/clipping; require adequate two-sided starter coverage; align form
availability with capture, not merely kickoff. Retrieval time does not prove
historical publication time. Plain CLI prediction/collection is opt-in, but the
unified update pipeline explicitly enables adjustment.

## 10. Prefer static delivery with explicit freshness boundaries

**Decision:** generate HTML/JavaScript for GitHub Pages instead of adding an
always-on application backend. Configure update and deployment workflows separately.

**Tradeoff:** Tailwind loads from a CDN, so the page is not zero-dependency or fully
offline. Its simulation is not live validation. The builder currently uses saved-
model predictions rather than the latest adjusted archive. Workflow definitions
do not prove successful execution, deployment or freshness.

The pipeline refreshes fixtures, collects appearances from the existing results
cache, collects lineups, and builds HTML. It does not refresh completed results or
refit parameters. Those remain follow-ups, along with verifying the Python 3.11 CI
environment against pinned dependencies and testing update-to-Pages delivery.

## 11. Make public documentation auditable

**Decision:** publish actual report values and test outcomes; normalize repository-
local provenance keys; ignore credential/environment files. Replace the agent-specific
README placeholder with engineering boundaries, results and this iteration log.

**Limits:** ignore rules do not prove secrets never entered history; regex scans do
not establish complete security. External paths can still appear in metadata.
No blanket “secret-free,” “fully safe,” “leak-free,” or uptime claim is made.
Documentation review is not a new security audit or validation of every extension.
