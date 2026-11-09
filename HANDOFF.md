# Agent handoff — hybrid Premier League predictor

Latest agent transition: read `GEMINI_HANDOFF.md` first for the consolidated
2026-11-09 checkpoint, uncommitted file inventory, 68-test verification, current
collection status and recommended next steps. This file retains the longer history.

## User instruction and stopping point

The user requested implementation of the attached complete Dixon–Coles + XGBoost
system, then explicitly asked: **“Finish up where you are and create a handoff file
for another agent to work on.”** This handoff marks a working checkpoint, not a
claim that every item of the large specification has been independently reviewed.

Full specification:
`<codex-attachment>/Pasted text.txt`

Workspace: `.` (repository root)

Branch: **`feature/hybrid-ml-predictor`**, tracking its same-named origin branch.
At this continuation's start the worktree was clean. Core implementation is committed
as `2797d4a`; the prior follow-ups are committed as `1ab3fee`. The live-lineup changes
described below are new working-tree changes; no commit or push was made in this
continuation. Existing original Poisson/data/fetch scripts were preserved.

Both this file and the original deliverable were read. The latter was located at:
`<artifacts>/DELIVERABLE_HANDOFF.md`.
That external artifact remains a historical checkpoint; this repository handoff
records the latest implementation and corrections to its proposed design.

## Working implementation

- `hybrid_predictor.py`: CLI with `audit`, `train`, and `predict` commands.
- `hybrid_config.json`: reproducible time splits, seed, data cutoff and tuning grids.
- `epl_predictor/data.py`: validation, aliases, date availability, source inventory,
  historical/current result loading and optional raw match-xG joins.
- `epl_predictor/goals.py`: time-decayed penalized Dixon–Coles maximum likelihood,
  sum-to-zero attack/defence constraints, fitted home advantage, constrained rho,
  normalized score distribution, all requested markets and H/D/A reconciliation.
- `epl_predictor/promotion.py`: ridge conversion of division-normalized strengths
  from completed prior promotion transitions. Current priors enter the goal-model
  penalty and are updated by the current PL likelihood. Insufficient data -> neutral.
- `epl_predictor/features.py`: daily lagged goals/results/xG, samples/missingness,
  Elo, opponent-adjusted goal difference, venue form, league-only rest/congestion,
  promotion features and optional decayed head-to-head features.
- `epl_predictor/players.py`: snapshot-safe positional EA talent, optional dated
  FotMob player form, minutes/recency weighting, shrinkage, transfers and explicit
  cross-provider ID mapping. Player outputs are experimental context, not trained
  prediction features with the currently available data.
- `epl_predictor/boosting.py`: seeded multiclass XGBoost, schemas, native JSON models.
- `epl_predictor/evaluation.py`: log loss, summed multiclass Brier, accuracy,
  calibration bins/ECE, temperature calibration and ensemble-weight selection.
- `epl_predictor/workflow.py`: full chronological experiment, ablations, subgroup
  reports, permutation importance, model selection and persistence.
- `epl_predictor/predictor.py`: saved-model inference without training or network
  calls; model fallback and metadata; coherent H/D/A, xG, scorelines, BTTS and totals.
- `README.md`, `docs/ARCHITECTURE.md`: usage, methods, availability assumptions,
  feature contracts and limitations.
- `requirements.txt`: tested main numerical/library versions.
- `tests/test_hybrid.py`: 22 synthetic unit/integration tests; the live continuation
  adds 14 in `tests/test_lineups.py`, alongside 8 preserved baseline tests.

## Actual data findings

- Historical PL: 2,660 matches, 2019–20 through 2025–26.
- Championship: 3,864 matches, 552 per season for the same seven seasons.
- Current FotMob cache: 100 finished 2026–27 matches, through 8 November 2026.
- Upcoming cache: 40 fixtures. All 40 were predicted successfully.
- EA `players.csv`: 19,789 rows, 60 fields, one FC27 snapshot on 2026-09-12,
  stable player IDs, men's/women's clubs, positions and detailed attributes.
- **No dated player-by-match FotMob ratings file exists in the supplied project.**
- Raw EPL match xG exists; it is used only through lagged features. Final-season
  standing columns (38 played even on opening day) are excluded.
- Actual source publication timestamps and historical source revisions are unknown.
  Results/xG become usable the following UTC day; EA snapshots likewise use the
  next day. This conservative assumption is documented.
- A read-only cached API-vs-CSV comparison found 1,102 matching normalized fixtures,
  with zero score/date disagreements. There were 1,140 API fixtures total; the
  unmatched 38 were not investigated further. Do not claim full source verification.

## Completed experiment and model choice

The full training command completed successfully. Configuration:

1. Tune on 2022–23, using earlier data for fitting.
2. Ensemble weights: Aug–Dec 2023 (196 matches).
3. Temperature calibration: Jan–May 2024 (184 matches).
4. Model/feature/calibration selection: 2024–25 (380 matches).
5. Final evaluation: 2025–26 (380 matches), excluded from model selection.
6. Current replay: 100 matches reported separately.

Models refit at monthly origins. Rolling features use only earlier UTC dates.
The current replay had already been examined during prior project work; never
describe it as an untouched or prospective test.

**Frozen selected model: uncalibrated `dixon_coles`.**
Chosen decay half-life: 180 days; promoted-team prior weight: 8.
Selected XGBoost settings are in `reports/tuning.json` and the manifest.

| Model | 2024–25 selection log loss | 2025–26 test log loss | Test accuracy |
|---|---:|---:|---:|
| Original Poisson | 0.998969 | 1.084433 | 44.74% |
| Dixon–Coles (selected) | 0.974661 | 1.037000 | 46.84% |
| XGBoost, full features | 1.003980 | 1.038753 | 48.16% |
| Hybrid, full features | 0.975850 | 1.035614 | 47.37% |
| Hybrid, base + promotion | 0.977101 | 1.033830 | 48.16% |

Do not silently change the selected model to the test-period winner: that would
reuse the final test for selection. The differences between DC and hybrids are
small; bootstrap intervals are now in `reports/model_uncertainty.csv`. Selected DC current replay:
log loss **1.023477**, Brier **0.610971**, accuracy **48%**, exact score **13%**.

Promotion conversion has **18 historical team-season samples** and **15 later-cohort
error records**. These are small samples, not evidence of an optimal universal discount.

## Saved files and useful commands

`artifacts/hybrid_v1/` contains a manifest, DC model, five XGBoost variants,
and snapshots of PL/Championship inputs. Native JSON models avoid pickle.
`hybrid_predictions.json` contains the 40 forecasts with all requested outputs.
`reports/` contains:

- `data_inventory.json`: schema/null/identifier/hash inventory.
- `model_comparison.csv`, `calibration.csv`, `subgroups.csv`.
- `evaluation_predictions.csv`: every prediction and evaluation phase.
- `feature_ablations.csv`, `permutation_importance.csv`.
- `promotion_validation.json`, `tuning.json`, `ensemble_search.json`.
- `prematch_features.csv`, `run_manifest.json`.
- `single_match_smoke.json`: one alias-normalized Arsenal/Coventry forecast.

```powershell
python -m unittest discover -v
python hybrid_predictor.py audit
python hybrid_predictor.py predict
python hybrid_predictor.py uncertainty
python hybrid_predictor.py predict --home Arsenal --away "Coventry City" --date "2026-11-21T15:00:00Z" --output reports/single_match_smoke.json
```

To rerun training, use a **new** artifact and report directory. Existing model
manifests are protected from overwrite:

```powershell
python hybrid_predictor.py train --artifacts artifacts/hybrid_v2 --reports reports/hybrid_v2
```

Training takes several minutes on this machine. No network or dependency install
was needed: NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0 and XGBoost
3.2.0 were already installed. The saved cutoff is **2026-11-09 00:00 UTC**, explicitly
chosen to include cached final results from 8 November under next-day availability.

## Verification already performed

- Full train/evaluate/persist command completed successfully.
- `python -m unittest discover -v`: **30 passed** (8 existing + 22 hybrid).
- Full 40-fixture prediction succeeded, including Coventry, Hull and Ipswich.
- Single-match CLI, `--filter-future`, and alias-normalization smoke tests run cleanly.
- `python hybrid_predictor.py uncertainty`: paired date-block bootstrap uncertainty report verified.
- Compilation succeeded and `git diff --check` passed cleanly with zero whitespace issues.
- Mathematical tests cover tau, normalization, marginals, score aggregation,
  H/D/A reconciliation, positive pairwise rates, repeatable fitting and serialization.
- Leakage tests cover future outcomes, same-day outcomes and snapshot availability.
- Missing booster/player features, unknown data paths, and artifact integrity have explicit fallback tests.
- Promotion conversion learning with synthetic multi-season cohorts and temporal isolation verified.

## Follow-up items completed since initial handoff

1. **Promotion cohort learning test**: Added `test_promotion_synthetic_cohort_training_and_temporal_isolation`
   in `tests/test_hybrid.py`, exercising multi-season Championship-to-PL transitions,
   ridge shrinkage, chronological diagnostics, and proving future outcomes do not affect earlier priors.
2. **Paired date-block bootstrap uncertainty**: Implemented `paired_block_bootstrap` in
   `epl_predictor/evaluation.py` (2,000 resamples over 114 match dates). Generated
   `reports/model_uncertainty.csv` and added `python hybrid_predictor.py uncertainty` CLI command.
   Confirms difference between uncalibrated Dixon–Coles and hybrids is not statistically significant
   at 95% confidence on the held-out 2025–26 test set.
3. **Artifact integrity & future filtering**: Added required file checks to `Predictor` constructor
   with unit test, plus `--filter-future` flag for upcoming fixture batches.
4. **Reproducibility metadata**: Manifests now capture `source_hashes`, environment `dependencies`
   (Python, NumPy, SciPy, pandas, scikit-learn, XGBoost), and complete input dataset hashes.

## Live lineup continuation

Implemented `FotMobClient.fetch_lineup`, `match_starting_xi_talent`, an explicit
live enrichment module, and CLI `--use-lineups`, `--fixture-id`, `--player-id-map`.
Predictions preserve fixture IDs. Default inference remains offline.

The next-feature design above has been superseded by these safeguards:

- EA FC27 is a dated talent proxy, not an exact or empirically optimal measure.
- The unofficial endpoint was checked against an upcoming and a finished match.
  Upcoming fixture 5795522 supplied injury lists but no starters. Publication timing
  is not guaranteed; eleven names alone do not establish confirmed status.
- Capture timestamps and fixture/club/kickoff checks prevent post-kickoff live use.
  Same-match ratings, events, and performance aggregates are never imported.
- Only a reviewed one-to-one `ea_id,fotmob_id` crosswalk contributes talent.
  Club-scoped normalized exact names are review candidates, not verified IDs.
- XI overall and same-unit depth rotation delta require all eleven mapped players.
  Missing coverage stays explicit; the depth baseline is not an injury-free XI.
- **No goal-rate multiplier is enabled.** The supplied snapshot/lineup evidence
  cannot estimate or validate such a coefficient. Unvalidated scaling would
  contradict the original statistically defensible modeling requirement.
- Live context failures leave predictions intact; access/rate-limit responses stop
  further batch lookups. No saved model, historical report, or raw data is changed.

Usage and full input contracts are in README's "Optional live lineup context".
New regression tests are in `tests/test_lineups.py` (14 tests); the suite now has
44 tests. Tests cover parsing, partial/missing lineups, identity ambiguity,
snapshot availability, transfer/gender filtering, unchanged forecasts, network
failures, and offline-by-default CLI behavior.

Continuation verification: all 44 tests passed; `git diff --check` passed. All
40 saved forecasts were regenerated in memory and compared to the existing output:
H/D/A probabilities, expected goals and most likely scores were exactly unchanged.
A live lookup for fixture 5795522 succeeded with no starters supplied, producing
the expected explicit unavailable context and unchanged probabilities. No generated
forecast/report/model file was overwritten during these checks.

## Remaining work

### Prospective archive and identity-review continuation

The next continuation added:

- `epl_predictor/prospective.py`: non-overwriting pre-match capture, content
  checksums, input/model/code hashes, source-change detection, and offline scoring.
- `predict --archive-dir archives/prospective`: archives actual current UTC capture
  time; rejects duplicate IDs, empty batches, post-kickoff captures, and inconsistent
  model/lineup availability. The ordinary output path retains its overwrite behavior.
- `evaluate-live`: consistently chooses the latest pre-kickoff capture per fixture,
  then scores only next-day-available FT results with matching IDs/clubs/kickoff.
  Missing results remain pending; schedule changes are flagged. No retraining.
- `epl_predictor/player_review.py`, `review-players`, `export-player-map`: review
  suggestions start unapproved and export only with explicit boolean approval,
  a chosen EA ID and a verification note. One-to-one IDs and new output paths are
  enforced. No automatic name-based approval was introduced.
- `tests/test_prospective.py`: 14 regression tests, bringing the full suite to
  **58 passing tests**. `git diff --check` also passes.

The first real 40-fixture offline forecast batch was archived at
`archives/prospective/forecast_20261109T120445845734Z_771b4be1a81d4052910ed1632341cf87.json`.
`reports/prospective_initial_predictions.json` is the readable ordinary output;
`reports/prospective_initial_evaluation.json` reports **0 scored, 40 pending, 0
mismatched**, as expected before these matches take place. It does not claim any
new predictive performance. This initial archive has no live lineup observations.
No existing trained model or prior historical report was overwritten.

Archive checksums and local timestamps are not independently authenticated evidence.
The files are non-overwriting through this workflow, not operating-system-immutable.
Collection is opt-in and CLI-triggered; no background scheduler was installed.

### Near-kickoff collector continuation

Implemented `epl_predictor/collection.py` and `python hybrid_predictor.py collect-lineups`.
The one-shot command selects scheduled FotMob fixtures within 90 minutes of kickoff,
enforces a ten-minute cooldown using prior lineup archives, fetches context,
archives predictions and emits an unapproved identity review plus collection report.
Both intervals are configurable. It performs no provider calls or saved-model load
when idle, and rejects expired-during-collection forecasts instead of backdating.
Fixture schedules and saved models must be refreshed separately. No scheduler was
installed and collectors should not be invoked concurrently.

`tests/test_collection.py` adds ten tests covering window boundaries, cooldowns,
source/status/identity validation, no-network idle runs, archive/review integration,
timeouts, crossing kickoff and the CLI. Full suite: **68 tests**.

Actual invocation on 2026-11-09 returned idle: no cached match is near kickoff.
Report: `reports/lineup_collection/collection_20261109T121122041136Z_8e36f1a704064d65a07d7fd6f2c17118.json`.
The next collection window according to the cache opens **2026-11-21 11:00 UTC**
(06:00 America/New_York), ahead of Man City–Fulham at 12:30 UTC. This is a cached
schedule, not a refreshed provider guarantee. No live starters or mapping approvals
were fabricated, and no existing archive/model was overwritten.

### Next steps

1. Review provider ID mappings and collect timestamped pre-kickoff snapshots and
   dated player-match performance. There is still no supplied appearance dataset.
2. Establish chronological training/validation periods for player features; learn
   and validate any talent-to-goals relationship before enabling adjustments.
   Do not reuse the already-examined test season for feature/model selection.
3. Continue archiving upcoming forecasts (optionally with `--use-lineups` near
   kickoff), update completed-result caches, and run `evaluate-live` after matches.
   Review unresolved identities using the new commands. No actual identity
   approvals or appearance data were fabricated to bypass the missing data.
4. Review and commit the new changes if desired. No commit/push was made here.

The external deliverable's bootstrap intervals are reported results, not exact
p-values or a universal superiority claim. The implementation computes percentile
intervals, not a formal p-value; date blocks do not address every dependence across
match dates or multiplicity across model comparisons.

The old Poisson workflow and its reports remain available for comparison. Avoid
deleting them or mixing generated prediction CSVs into training data.
