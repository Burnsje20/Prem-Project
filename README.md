# Premier League Match Predictor

The project now includes a chronological Dixon–Coles/XGBoost evaluation and
prediction workflow alongside the original Poisson scripts. The deployed model is
chosen by held-out log loss; a hybrid is one candidate rather than a requirement.
The terminal is the prediction interface; this repository had no existing web UI.

## Run the saved predictor

Use Python 3.14 (the tested runtime) and install the recorded dependencies:

```powershell
python -m pip install -r requirements.txt
python hybrid_predictor.py predict
```

This reads the saved model and cached upcoming fixtures and writes
`hybrid_predictions.json`. It does not train or make network requests. Each forecast
contains H/D/A probabilities, expected goals, top five scores, BTTS, over/under 2.5,
the model/cutoff, promoted-team prior flags, player availability and evaluation metrics.
JSON probabilities are fractions; terminal H/D/A values are percentages. Pass
`--filter-future` to automatically skip any fixture scheduled before the model cutoff.

Select a particular match:

```powershell
python hybrid_predictor.py predict --home "Arsenal" --away "Coventry City" --date "2026-11-21T15:00:00Z" --output arsenal_coventry.json
```

The date must be on or after the model's data cutoff and in its current season.
Club aliases are normalized. An unknown club gets a clearly labelled goal-model
fallback; there is no invented player rating or unsupported prediction-confidence score.

## Optional live lineup context

```powershell
python hybrid_predictor.py predict --use-lineups --output lineup_predictions.json
python hybrid_predictor.py predict --home "Man City" --away Fulham --date "2026-11-21T12:30:00Z" --fixture-id 5795522 --use-lineups --output city_fulham_lineup.json
```

`--use-lineups` explicitly enables FotMob network requests (one per distinct fixture
ID). Normal prediction stays offline. JSON includes `fixture_id` and, in live mode,
`live_lineup_context` with retrieval time, provider status, identity-only starters,
coverage, and missing-data reasons. Refresh sparingly: access/rate-limit failures
stop further live requests in the batch. Forecasts still complete if the feed fails.
The feed is unofficial and may change; injury lists do not imply starters exist.
Eleven listed players are not automatically labelled confirmed. No guarantee is
made about how many minutes before kickoff lineups appear.

For talent calculations, supply a **reviewed**, one-to-one CSV linking provider IDs:

```text
ea_id,fotmob_id
```

```powershell
python hybrid_predictor.py predict --use-lineups --player-id-map player_id_map.csv --output lineup_predictions.json
```

Without a crosswalk, club-scoped exact-name matches appear as `candidate_ea_ids`
for review only. They are never silently treated as verified identity. Gender,
club, eligible snapshot date and transfers are checked. Output includes positional
unit talent and matched-player coverage; a full XI overall and rotation delta
require all eleven players to be mapped. The baseline is the best available
same-unit squad-depth proxy, **not** a known injury-free starting XI. Missing
attributes, players and units are not invented.

Live snapshots are rejected at/after kickoff and checked against the fixture's
clubs, ID and kickoff. Same-match ratings, events, and season performance fields
are excluded. Snapshot eligibility uses its capture time, separately from the saved
team model's cutoff. Save each pre-match output to a distinct filename if collecting
prospective evidence; later retrieval cannot reconstruct historical availability.
Use the archive workflow below to preserve runs automatically.

**Lineups do not yet adjust expected goals or probabilities.** EA ratings are a
talent proxy, not an exact measurement of ability, and no validated coefficient
converts their difference into goal rates. A dated lineup/identity dataset and
chronological validation are required before enabling that adjustment. The chosen
Dixon–Coles model, its calibration and its test results remain unchanged.

## Archive forecasts and evaluate future results

```powershell
python hybrid_predictor.py predict --archive-dir archives/prospective --output latest_predictions.json
```

Each run creates a uniquely named `forecast_*.json` archive containing predictions,
optional live lineup context, current UTC capture time, input/model/code hashes,
and a content checksum. Add `--use-lineups` when you want live collection, ideally
near kickoff rather than repeatedly fetching distant fixtures. Archiving verifies
that source files did not change during prediction, requires fixture IDs and
timezone-aware dates, and rejects empty batches, duplicate fixtures, future model
cutoffs and at/after-kickoff captures. `--filter-future` still filters against the
**model cutoff**, not the wall clock: supply a genuinely upcoming fixture batch.
Normal prediction output can still be overwritten; archive files are created
exclusively with unique names. These are local records, not independently notarized
evidence; their checksums detect accidental edits, not intentional forgery.

After updating the completed-results cache, evaluate without retraining:

```powershell
python hybrid_predictor.py evaluate-live --archive-dir archives/prospective --results current_premier_league_results.csv --output reports/live_evaluation_01.json
```

The fixed policy scores the **latest captured pre-kickoff forecast per fixture**,
once only. It does not select whichever forecast best predicted the result.
Only completed results available by the next UTC day are eligible. Fixture IDs,
clubs and exact kickoff must agree; schedule/identity mismatches are reported
separately. Missing results stay pending, with null metrics if none can be scored.
Evaluation never changes models. Choose a new report filename on every run;
evaluation and identity-review outputs refuse to overwrite existing files.

## Collect near-kickoff lineups in one command

```powershell
python hybrid_predictor.py collect-lineups
```

This runs **once**, using the cached fixtures. It selects scheduled FotMob matches
starting within the next 90 minutes (strictly before kickoff), predicts them with
the saved model, fetches lineup context, archives the forecasts, and creates both
a collection report and an unapproved player review JSON in
`reports/lineup_collection/`. It makes no provider requests and loads no saved model
when no matches are eligible. The report prints the next window/retry time in UTC.
Refresh the fixture cache separately with `python fetch_upcoming_fixtures.py` when
needed; collection does not silently refresh schedules or retrain the saved model.

```powershell
python hybrid_predictor.py collect-lineups --within-minutes 90 --refresh-minutes 10 --player-id-map player_id_map_01.csv
```

Repeated runs use archived lineup attempts to enforce a ten-minute per-fixture
cooldown, including unsuccessful requests. Offline forecasts do not trigger that
cooldown. Both timings are configurable operational settings, not a promise of
when FotMob publishes confirmed lineups. Unsupported sources, cancelled/finished/
postponed fixtures, unknown statuses, past kickoffs and distant matches are skipped
with reasons. If kickoff passes during a request batch, those forecasts are not
backdated into the archive. Provider access/rate-limit failures stop further
requests within the batch as in normal live prediction.

Review the emitted `player_review_*.json`, then use `export-player-map` below.
When there are no starters, the review contains no players. This command neither
approves identities nor installs a background task; rerun it near kickoff and after
the cooldown if you want another capture. The collector acquires an exclusive process
lockfile (`.collector.lock` in `--output-dir` or via `--lock-path`) with stale-lock clearing
(15-minute timeout) to safely prevent race conditions or corrupted archives if triggered
concurrently. Network calls feature exponential backoff retry adapters (3 attempts on 429/5xx).
Supply `--adjust-with-lineup-form` to automatically apply the cross-validated starter form
adjustment ($\beta = 0.1675$) to archived prospective forecasts whenever starting lineups are retrieved.

## Review player identities

Create a review JSON from one or more lineup prediction files or forecast archives:

```powershell
python hybrid_predictor.py review-players --predictions lineup_predictions.json --output player_review_01.json
```

Repeated unresolved FotMob IDs are consolidated with observed names, clubs,
fixtures and candidate EA IDs. All decisions start with `approved: false` and
`ea_id: null`, including unique name matches. Verify each identity, supply the
chosen `ea_id`, set `approved: true`, and add a `review_note` describing verification.
Do not approve a match just because its name is similar. Then export:

```powershell
python hybrid_predictor.py export-player-map --review player_review_01.json --output player_id_map_01.csv
python hybrid_predictor.py predict --use-lineups --player-id-map player_id_map_01.csv --archive-dir archives/prospective --output latest_predictions.json
```

Only explicitly approved rows export; duplicate/conflicting IDs, missing review
notes and string-valued approvals are rejected. No automatic approval or fuzzy
identity assignment occurs. If the provider has not supplied starters, the review
will be empty; no players are invented. Collection is manual/CLI-triggered, not a
background scheduler. Historical player-performance collection and validation of
player-enhanced models remain separate next steps.

## Reproduce the experiment

```powershell
python hybrid_predictor.py audit
python hybrid_predictor.py train --artifacts artifacts/hybrid_v2 --reports reports/hybrid_v2
python hybrid_predictor.py predict --artifacts artifacts/hybrid_v2
python hybrid_predictor.py uncertainty
python -m unittest discover -v
```

Training uses `hybrid_config.json`. Choose a new artifact directory for each run;
an existing manifest is protected from accidental overwrite. Change `as_of` to
advance the information cutoff after updating the cached sources. Matches only
become available the next UTC day, because final-whistle timestamps are absent.

The saved run uses information available by **2026-11-09 00:00 UTC**, including
the supplied completed results through 8 November. It produces frozen model files,
feature schemas, data hashes, configuration and exact training-input snapshots.
Current-season matches are permitted as prior evidence for later predictions.
They never enter a prediction for themselves or earlier fixtures.

The replay refits models at the beginning of each month. Team rolling features use
only earlier days within that month; fitted model parameters use the monthly origin.
This is an expanding-window evaluation of a documented refresh schedule.

| Stage | Period | Purpose |
|---|---|---|
| Warm-up/training | 2019–20 onward | Lagged features, then model fitting from 2020–21 |
| Tuning | 2022–23 | Goal decay/prior strength and XGBoost settings |
| Ensemble | Aug–Dec 2023 | Probability blend weights |
| Calibration | Jan–May 2024 | Optional temperature scaling |
| Selection | 2024–25 | Model, feature group and calibrated/raw choice |
| Test | 2025–26 | Evaluation after freezing the choice |
| Current replay | 2026–27 cached matches | Separate supplementary evaluation |

The current-season results had already been examined in earlier project work.
Neither that replay nor a historical dataset with unknown revision history is a
prospective live trial. The final test period is excluded from the new model's
selection; earlier test matches may enter subsequent monthly fits, as in deployment.

## Models and features

- Original Poisson: preserved in `poisson_distribution.py` and evaluated separately.
- Dixon–Coles: time-decayed penalized maximum likelihood; identified attack/defence
  parameters, fitted home advantage and a strictly valid low-score correlation.
- Promotion conversion: ridge regression on completed historical Championship-to-PL
  transitions, normalized to each division's scoring environment. Chronological
  transition errors are reported. Sparse conversion data uses a neutral prior.
- XGBoost: seeded multiclass probabilities, conservative configuration search and
  ordered feature schemas, saved using native JSON rather than pickle.
- Hybrid: out-of-sample fitted probability weight and optional temperature scaling.
  H/D/A regions of the goal matrix are reweighted to match the final outcome model.
  Expected goals, scorelines and every displayed market come from the same matrix.

Available features include rolling five/ten-game goals and results, recency weighting,
venue form, Elo, league goal environment, opponent-adjusted goal difference, lagged
xG with missingness/sample counts, league-only rest/congestion, promotion priors and
optional decayed head-to-head form. Rest is explicitly **league-only**: cup and
European fixtures are not supplied. Elo K=20 and the feature recency half-lives are
documented fixed baseline settings, not claimed to be optimized.

Feature experiments compare base, base+rest, base+promotion, base+xG and the full
available group. All use identical evaluation fixtures. Permutation importance is
diagnostic association, not a causal explanation. The full group is not automatically
selected. The promotion penalty strength and goal decay are selected chronologically.

## Player data status and input contracts

`players.csv` contains one FC27 snapshot dated 12 September 2026. The player
pipeline filters by availability, handles men's clubs, resolves aliases, and keeps
GK/DEF/MID/ATT talent separate. Without minutes/starts, it reports a positional
depth proxy (best goalkeeper and top three available players per outfield unit),
not an actual or confirmed starting XI. With explicit EA-ID expected-minutes
weights, aggregation uses those weights instead. No global 60/40 talent/form blend.

Completed match player performances can be collected incrementally from FotMob:

```powershell
python hybrid_predictor.py collect-appearances
```

This collects all completed 2026–27 matches into `fotmob_player_appearances.csv`,
calculating accurate minutes from starter/sub-event intervals and red cards,
capturing ratings, and enforcing the schema:

```text
player_id,match_id,date,available_at,team,position,rating,minutes
```

Dates record kickoff and when the performance became available (next UTC day).
`retrieved_at` records the crawl timestamp to distinguish retrieval time from
match availability. Null ratings remain missing. IDs are provider IDs; joining to EA
requires an explicit one-to-one `ea_id,fotmob_id` bridge. Names alone are never
used as identity. Transfers use the latest available club observation.

```powershell
python hybrid_predictor.py predict --appearances fotmob_player_appearances.csv
python hybrid_predictor.py predict --home Arsenal --away Chelsea --date 2026-11-21T12:30:00Z --fixture-id 123 --use-lineups --adjust-with-lineup-form
```

The module computes five/ten-appearance ratings, minutes/recency weighting, long-term
form, sample sizes, position-based shrinkage and unit form. When `--adjust-with-lineup-form`
is supplied with `--use-lineups`, confirmed starting XI player ratings from prior appearances
are aggregated into team starter form. Goal rates are scaled by the net starter form difference
($\lambda_{adj} = \lambda \cdot e^{\beta \Delta}$, $\mu_{adj} = \mu \cdot e^{-\beta \Delta}$, with cross-validated $\beta = 0.1675$).
The baseline Dixon-Coles forecast is preserved in `baseline_forecast` alongside the adjusted
probabilities. Without `--adjust-with-lineup-form`, lineup context is attached as observational
metadata and forecast probabilities remain strictly unchanged.

## Reports and implementation

- `reports/data_inventory.json`: all CSV schemas, nulls, dates, identifiers and hashes.
- `reports/model_comparison.csv`: outcome/score loss, Brier, accuracy and calibration.
- `reports/model_uncertainty.csv`: paired date-block bootstrap log loss uncertainty vs selected model.
- `reports/evaluation_predictions.csv`: every out-of-sample probability with phase.
- `reports/calibration.csv`: per-class probability bins and observed frequencies.
- `reports/feature_ablations.csv`: comparable feature experiments.
- `reports/subgroups.csv`: promoted clubs, favorites and competitive fixtures.
- `reports/promotion_validation.json`: transition sample records and later-cohort errors.
- `reports/permutation_importance.csv`: selection-period feature diagnostics.
- `reports/run_manifest.json`: selection, config, cutoffs and source hashes.
- `reports/matchday_12_dashboard.html`: interactive Matchday 12 fixture dashboard with live starter form simulation.
- `epl_predictor/`: data, goal model, promotion, features, players, boosting,
  evaluation, workflow and inference modules.
- `tests/test_hybrid.py`: synthetic unit/integration and leakage regression tests.

Source limitations: match/xG publication histories and source licensing are not
independently verified. The workflow reads local supplied data and does not hide
scraping or network calls in training/inference. Existing FotMob fetch scripts remain
optional; access/reuse must comply with the provider's terms. No tactical matchup
features are implemented. Final-season standings, same-match statistics and future
player snapshots are prohibited from pre-match inputs.

Mathematical references: [Dixon & Coles (1997)](https://doi.org/10.1111/1467-9876.00065),
[XGBoost multiclass objective](https://xgboost.readthedocs.io/en/stable/parameter.html),
[native model persistence](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
See [architecture and availability rules](docs/ARCHITECTURE.md) for the experiment plan.
