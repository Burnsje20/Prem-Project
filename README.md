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

There is **no supplied player-by-match FotMob file**. The optional importer expects:

```text
player_id,match_id,date,available_at,team,position,rating,minutes
```

Dates must identify kickoff and when the performance became available. `is_home`
is optional. Null ratings remain missing. IDs are provider IDs; joining to EA
requires an explicit one-to-one `ea_id,fotmob_id` bridge. Names alone are never
used as identity. Transfers use the latest available club observation.

```powershell
python hybrid_predictor.py predict --appearances fotmob_player_matches.csv
```

The module computes five/ten-appearance ratings, minutes/recency weighting, long-term
form, sample sizes, position-based shrinkage and unit form. Shrinkage uses two
equivalent games as an experimental default. Reliable starting-XI histories, injury
histories, positional normalization experiments and player-enhanced model training
remain data-limited. Player summaries are exported as **experimental context** and
do not change the selected prediction. Adding a file alone does not constitute a
validated player-enhanced model; a separate chronological training/validation/test
sample must be established first. Unavailable player ablations are recorded explicitly.

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
