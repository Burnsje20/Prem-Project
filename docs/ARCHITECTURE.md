# Implementation and data availability plan

The original `poisson_distribution.py` and its backtest remain usable. New code
lives in `epl_predictor`, with one CLI for audit, training/evaluation and inference.
Models are saved separately from evaluation reports; inference never fits a model.

## Observed sources

- Premier League: 2,660 completed matches, 2019–20 through 2025–26.
- Championship: 3,864 matches, seven complete 552-match seasons. July 2020 belongs
  to 2019–20, as recorded in the existing processed season field.
- Current FotMob result cache: 100 completed 2026–27 matches.
- Upcoming fixture cache: 40 scheduled fixtures, held separate from results.
- EA `players.csv`: 19,789 records, 60 columns, FC27 snapshot 2026-09-12; stable
  player IDs, positional and detailed attributes, both men's and women's clubs.
- No player-by-match FotMob ratings/minutes/start-status file is present.
- The raw EPL CSV includes match xG and final-season standings. Standings with
  `SeasonPlayedHome=38` even on opening day are prohibited as historical features.
- Existing generated predictions and reports are outputs, never training inputs.

`python hybrid_predictor.py audit` produces `reports/data_inventory.json` with
every CSV's hash, schema, missing counts, duplicates, identifiers and coverage.
Raw API JSON is a cached overlapping fixture source, not an independent sample.

## Availability contracts

Results and xG become usable on the next UTC day: final-whistle times are missing.
Same-day matches never update each other's features. This is a conservative
reconstruction assumption, not evidence of each source's original publication time.
Historical source snapshots are unavailable; corrections/revisions cannot be replayed.
Raw xG is used only through lagged rolling windows; it has no claim of independent
provider verification. Closing odds, match shots and end-of-season summaries are excluded.

EA snapshot dates are treated as availability dates, conservatively usable the
following day. No EA snapshot is backfilled into earlier matches. Position-aware
best-available-unit talent is experimental when starts/minutes are missing: it is
explicitly not an observed lineup. FotMob player form requires dated performances,
stable IDs and a known result-availability time. Cross-provider player IDs must be
mapped explicitly; names alone never identify a player.

## Chronological experiment

1. Tune goal-model decay/regularization and conservative XGBoost configurations on
   2022–23, trained using earlier results.
2. Fit ensemble weights on August–December 2023 out-of-sample predictions.
3. Fit optional temperature calibration on January–May 2024 predictions.
4. Choose model/feature group/calibration on 2024–25 by log loss, with Brier reported.
5. Evaluate the frozen choice on 2025–26. This period is not used for selection.
6. Report current-season replay separately; its outcomes were examined in earlier
   project work and cannot be described as an untouched test.

Dixon–Coles and XGBoost refit at monthly origins using only available earlier
results. The configured cutoff is explicit and reproducible. Promotion conversion
uses only completed earlier seasons, with regularization and a neutral fallback
when transition evidence is insufficient. Goal-model shrinkage toward those priors
weakens naturally as Premier League likelihood evidence accumulates.

Hybrid probabilities are reconciled by scaling the three H/D/A regions of the
Dixon–Coles score matrix. This is the minimum-KL update preserving conditional
score probabilities within each outcome. All displayed probabilities and expected
goals are derived from that same reconciled matrix.

Player-enhanced model training/ablation is unavailable until dated performances
and sufficient independent training/validation periods exist. The implemented
player modules must report missingness rather than synthesize ratings. Cup fatigue,
injury/lineup histories and tactical features are not inferred from league data.
