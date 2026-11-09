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

Live lineup context is an opt-in CLI enrichment step (`epl_predictor/lineups.py`),
not a change to model fitting or the offline `Predictor`. FotMob match-details
responses are reduced to fixture/team/player identities, provider lineup status
and a UTC retrieval timestamp. Started/finished/cancelled/postponed fixtures and
captures at or after kickoff are excluded. Club, fixture ID and kickoff must agree
with the forecast. Per-fixture failures leave probabilities unchanged; provider
401/403/429 responses stop further requests in that batch.

Starting XI talent uses an explicitly reviewed one-to-one ID bridge. Club-scoped
normalized exact names provide review candidates only, never automatic identities.
EA records must be men's records from eligible snapshots; resolve transfers before
club filtering. Missing players reduce coverage. Full-XI overall and same-unit
squad-depth rotation delta require eleven matched players (and known units for
the delta). A depth proxy is not a known full-strength lineup. Live EA availability
is evaluated at snapshot capture, while fitted team parameters retain their saved
cutoff. Neither lineup presence nor EA ratings imply empirical predictive benefit.
No unvalidated multiplier is applied to Dixon–Coles rates. Historical training and
reported test metrics are unchanged; future player-enhanced training needs dated
pre-kickoff captures, verified identities and separate chronological validation.

Prospective captures (`prospective.py`) use actual capture time, never a supplied
backdated timestamp. Unique, exclusive-create JSON files store predictions, lineup
context and input/model/code SHA-256 hashes. Rehashing detects source changes during
inference; content checksums detect later accidental edits. These local timestamps
are not independently authenticated. Evaluation uses the latest valid capture per
fixture, irrespective of outcome, and requires exact fixture/club/kickoff agreement
with next-day-available FT results. Pending and mismatched matches are reported,
not assigned fabricated targets. No model selection or fitting occurs here.

`player_review.py` creates unapproved identity-review JSON from unresolved lineup
players. Candidate names are suggestions only; a boolean approval, selected EA ID
and verification note are required to export a one-to-one crosswalk. Review and
export outputs use exclusive creation to preserve earlier review decisions.

`collection.py` combines these pieces in a one-shot near-kickoff workflow. It
filters cached fixtures by actual UTC time, scheduled status and FotMob source,
then observes a per-fixture cooldown from prior archived lineup attempts. Offline
forecasts do not block collection. No eligible matches means no provider calls or
saved-model loading. Successful runs emit an archive, an unapproved review queue
and an exclusive-create collection report. Matches that cross kickoff during the
batch are excluded rather than backdated. The 90-minute window and 10-minute
refresh interval are configurable operational defaults. No scheduler or concurrent
collector coordination is installed; invoke serially and keep fixtures/models fresh.

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
