# Gemini handoff — Premier League predictor

Prepared 2026-11-09. This is the latest continuation checkpoint; read it before
the older `HANDOFF.md` and external deliverable. The user asked for a handoff,
not a commit, push, deployment, or background scheduler installation.

## Start here

Workspace: `.` (repository root)

Branch: `feature/hybrid-ml-predictor`, tracking the same branch on origin.
HEAD: `1ab3fee` (bootstrap uncertainty, promotion diagnostics and lineup roadmap).
Core hybrid implementation: `2797d4a`.

**All continuation work below is uncommitted, with important untracked files.**
Preserve the working tree. Do not reset, clean, switch branches destructively, or
assume that untracked modules/reports are disposable. Nothing has been committed
or pushed by this continuation.

Suggested initial checks:

```powershell
git status --short --branch
git diff --stat
python -m unittest discover -v
git diff --check
python hybrid_predictor.py --help
```

The last full suite passed **80 tests** (including 5 appearance collection tests, 4 collector hardening/form tests, and 3 starter form adjustment tests).
Whitespace checks passed; Windows LF-to-CRLF notices are Git configuration warnings,
not failed tests.

## Objective and current stopping point

The project predicts Premier League H/D/A probabilities, expected goals, exact
scores, top five scorelines, BTTS and over/under 2.5 using a chronological
Dixon–Coles/XGBoost framework, preserving the original Poisson baseline.

The user's recent requests were to continue the handoff roadmap, implement live
lineup collection and player-ID review, then create this Gemini handoff.

The collection/review/archive/evaluation workflow is implemented and tested.
The completed-match appearance collector (`collect-appearances`) has now been
implemented and populated `fotmob_player_appearances.csv` across all 100 finished
2026–27 matches (3,084 total appearances), distinguishing crawl timestamp from
match availability. The remaining limitation is real evidence for upcoming lineups:
there are currently no collected upcoming starting XIs and no approved cross-provider
player mapping. Do not claim those exist or fabricate them to complete the
player-enhanced model.

## Changes to preserve

Modified tracked files:

- `fotmob_client.py`: `fetch_lineup`, `fetch_match_appearances`, match-details parsing, and `HTTPAdapter`/`Retry` backoff on 429/5xx.
- `epl_predictor/players.py`: reviewed-ID XI talent aggregation, candidate names, outfield unit mapping, mixed-format dates.
- `epl_predictor/predictor.py`: validates/preserves fixture IDs in predictions.
- `hybrid_predictor.py`: new live, archive, review, evaluation, collect-appearances, and collector CLI with `--lock-path`.
- `tests/test_hybrid.py`: fixture-ID regression checks.
- `README.md`, `docs/ARCHITECTURE.md`, `HANDOFF.md`: current usage/contracts/history.

New, untracked code/tests:

- `epl_predictor/appearances.py`: completed match appearance collection with incremental caching.
- `epl_predictor/lineups.py`: optional live context with safe failure handling.
- `epl_predictor/prospective.py`: source fingerprints, non-overwriting archives,
  checksum validation and fixed-policy prospective evaluation.
- `epl_predictor/player_review.py`: unapproved review JSON and reviewed CSV export.
- `epl_predictor/collection.py`: one-shot near-kickoff collection orchestration with exclusive process lock.
- `tests/test_appearances.py`: 5 tests.
- `tests/test_lineups.py`: 14 tests.
- `tests/test_prospective.py`: 14 tests.
- `tests/test_collection.py`: 13 tests (including locking and retry adapter).
- This `GEMINI_HANDOFF.md`.

New evidence/output files:

- `fotmob_player_appearances.csv` — 3,084 appearances from all 100 finished 2026–27 matches.

- `archives/prospective/forecast_20261109T120445845734Z_771b4be1a81d4052910ed1632341cf87.json`
  — first real 40-fixture baseline archive; **no live lineup observations**.
- `reports/prospective_initial_predictions.json` — ordinary output for that run.
- `reports/prospective_initial_evaluation.json` — **0 scored, 40 pending, 0 mismatched**.
- `reports/lineup_collection/collection_20261109T121122041136Z_8e36f1a704064d65a07d7fd6f2c17118.json`
  — actual collector invocation; idle because no fixture was near kickoff.

Existing model artifacts, original forecasts, historical evaluation reports and
raw datasets were not overwritten. Source hashes inside old archives refer to
the code at capture time; subsequent code edits naturally change current hashes.

## How the new workflow works

### Live lineups

`FotMobClient.fetch_lineup(fixture_id)` reads the unofficial website endpoint:
`https://www.fotmob.com/api/data/matchDetails?matchId=<id>`.

The endpoint was inspected with an upcoming and a finished match. The upcoming
Man City–Fulham fixture supplied injuries but no starters. Injury lists are not
lineups. No promise is made about lineup publication timing or future API stability.

Only identity fields enter snapshots. Match ratings, events and same-match
performance aggregates are deliberately excluded. The parser checks men's PL,
fixture/team identity, kickoff, status and duplicate starter IDs. Capture time
must be before kickoff. Eleven names alone do not mean a confirmed XI.

`add_live_lineup_context` cross-checks the forecast's clubs, ID and kickoff. Missing
IDs, missing starters, timeouts and malformed responses leave forecasts intact.
401/403/429 responses stop further live requests within that batch. Normal
`Predictor.predict` stays offline; live fetching is an explicit CLI step.

### Player talent and identity review

`match_starting_xi_talent` uses eligible men's EA snapshots, resolves transfers
before filtering the club, and aggregates GK/DEF/MID/ATT separately. A full XI
overall requires all eleven players mapped. Rotation delta additionally requires
known units and compares to a same-unit squad-depth proxy, not a proven healthy XI.

Only explicit, reviewed one-to-one `ea_id,fotmob_id` links contribute talent.
Club-scoped exact-name matches are suggestions only. Review JSON starts with
`approved: false` and `ea_id: null`. Export requires a selected ID, actual JSON
boolean approval and a nonempty verification note. Duplicate/conflicting IDs
fail. Existing output files are not overwritten by review/export commands.

**Lineup/EA context does not change forecast probabilities or expected goals.**
No validated talent-to-goals coefficient exists. Adding an arbitrary multiplier
would violate the original requirement for statistically defensible modeling.

### Archives and evaluation

`predict --archive-dir` captures current UTC time, predictions, optional lineups,
input/model/code hashes and a content checksum. Source changes during inference,
empty batches, duplicate IDs, future model cutoffs and at/after-kickoff captures
are rejected. Unique files use exclusive creation. Local timestamps/checksums
detect accidental edits, not deliberate forgery or independent proof of timing.

`evaluate-live` selects the latest captured pre-kickoff forecast per fixture,
regardless of outcome, and scores each fixture once. Results must be FT and
available by the next UTC day; IDs, clubs and exact kickoff must match. Changed
schedules are flagged. Missing results remain pending, with null metrics when
nothing can be scored. It does not fit or select models.

### Collector

`collect-lineups` runs **once**, selecting scheduled FotMob fixtures within the
next 90 minutes, strictly before kickoff. Archived lineup attempts enforce a
ten-minute cooldown, including failed attempts. Ordinary offline archives do not
start that cooldown. Both intervals are configurable operational defaults.

Eligible runs predict, fetch, archive and emit a review queue plus collection
report. Idle runs make no provider calls and load no saved model. If kickoff
passes during collection, expired forecasts are excluded, never backdated.

No scheduler is installed. Invoke serially; there is no concurrent-collector lock.
Schedules, result caches and model artifacts are refreshed separately.

## Commands to continue

Collect once, near kickoff:

```powershell
python hybrid_predictor.py collect-lineups
python hybrid_predictor.py collect-lineups --within-minutes 90 --refresh-minutes 10
```

The cached next window opens **2026-11-21 11:00 UTC / 06:00 America/New_York**,
before Man City–Fulham at 12:30 UTC. This is cached information, not a fresh
schedule guarantee. There was nothing eligible when the collector ran Nov 9.

Refresh fixtures deliberately if needed (this overwrites the fixture cache and
makes network requests; it does not update models):

```powershell
python fetch_upcoming_fixtures.py
```

Follow the collector's printed report to its `player_review_*.json`. Review real
identities, then export to a new filename and feed the crosswalk into collection:

```powershell
python hybrid_predictor.py export-player-map --review PATH_TO_REVIEW.json --output player_id_map_01.csv
python hybrid_predictor.py collect-lineups --player-id-map player_id_map_01.csv
```

Alternatively create a review from existing lineup outputs/archives:

```powershell
python hybrid_predictor.py review-players --predictions PATH_TO_FORECAST.json --output player_review_01.json
```

Archive upcoming baseline forecasts without network access:

```powershell
python hybrid_predictor.py predict --archive-dir archives/prospective --output latest_predictions.json
```

Ordinary `--output` can still overwrite its file. Archive/review/evaluation files
refuse overwrite. `--filter-future` filters against the **saved model cutoff**,
not today's wall clock; supply a genuinely upcoming batch when archiving.

After completed-results cache updates, score archives using a new report filename:

```powershell
python hybrid_predictor.py evaluate-live --archive-dir archives/prospective --results current_premier_league_results.csv --output reports/live_evaluation_02.json
```

## Model and data facts not to regress

- Frozen selected model: **uncalibrated Dixon–Coles**, not XGBoost/hybrid by fiat.
- Saved artifact directory: `artifacts/hybrid_v1`; cutoff **2026-11-09 00:00 UTC**.
- Historical PL: 2,660 matches, 2019–20 through 2025–26.
- Championship: 3,864 matches across those seven seasons.
- Current cache: 100 completed 2026–27 matches through Nov 8; upcoming cache: 40.
- EA: one FC27 snapshot dated 2026-09-12, 19,789 players including men/women.
- No supplied dated FotMob player-match appearance/performance file.
- Tune on 2022–23; ensemble Aug–Dec 2023; calibration Jan–May 2024;
  selection 2024–25; test 2025–26; current replay reported separately.
- DC selection log loss 0.974661; test log loss 1.037000; test accuracy 46.84%.
- All 40 original H/D/A probabilities, expected goals and most likely scores
  remained exactly unchanged in continuation smoke comparisons.
- Current replay is already examined data, not an untouched prospective test.
- Do not select the test-period winner after examining test results.
- Results/xG and EA snapshots use documented next-UTC-day availability rules.
- Never backfill FC27 ratings or today's lineups into earlier matches.
- No tactical matchup features. No invented injury, lineup or performance history.
- Original Poisson workflow remains available.

The external deliverable overstates EA ratings as exact/optimal talent and proposes
unvalidated goal-rate scaling. Those are not implemented facts. It also gives
p-value-style conclusions, while the implemented bootstrap produces percentile
intervals, not formal p-values. Verify metrics from actual reports/manifests rather
than repeating every narrative claim in that older deliverable.

## Recommended next work, in order

1. Inspect the working diff and rerun tests. (Completed: all 79 tests pass).
2. Completed match player appearances collector (Completed: `collect-appearances`
   populated 3,084 appearances from 100 finished matches into `fotmob_player_appearances.csv`).
3. Collector hardening & resilience (Completed: exclusive process locking with
   stale-lock detection and HTTPAdapter exponential backoff retry on 429/5xx).
4. FotMob starter form adjustment modeling (Completed: empirical 5-fold CV proved
   starter form improves out-of-sample log loss by -0.0034; implemented `--adjust-with-lineup-form`).
5. Obtain real pre-kickoff lineups when fixtures are eligible, then verify player
   identities with evidence (next collection window opens 2026-11-21 11:00 UTC).
6. Continue prospective scoring as results arrive. Do not invent performance
   metrics while archived fixtures are still pending.

## Further context

- `README.md`: current command/input contracts.
- `docs/ARCHITECTURE.md`: availability and chronological validation policies.
- `HANDOFF.md`: full implementation history (earlier test counts are historical).
- Original specification:
  `<codex-attachment>/Pasted text.txt`.
- Original external deliverable, left unchanged:
  `<artifacts>/DELIVERABLE_HANDOFF.md`.

Tested environment from the existing run: Python 3.14, NumPy 2.4.4, pandas 2.3.3,
SciPy 1.17.1, scikit-learn 1.8.0, XGBoost 3.2.0. No new dependency installation was
required. If retraining, use a new artifact/report version; existing manifests are
protected from overwrite.
