# Premier League Match Predictor & Dashboard

A Python forecasting pipeline with a browser-based matchday dashboard, scheduled
data collection, and auditable pre-match predictions. Developed iteratively from
a Poisson baseline into a modular Dixon–Coles/XGBoost evaluation system, with
explicit data-availability rules and a separate experimental player-form layer.

[Dashboard](https://burnsje20.github.io/Prem-Project/) ·
[Engineering decisions](docs/DECISIONS.md) ·
[Architecture](docs/ARCHITECTURE.md) ·
[Tests](tests/) ·
[Evaluation results](reports/model_comparison.csv)

## Engineering highlights

- **End-to-end delivery:** API/CSV ingestion, validated features, training,
  saved-model inference, forecast archives, evaluation, and a static web interface.
  GitHub Actions workflows configure scheduled updates and Pages deployment.
- **Measured baseline improvement:** the selected Dixon–Coles model reduced
  held-out log loss from **1.0844 to 1.0370**, approximately **4.4%**, across 380
  matches. Selection used an earlier season—not the final test results.
- **Testable boundaries:** separate data, modeling, provider-client, collection,
  presentation, and evaluation modules; **80 unit/integration tests passed** in
  local verification on November 9, 2026.
- **Failure-aware integration:** HTTP timeouts, explicit missing-data states,
  access/rate-limit handling for lineup batches, collector locking, and persisted
  refresh cooldowns. Offline inference requires no provider request.
- **Traceable predictions:** timestamped, non-overwriting archives with SHA-256
  source/model fingerprints, fixture identity checks, and fixed-policy evaluation.
- **Data engineering under constraints:** combined **6,524 historical PL and
  Championship matches** across seven seasons; added incremental player appearance
  ingestion and a human-reviewed cross-provider identity workflow.

**Stack:** Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · Requests ·
unittest · HTML/CSS · JavaScript · GitHub Actions · GitHub Pages.

## What the application produces

For each fixture: home/draw/away probabilities, expected goals, most likely
outcome and exact score, five likely scorelines, both-teams-to-score probability,
and over/under 2.5 goals. Forecasts include model/cutoff metadata and fallback
explanations. JSON probabilities are fractions; terminal output uses percentages.

The dashboard presents forecasts, recent team results, and an interactive
goal-rate simulation. It is generated static HTML with vanilla JavaScript and a
Tailwind CSS CDN dependency—no always-on application server. The slider is a
**what-if simulation**, not a newly validated forecast. The current builder reads
saved-model predictions; it does not automatically display the latest adjusted
forecast from the archive.

## Results: choose the model on evidence

The persisted experiment selected **uncalibrated Dixon–Coles** on 2024–25, then
evaluated the frozen choice on 2025–26. Lower log loss is better.

| Model | Selection log loss, 2024–25 | Test log loss, 2025–26 | Test outcome accuracy |
| --- | ---: | ---: | ---: |
| Original Poisson | 0.9990 | 1.0844 | 44.74% |
| **Dixon–Coles — selected** | **0.9747** | **1.0370** | **46.84%** |
| XGBoost, full features | 1.0040 | 1.0388 | 48.16% |
| Hybrid, full features | 0.9758 | 1.0356 | 47.37% |

Sources: [model comparison](reports/model_comparison.csv),
[saved manifest](artifacts/hybrid_v1/manifest.json), and
[paired date-block bootstrap intervals](reports/model_uncertainty.csv).
The hybrid's slightly lower test loss is not a reason to reselect on the test set;
its difference from Dixon–Coles has a bootstrap interval spanning zero. Outcome
accuracy is reported separately from the probability-quality objective.

These are historical results, **not proof of live performance**. The initial
[prospective evaluation](reports/prospective_initial_evaluation.json) contains
40 pending forecasts and no scored matches. Current-season replay was already
examined during development and is not an untouched evaluation.

## How the project evolved

| Iteration | Engineering decision | Result / tradeoff |
| --- | --- | --- |
| 1. Establish a baseline | Start with independent Poisson scoring and preserve the original backtest. | A reproducible comparison before adding complexity. |
| 2. Work around API limits | Limit API-Football fetching to three seasons; use supplied CSVs for 2019–20 through 2025–26. | Seven historical seasons without unavailable API requests. |
| 3. Handle promoted teams | Add Championship data and a regularized promotion-strength conversion. | Team-specific priors rather than a universal hand-picked discount. |
| 4. Compare model families | Evaluate Dixon–Coles, XGBoost, blending, and calibration chronologically. | Retain the simpler selected model when added complexity is not justified. |
| 5. Integrate live inputs | Separate lineup fetching from inference; require reviewed EA/FotMob ID mappings. | Provider failures and uncertain identities remain visible. |
| 6. Measure future forecasts | Add timestamped archives, latest-pre-kickoff evaluation, and a windowed collector. | Preserve what was forecast before observing a result. |
| 7. Explore player form | Collect appearances and add optional starter-form adjustment, retaining baseline fields. | A testable experimental layer, not a replacement for baseline validation. |
| 8. Deliver the interface | Generate a static dashboard and configure update/deployment workflows. | Low hosting complexity; freshness depends on jobs and model/cache updates. |

The [decision log](docs/DECISIONS.md) records rationale, alternatives, evidence,
and unresolved limitations—including decisions not to enable unsupported features.

## Architecture

~~~text
Historical PL + Championship CSVs
    -> validation -> time-safe features -> chronological experiments
    -> saved models -> offline prediction -> dashboard HTML

FotMob fixtures / appearances / lineups + saved-model forecasts
    -> provider client -> near-kickoff collector
    -> player context / optional experimental form adjustment
    -> timestamped archive -> later evaluation against completed results
                |
                +-> unresolved player IDs -> review -> approved crosswalk

GitHub Actions -> fixture refresh / appearance collection / lineups / dashboard
GitHub Pages   -> serves generated docs/ assets
~~~

| Boundary | Main implementation |
| --- | --- |
| Data contracts and lagged features | [data.py](epl_predictor/data.py), [features.py](epl_predictor/features.py) |
| Goal model, promotion priors, boosting | [goals.py](epl_predictor/goals.py), [promotion.py](epl_predictor/promotion.py), [boosting.py](epl_predictor/boosting.py) |
| Training, evaluation, inference | [workflow.py](epl_predictor/workflow.py), [evaluation.py](epl_predictor/evaluation.py), [predictor.py](epl_predictor/predictor.py) |
| Provider integration and collection | [fotmob_client.py](fotmob_client.py), [appearances.py](epl_predictor/appearances.py), [collection.py](epl_predictor/collection.py) |
| Player identity and form | [players.py](epl_predictor/players.py), [player_review.py](epl_predictor/player_review.py), [lineups.py](epl_predictor/lineups.py) |
| Provenance and delivery | [prospective.py](epl_predictor/prospective.py), [dashboard.py](epl_predictor/dashboard.py), [update_pipeline.py](update_pipeline.py) |

## Run locally

Run from the repository root in a virtual environment. Python **3.14** is the
locally verified runtime; the update workflow currently targets Python 3.11.
Dependency/runtime compatibility in that CI environment should be verified independently.

~~~powershell
python -m pip install -r requirements.txt
python -m unittest discover -v
python hybrid_predictor.py predict --output local_predictions.json
python hybrid_predictor.py build-dashboard --output reports/local_dashboard.html
~~~

Open the generated HTML in a browser. These prediction/dashboard commands use
local caches. The original API-Football fetcher separately needs FOOTBALL_API_KEY
in an ignored .env file. Never commit credentials.

The saved hybrid_v1 model has a **2026-11-09 00:00 UTC** data cutoff. Fixtures must
be on/after that cutoff and in the model's season. Refreshing fixtures is not the
same as retraining model parameters.

### Collect, review, and archive

~~~powershell
python hybrid_predictor.py collect-appearances
python hybrid_predictor.py collect-lineups --within-minutes 90 --refresh-minutes 10
python hybrid_predictor.py predict --archive-dir archives/prospective --output latest_predictions.json
~~~

The collector runs once, respects its lock/cooldown, and emits an archive and
unapproved player review when fixtures are eligible. Missing starters stay missing.
Review the emitted JSON before exporting an identity map:

~~~powershell
python hybrid_predictor.py export-player-map --review PATH_TO_REVIEW.json --output player_id_map_01.csv
python hybrid_predictor.py collect-lineups --player-id-map player_id_map_01.csv
~~~

Export requires an EA ID, explicit boolean approval, and a verification note;
name similarity alone is not approval. EA talent uses eligible men's snapshots
and separate positional units. There is no fixed EA/FotMob 60/40 blend.

### Experimental starter-form adjustment

~~~powershell
python hybrid_predictor.py collect-lineups --adjust-with-lineup-form
~~~

This optional path uses minutes-weighted recent player ratings and exponential
goal-rate scaling, with current default beta **0.1675**. Missing player history
uses a helper default of **6.80**; coverage/default flags must be inspected.
Selected baseline fields remain in the baseline_forecast object for comparison.

The earlier project narrative reported a 100-match form experiment, but its
reproducible fold definitions, fitting script, and prediction-level validation
artifact were not found in the repository review. Treat this coefficient and
reported benefit as **experimental**, not established out-of-sample improvement.
The unified update pipeline currently enables the adjustment; plain prediction
and lineup-collection commands do not enable it by default.

### Evaluate and reproduce

~~~powershell
python hybrid_predictor.py evaluate-live --archive-dir archives/prospective --results current_premier_league_results.csv --output reports/live_evaluation_02.json
python hybrid_predictor.py audit
python hybrid_predictor.py uncertainty
python hybrid_predictor.py train --artifacts artifacts/hybrid_v2 --reports reports/hybrid_v2
~~~

Use new report/artifact names. Archives and review/evaluation outputs refuse
overwrite; ordinary prediction output can be overwritten. Evaluation selects the
latest pre-kickoff forecast per fixture, checks ID/club/kickoff agreement, and
leaves missing results pending. The --filter-future flag means after the **model
cutoff**, not after the current wall clock.

## Automation and deployment

[update_pipeline.py](update_pipeline.py) refreshes a 30-day fixture window,
collects appearances for matches in the existing completed-results cache, collects
near-kickoff lineups, and rebuilds the dashboard. Run with:

~~~powershell
python update_pipeline.py
~~~

The configured [update workflow](.github/workflows/update_predictions.yml) runs
every 30 minutes during hours 11–22 UTC Friday–Monday, and every six hours
Tuesday–Thursday. Tests run after the pipeline and before generated changes are
committed. A separate [Pages workflow](.github/workflows/deploy_pages.yml) defines
deployment of docs/ on qualifying pushes or manual dispatch.

These are configured schedules, not uptime guarantees. The pipeline currently
does **not** refresh current_premier_league_results.csv or retrain the model.
Verify workflow executions, dependency compatibility, and Pages delivery end to
end before treating the dashboard as continuously current.

## Reliability, limitations, and next steps

- Tests cover probability normalization, positive rates, serialization, temporal
  isolation, promotion conversion, identity ambiguity, provider failures, locking,
  checksums, and evaluation. Passing tests are not complete test coverage or a
  production-readiness certification.
- Results/xG and EA snapshots use next-UTC-day availability assumptions.
  Appearances record retrieval time separately; assumed historical availability
  is not proof of original publication or revision timing.
- The EA dataset is **FC27**, one September 12, 2026 snapshot—not historical
  ratings for every training season. Current appearance data has **3,084 records
  across 100 matches**. These are repository snapshots, not live counters.
- Hashes detect accidental modification; they are not signatures or notarized
  timestamps. Repository-local provenance paths are normalized; external input
  paths and exports still require privacy review.
- Ignore rules exclude environment/key files. No claim of a complete security
  audit, permanently secret-free history, or guaranteed provider availability is made.
- Next priorities: reproduce the form experiment; enforce adequate two-sided
  starter coverage and strict as-of availability; refresh result/model state in
  automation; connect the dashboard to the intended archived forecast; evaluate
  future matches before claiming a live accuracy improvement.

See [engineering decisions](docs/DECISIONS.md) for the detailed iteration history
and [architecture notes](docs/ARCHITECTURE.md) for underlying modeling contracts.
