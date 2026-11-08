# Agent handoff — hybrid Premier League predictor

## User instruction and stopping point

The user requested implementation of the attached complete Dixon–Coles + XGBoost
system, then explicitly asked: **“Finish up where you are and create a handoff file
for another agent to work on.”** This handoff marks a working checkpoint, not a
claim that every item of the large specification has been independently reviewed.

Full specification:
`C:\Users\Kaku\.codex\attachments\aefdd8fb-38d5-4261-8671-1a50af6fb873\Pasted text.txt`

Workspace: `C:\Users\Kaku\Documents\GitHub\AI_Premier_League_Project\Prem Project`

Branch: **`feature/hybrid-ml-predictor`**, tracking its same-named origin branch.
All work in this implementation is currently **uncommitted/untracked**. No commits,
pushes, merges or deployments were made. Do not lose the new directories when
switching branches. Existing original Poisson/data/fetch scripts were preserved.

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
- `tests/test_hybrid.py`: 19 new synthetic unit/integration tests.

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
small and currently have no uncertainty interval. Selected DC current replay:
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
- `python -m unittest discover -v`: **27 passed** (8 existing + 19 new).
- Full 40-fixture prediction succeeded, including Coventry, Hull and Ipswich.
- Single-match CLI and alias-normalization smoke test run at handoff.
- Compilation succeeded and `git diff --check` passed for tracked changes.
- Mathematical tests cover tau, normalization, marginals, score aggregation,
  H/D/A reconciliation, positive pairwise rates, repeatable fitting and serialization.
- Leakage tests cover future outcomes, same-day outcomes and snapshot availability.
- Missing booster/player features and unknown data paths have explicit fallback tests.

## Recommended next work (do not mistake these for completed checks)

1. **Independent code/statistical review.** Review goal-model parameterization,
   promotion conversion, chronological splitting and score reconciliation. Add a
   synthetic test that actually learns a conversion from full-sized promotion
   cohorts and proves later cohorts cannot change it; the current synthetic
   promotion test exercises the insufficient-sample fallback only.
2. **Tighten experiment evidence.** Add paired date-block bootstrap uncertainty for
   model loss differences. Current ablations compare XGBoost feature groups; they
   do not isolate switching the DC promotion prior off. Rest/H2H and baseline Elo
   settings are transparent fixed feature choices, not separately optimized.
3. **Player-data gap.** Obtain legitimate dated FotMob appearances and valid provider
   ID links before training player-enhanced models. The EA depth proxy is not an
   expected actual XI. Reliable starter/minutes histories, explicit lineup input,
   injuries, positional normalization validation and player ablations remain limited.
   Do not backfill the FC27 snapshot into earlier seasons to manufacture training data.
4. **Reproducibility metadata.** Manifest hashes currently cover processed PL,
   Championship, current result cache and EA CSV. Add the raw EPL xG source hash,
   source-code/config hashes and installed dependency versions. Date parsing details
   were added to the inventory after the training process began; `audit` was rerun
   at handoff to refresh that report. The fitted model code itself was unchanged.
5. **Persistence and CLI polish.** Review partial-run recovery and report overwrite
   behavior, input-validation errors, artifact integrity checks and future fixture
   filtering. This is a CLI; no web frontend was built because none existed.
6. **Source quality.** Continue checking the supplied data and unresolved API aliases;
   do not infer that supplied records or snapshot timestamps were independently
   verified. Current data has already been seen; plan a future untouched evaluation.
7. **Final requirement audit and commit.** Compare the original large specification
   to this implementation, document remaining omissions, decide which generated
   reports/artifacts to version, then stage/commit only when requested. No user
   permission to push or deploy has been assumed.

The old Poisson workflow and its reports remain available for comparison. Avoid
deleting them or mixing generated prediction CSVs into training data.
