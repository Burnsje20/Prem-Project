# Premier League Match Predictor & Live Dashboard

An end-to-end, mathematically rigorous English Premier League match prediction system and live web dashboard. The project combines time-decayed Dixon–Coles Poisson modeling, empirical lineup and player form adjustments, leak-free prospective forecast archiving, and automated cloud execution on GitHub Pages.

**Live Dashboard**: [https://burnsje20.github.io/Prem-Project/](https://burnsje20.github.io/Prem-Project/)

---

## Table of Contents

- [Key Architectural & Empirical Decisions](#key-architectural--empirical-decisions)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Lineup & Starter Form Adjustment](#lineup--starter-form-adjustment)
- [Automated Pipeline & Cloud Scheduling](#automated-pipeline--cloud-scheduling)
- [Prospective Forecast Archiving & Evaluation](#prospective-forecast-archiving--evaluation)
- [Model Training & Reproducibility](#model-training--reproducibility)
- [Test Suite & Security Integrity](#test-suite--security-integrity)
- [Codex Contributions & Extensions](#codex-contributions--extensions)

---

## Key Architectural & Empirical Decisions

Throughout development, every model choice, feature addition, and engineering step was governed by empirical validation, strict chronological integrity, and parsimony:

1. **Model Selection by Held-Out Log Loss (Dixon–Coles vs. XGBoost/Hybrid)**:
   - Rather than forcing an arbitrary machine learning hybrid, models were evaluated on chronological out-of-sample test data (2025–26 season).
   - Time-decayed Dixon–Coles achieved a test log loss of **0.9781**, outperforming both XGBoost (**1.0029**) and the probability-weighted ensemble (**0.9858**).
   - *Decision*: Dixon–Coles was chosen as the deployed production model for superior probability calibration and low-score dependence modeling ($\tau(\lambda, \mu, \rho)$).

2. **Empirical Validation Before Engineering (Option B Before Option A)**:
   - Before investing extensive engineering effort into a manual cross-provider ID bridge (Option A), we ran a full 5-fold cross-validation experiment (Option B) across all 100 completed 2026–27 matches (2,200 starting lineup slots) to evaluate EA FC 25 ratings vs. FotMob starter form.
   - *Findings*:
     - **EA FC Baseline Squad Talent** was redundant with Dixon–Coles attack/defence parameters ($\Delta \text{LL} = -0.000024$).
     - **Starting XI Rotation Delta** was noisy and actively degraded performance ($+0.0255$ log loss penalty; accuracy dropped from 49% to 43%).
     - **FotMob Starter Form is genuine predictive signal**: Reduced out-of-sample log loss from 0.961938 to **0.958515** ($\Delta = -0.003423$) and improved Brier score from 0.571716 to **0.569609**, with an optimal cross-validated scaling parameter $\beta = 0.1675$.
   - *Decision*: Formally integrated FotMob starter form adjustments into the prospective lineup pipeline while keeping unvalidated player talent adjustments strictly observational.

3. **Immutable Baseline Preservation**:
   - Whenever starter form adjustments are applied, expected goal rates scale as $\lambda_{adj} = \lambda \cdot e^{\beta \Delta}$ and $\mu_{adj} = \mu \cdot e^{-\beta \Delta}$.
   - *Decision*: The original baseline Dixon–Coles forecast is always preserved in `baseline_forecast` alongside the form-adjusted probabilities, guaranteeing complete auditability and preventing destructive updates.

4. **Zero-Dependency Interactive Dashboard on GitHub Pages**:
   - *Decision*: Built a standalone, client-side interactive web dashboard in `docs/index.html` using modern CSS and vanilla JavaScript with zero external frameworks or CDNs. Includes an interactive Poisson simulation slider allowing users to explore hypothetical starter form swings in real time.

5. **Automated Cloud Pipeline via GitHub Actions**:
   - *Decision*: Deployed a 4-step pipeline running on a dual-cron schedule: every 30 minutes during match windows (Friday–Monday 11:00–22:00 UTC) to capture live 60–90 minute pre-kickoff lineups, and every 6 hours midweek to ingest completed fixtures and update ratings.

6. **Repository Security & Provenance Sanitization**:
   - *Decision*: Performed full historical security audits (204,783 lines of git diffs verified secret-free). Sanitized all local machine paths in provenance digests to relative repository paths (e.g. `epl_predictor/players.py`), making the repository fully safe for public hosting.

---

## System Architecture

```text
Prem-Project/
├── epl_predictor/              # Core prediction package
│   ├── goals.py                # Penalized time-decayed Dixon–Coles MLE & market reconciliation
│   ├── boosting.py             # Multiclass XGBoost models (native JSON persistence)
│   ├── features.py             # Lagged rolling features, league-only rest, Elo, xG
│   ├── promotion.py            # Ridge regression Championship-to-PL prior conversion
│   ├── players.py              # FotMob appearances, starter form calculation (beta=0.1675)
│   ├── lineups.py              # Starter form goal rate scaling & market re-normalization
│   ├── collection.py           # Imminent lineup collector with process locking & cooldowns
│   ├── prospective.py          # Tamper-evident forecast archiving & evaluation
│   ├── dashboard.py            # Matchday HTML dashboard generator
│   └── data.py                 # Schema validation, aliases, and chronological availability
├── docs/
│   ├── index.html              # Live dashboard hosted on GitHub Pages
│   └── ARCHITECTURE.md         # Detailed chronological validation and availability rules
├── update_pipeline.py          # Unified 4-step execution script for local and CI/CD runs
├── hybrid_predictor.py         # Unified CLI (audit, train, predict, collect-lineups, etc.)
├── fotmob_client.py            # Resilient HTTP client for FotMob match and lineup data
├── fetch_upcoming_fixtures.py  # 30-day upcoming fixture ingestion
├── .github/workflows/
│   ├── update_predictions.yml  # Scheduled pipeline runner (every 30m matchdays, 6h midweek)
│   └── deploy_pages.yml        # GitHub Pages automated deployment
└── tests/                      # Comprehensive test suite (80 unit and integration tests)
```

---

## Quick Start

### 1. Installation

Requires Python 3.11+:

```powershell
python -m pip install -r requirements.txt
```

### 2. Run Predictions (Offline Mode)

Generate predictions for upcoming fixtures using the saved production model:

```powershell
python hybrid_predictor.py predict
```

This reads cached upcoming fixtures, computes H/D/A probabilities, expected goals, top five scorelines, Both Teams To Score (BTTS), and Over/Under 2.5 markets, saving the results to `hybrid_predictions.json`.

Predict a specific fixture:

```powershell
python hybrid_predictor.py predict --home "Arsenal" --away "Chelsea" --date "2026-11-21T17:30:00Z"
```

---

## Lineup & Starter Form Adjustment

### Incremental Player Appearance Collection

Collect match appearances, starter/sub minutes, and FotMob match ratings for all completed matches:

```powershell
python hybrid_predictor.py collect-appearances
```

This updates `fotmob_player_appearances.csv` while enforcing chronological availability (performances only become available the day after the match).

### Near-Kickoff Lineup Collection with Starter Form Adjustment

Collect official starting lineups for matches kicking off within the next 90 minutes and apply the empirically validated starter form adjustment:

```powershell
python hybrid_predictor.py collect-lineups --adjust-with-lineup-form
```

What this does:
1. Queries FotMob for confirmed starting XIs within the 90-minute window.
2. Calculates the 5-match weighted average form differential ($\Delta$) between home and away starting XIs.
3. Scales Dixon–Coles expected goals by $e^{\pm 0.1675 \Delta}$.
4. Re-normalizes the bivariate Poisson matrix and updates all market probabilities.
5. Preserves the untouched Dixon–Coles baseline in `baseline_forecast`.
6. Archives the forecast with SHA-256 provenance hashes to `archives/prospective/`.

---

## Automated Pipeline & Cloud Scheduling

The entire update lifecycle is unified into `update_pipeline.py`, which executes four automated steps:

```powershell
python update_pipeline.py
```

1. **Step 1: Upcoming Fixtures** — Refreshes the 30-day Premier League schedule from FotMob.
2. **Step 2: Completed Appearances** — Incrementally captures minutes and ratings for newly finished fixtures into `fotmob_player_appearances.csv`.
3. **Step 3: Lineup Collection** — Checks for near-kickoff matches (<90m), fetches confirmed starters, applies form adjustments, and archives forecasts.
4. **Step 4: Dashboard Build** — Compiles the interactive Matchday 12 dashboard into `docs/index.html` and `reports/matchday_12_dashboard.html`.

### GitHub Actions Workflow

Two automated workflows in `.github/workflows/` keep the predictor running without local intervention:
- **`update_predictions.yml`**:
  - **Matchday Schedule**: Every 30 minutes from 11:00 to 22:00 UTC on Friday, Saturday, Sunday, and Monday (`*/30 11-22 * * 5,6,0,1`).
  - **Midweek Schedule**: Every 6 hours on Tuesday, Wednesday, and Thursday (`0 */6 * * 2,3,4`).
  - Verifies test suite integrity (`python -m unittest discover -v`) before committing updates.
- **`deploy_pages.yml`**: Automatically deploys `docs/` to GitHub Pages whenever `docs/index.html` is updated.

---

## Prospective Forecast Archiving & Evaluation

To prevent hindsight bias, forecasts can be archived with cryptographic checksums before kickoff and evaluated later against actual results.

### Archiving Pre-Match Predictions

```powershell
python hybrid_predictor.py predict --archive-dir archives/prospective --output latest_predictions.json
```

Each run generates a unique `forecast_*.json` file containing:
- Full prediction payload (baseline + form-adjusted probabilities).
- UTC capture timestamp (replaces backdated claims).
- Cryptographic SHA-256 digests of all source files and model artifacts.

### Fixed-Policy Out-of-Sample Evaluation

Once matches conclude and `current_premier_league_results.csv` is updated, evaluate forecast accuracy:

```powershell
python hybrid_predictor.py evaluate-live --archive-dir archives/prospective --results current_premier_league_results.csv --output reports/live_evaluation_01.json
```

- Strictly scores the **latest captured pre-kickoff forecast per fixture**.
- Matches must match exact IDs, clubs, and scheduled kickoff.
- Evaluates out-of-sample log loss, Brier score, and accuracy.

---

## Model Training & Reproducibility

To audit the data, retrain models from scratch, or evaluate bootstrap uncertainty:

```powershell
# Audit data integrity, missingness, and aliases
python hybrid_predictor.py audit

# Retrain models across chronological splits
python hybrid_predictor.py train --artifacts artifacts/hybrid_v2 --reports reports/hybrid_v2

# Evaluate paired block-bootstrap uncertainty
python hybrid_predictor.py uncertainty
```

### Chronological Splitting Strategy

| Phase | Time Period | Objective |
| :--- | :--- | :--- |
| **Warm-up** | 2019–20 | Initial feature lag generation |
| **Model Fitting** | 2020–21 to 2021–22 | Base Dixon–Coles parameter estimation |
| **Tuning** | 2022–23 | Optimal decay rate and prior regularization search |
| **Ensemble & Weighting** | Autumn 2023 | Out-of-sample probability blend weights |
| **Calibration** | Spring 2024 | Temperature scaling validation |
| **Model Selection** | 2024–25 | Final model family selection via log loss |
| **Held-Out Test** | 2025–26 | Final out-of-sample test evaluation |
| **Current Season** | 2026–27 | Prospective live forecasting and lineup tracking |

---

## Test Suite & Security Integrity

The test suite covers unit logic, leak-free temporal data isolation, Dixon–Coles mathematical constraints, lineup collection, process locking, and prospective archiving:

```powershell
python -m unittest discover -v
```

**Status**: **80/80 tests passing** (100% pass rate).

### Security & Privacy Audits
- **0 secrets in git history**: Verified by automated regex scanning over 204,783 lines of git diff history.
- **`.env` protected**: Environment secrets are strictly ignored by `.gitignore` and have never been committed.
- **Sanitized provenance**: All archive hash dictionaries and markdown documents record clean repository-relative paths with no local machine paths or personal identifiers.

---

## Codex Contributions & Extensions

<!-- 
NOTE FOR CODEX:
Please add your specific model enhancements, data scripts, or feature notes in this section.
-->

- *Original Poisson Distribution Baseline*: Implemented initial bivariate Poisson baseline in `poisson_distribution.py` and historical backtesting harness in `backtest_poisson.py`.
- *Historical Data Ingestion*: API-Football fixture ingestion pipeline implemented in `fetch_data.py`.
- *Additional notes*: [Add additional Codex contributions or feature details here]

---

## Mathematical References

- **Dixon, M. J., & Coles, S. G. (1997)**. *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics, 46(2), 265–280.
- **Brier, G. W. (1950)**. *Verification of forecasts expressed in terms of probability*. Monthly Weather Review, 78(1), 1–3.
- **Chen, T., & Guestrin, C. (2016)**. *XGBoost: A Scalable Tree Boosting System*. KDD '16.
