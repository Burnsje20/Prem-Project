"""Replay completed matches without exposing the model to their results.

Default input is the local FotMob cache; this command makes no network requests.
Historical-only mode withholds the entire target season. Walk-forward mode uses
only results from earlier UTC dates, because the cache has kickoff times but no
final-whistle timestamps. All games on a day are predicted with the same cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from poisson_distribution import (
    CHAMPIONSHIP_PATH, CURRENT_RESULTS_PATH, PREMIER_LEAGUE_PATH,
    build_predictions, load_matches, poisson_probabilities,
)


PROBABILITY_COLUMNS = ["home_win_probability", "draw_probability", "away_win_probability"]
OUTCOMES = ["H", "D", "A"]


def validate_matches(matches: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reject incomplete, duplicate, or invalid result data before evaluating."""
    data = matches.copy()
    required = ["date", "season", "home_team", "away_team", "home_goals", "away_goals"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError(f"{label}: missing match values")
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="raise")
    for column in ["home_goals", "away_goals"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].map(lambda x: math.isfinite(x) and x >= 0 and x == int(x)).all():
            raise ValueError(f"{label}: invalid goal values")
    if "status" in data and not data["status"].eq("FT").all():
        raise ValueError(f"{label}: only completed full-time league games are supported")
    if data.duplicated(["season", "home_team", "away_team"]).any():
        raise ValueError(f"{label}: duplicate league fixture")
    if "fixture_id" in data and data["fixture_id"].duplicated().any():
        raise ValueError(f"{label}: duplicate fixture ID")
    if data["home_team"].eq(data["away_team"]).any():
        raise ValueError(f"{label}: a club cannot play itself")
    return data.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Brier is the sum of squared errors across H/D/A (range 0 to 2)."""
    accuracy, exact, loss, brier, goal_error = [], [], [], [], []
    for row in predictions.to_dict("records"):
        probabilities = [row[column] for column in PROBABILITY_COLUMNS]
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
            raise ValueError("Invalid outcome probability")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("Outcome probabilities must sum to one")
        actual_index = OUTCOMES.index(row["actual_result"])
        predicted_index = max(range(3), key=probabilities.__getitem__)
        accuracy.append(int(actual_index == predicted_index))
        exact.append(int(row["most_likely_score"] == row["actual_score"]))
        loss.append(-math.log(max(probabilities[actual_index], 1e-15)))
        brier.append(sum((p - int(i == actual_index)) ** 2 for i, p in enumerate(probabilities)))
        goal_error.extend([
            abs(row["expected_home_goals"] - row["actual_home_goals"]),
            abs(row["expected_away_goals"] - row["actual_away_goals"]),
        ])
    n = len(predictions)
    return {
        "matches": n,
        "correct_results": sum(accuracy),
        "result_accuracy": sum(accuracy) / n,
        "exact_scores": sum(exact),
        "exact_score_accuracy": sum(exact) / n,
        "log_loss": sum(loss) / n,
        "brier_score": sum(brier) / n,
        "goals_mae_per_team": sum(goal_error) / (2 * n),
    }


def run_backtest(
    premier_league: pd.DataFrame,
    championship: pd.DataFrame,
    results: pd.DataFrame,
    season: str = "2026-27",
) -> tuple[pd.DataFrame, dict]:
    results = validate_matches(results, "Test results")
    results = results.loc[results["season"].eq(season)].reset_index(drop=True)
    if results.empty:
        raise ValueError(f"No completed matches available for {season}")
    start_year = int(season[:4])
    previous_season = f"{start_year - 1}-{str(start_year)[-2:]}"
    histories = []
    for name, frame in [("Premier League", premier_league), ("Championship", championship)]:
        history = validate_matches(frame, name)
        # Filter by season AND check chronology, so accidentally supplying the
        # target season in either historical file cannot leak into calibration.
        history = history.loc[history["season"].str[:4].astype(int) < start_year]
        if history.empty or previous_season not in set(history["season"]):
            raise ValueError(f"{name}: missing prior season {previous_season}")
        if history["date"].max() >= results["date"].min():
            raise ValueError(f"{name}: historical dates overlap the test season")
        histories.append(history)
    premier_league, championship = histories
    latest_pl = premier_league.loc[premier_league["season"].eq(previous_season)]
    baseline_home, baseline_away = latest_pl.home_goals.mean(), latest_pl.away_goals.mean()
    baseline_h, baseline_d, baseline_a, baseline_score = poisson_probabilities(baseline_home, baseline_away)
    records = []
    for cutoff, day_matches in results.groupby(results["date"].dt.normalize(), sort=True):
        fixtures = day_matches[["date", "season", "home_team", "away_team"]].rename(
            columns={"date": "kickoff_utc"}
        )
        for mode in ["historical_only", "walk_forward", "league_average"]:
            available = results.loc[results["date"] < cutoff].copy()
            if mode != "walk_forward":
                available = available.iloc[:0].copy()
            if mode == "league_average":
                predicted = fixtures.copy()
                predicted["expected_home_goals"] = baseline_home
                predicted["expected_away_goals"] = baseline_away
                predicted["home_win_probability"] = baseline_h
                predicted["draw_probability"] = baseline_d
                predicted["away_win_probability"] = baseline_a
                predicted["most_likely_score"] = baseline_score
                adjustment = {"current_season_weight": 0}
            else:
                predicted, adjustment = build_predictions(
                    premier_league, championship, available, fixtures
                )
            # Scores become visible only after prediction, exclusively for scoring.
            for prediction, actual in zip(predicted.to_dict("records"), day_matches.to_dict("records"), strict=True):
                home, away = int(actual["home_goals"]), int(actual["away_goals"])
                prediction.update({
                    "mode": mode,
                    "fixture_id": actual.get("fixture_id"),
                    "season": season,
                    "information_cutoff_utc": cutoff.isoformat(),
                    "current_results_used": len(available),
                    "latest_result_used_utc": None if available.empty else available["date"].max().isoformat(),
                    "current_promotion_weight": adjustment["current_season_weight"],
                    "actual_home_goals": home,
                    "actual_away_goals": away,
                    "actual_score": f"{home}-{away}",
                    "actual_result": "H" if home > away else "A" if away > home else "D",
                })
                records.append(prediction)
    predictions = pd.DataFrame(records)
    summary = {
        "season": season,
        "first_test_kickoff": results["date"].min().isoformat(),
        "last_test_kickoff": results["date"].max().isoformat(),
        "training_season_end": previous_season,
        "cutoff_policy": "Only results from earlier UTC dates; same-day results excluded.",
        "parameter_policy": "Existing model parameters held fixed; no tuning on test outcomes.",
        "limitations": [
            "Retrospective replay of the supplied cache, not a prospective live trial.",
            "Source data authenticity and historical corrections are not independently verified.",
            "A single partial season is insufficient to establish future accuracy.",
        ],
        "metrics": {
            mode: evaluate_predictions(group) for mode, group in predictions.groupby("mode")
        },
    }
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=CURRENT_RESULTS_PATH)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--output", type=Path, default=Path("poisson_backtest_predictions.csv"))
    parser.add_argument("--summary", type=Path, default=Path("poisson_backtest_summary.json"))
    args = parser.parse_args()
    predictions, summary = run_backtest(
        load_matches(PREMIER_LEAGUE_PATH), load_matches(CHAMPIONSHIP_PATH),
        load_matches(args.results), args.season,
    )
    predictions.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary["metrics"]).T.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Saved match-level comparisons to {args.output} and metrics to {args.summary}.")


if __name__ == "__main__":
    main()
