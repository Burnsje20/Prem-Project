"""Predict Premier League scorelines with a form-adjusted Poisson model.

The model trains only on completed historical seasons. It uses completed matches
from the live 2026-27 season solely to update team form before predicting the
remaining fixtures. Promoted teams begin from their prior Championship record,
adjusted using observed historical promotion transitions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fotmob_client import FotMobClient


PREMIER_LEAGUE_PATH = Path("premier_league_match_history.csv")
CHAMPIONSHIP_PATH = Path("championship_match_history.csv")
FIXTURES_PATH = Path("upcoming_premier_league_fixtures.csv")
CURRENT_RESULTS_PATH = Path("current_premier_league_results.csv")
PREDICTIONS_PATH = Path("poisson_predictions.csv")
ADJUSTMENT_PATH = Path("promotion_adjustment.json")
FORM_MATCHES = 5
FORM_PRIOR_GAMES = 8
MAX_CURRENT_PROMOTION_WEIGHT = 0.50


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Poisson score predictions.")
    parser.add_argument(
        "--no-refresh-current",
        action="store_true",
        help="Use the cached current-season results instead of refreshing from FotMob.",
    )
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    return parser.parse_args()


def load_matches(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path, parse_dates=["date"])
    matches["home_goals"] = pd.to_numeric(matches["home_goals"])
    matches["away_goals"] = pd.to_numeric(matches["away_goals"])
    return matches.sort_values("date").reset_index(drop=True)


def refresh_current_results() -> pd.DataFrame:
    """Fetch and cache completed current-season Premier League results."""
    results = pd.DataFrame(asdict(match) for match in FotMobClient().current_premier_league_results())
    if results.empty:
        raise RuntimeError("FotMob returned no completed current Premier League matches.")
    results.to_csv(CURRENT_RESULTS_PATH, index=False)
    return load_matches(CURRENT_RESULTS_PATH)


def team_rates(matches: pd.DataFrame, team: str) -> tuple[float, float, int]:
    """Return goals scored/game, goals conceded/game, and games for one club."""
    home = matches.loc[matches["home_team"] == team, ["home_goals", "away_goals"]]
    away = matches.loc[matches["away_team"] == team, ["away_goals", "home_goals"]]
    goals_for = home["home_goals"].sum() + away["away_goals"].sum()
    goals_against = home["away_goals"].sum() + away["home_goals"].sum()
    games = len(home) + len(away)
    if games == 0:
        raise ValueError(f"No matches found for {team}.")
    return goals_for / games, goals_against / games, games


def league_goal_rate(matches: pd.DataFrame) -> float:
    """Average goals per team per match in a competition."""
    return (matches["home_goals"].sum() + matches["away_goals"].sum()) / (2 * len(matches))


def transition_factors(
    premier_league: pd.DataFrame, championship: pd.DataFrame
) -> tuple[float, float, int]:
    """Estimate attack/defence promotion factors from completed prior transitions."""
    seasons = sorted(premier_league["season"].unique())
    attack_ratios, defence_ratios = [], []
    for season in seasons[1:]:
        start_year = int(season[:4])
        previous_season = f"{start_year - 1}-{str(start_year)[-2:]}"
        current_pl = premier_league.loc[premier_league["season"] == season]
        previous_pl = premier_league.loc[premier_league["season"] == previous_season]
        previous_championship = championship.loc[
            championship["season"] == previous_season
        ]
        promoted = (
            set(current_pl["home_team"])
            - set(previous_pl["home_team"])
        ) & (set(previous_championship["home_team"]) | set(previous_championship["away_team"]))
        for team in promoted:
            pl_for, pl_against, _ = team_rates(current_pl, team)
            ch_for, ch_against, _ = team_rates(previous_championship, team)
            if ch_for > 0 and ch_against > 0:
                attack_ratios.append(pl_for / ch_for)
                defence_ratios.append(pl_against / ch_against)

    if not attack_ratios:
        raise RuntimeError("Could not calculate historical promotion transition factors.")
    # Medians prevent one unusually strong or weak promoted side dominating the prior.
    return float(pd.Series(attack_ratios).median()), float(pd.Series(defence_ratios).median()), len(attack_ratios)


def current_promoted_teams(
    premier_league: pd.DataFrame, current_results: pd.DataFrame, fixtures: pd.DataFrame
) -> set[str]:
    """Find current PL teams absent from the most recent completed PL season."""
    latest_season = max(premier_league["season"])
    prior_teams = set(premier_league.loc[premier_league["season"] == latest_season, "home_team"])
    current_teams = (
        set(current_results["home_team"])
        | set(current_results["away_team"])
        | set(fixtures["home_team"])
        | set(fixtures["away_team"])
    )
    return current_teams - prior_teams


def current_transition_factors(
    promoted: set[str], current_results: pd.DataFrame, championship: pd.DataFrame
) -> tuple[float | None, float | None, int]:
    """Measure this season's promoted-team transition from their 2025-26 form."""
    prior_championship = championship.loc[championship["season"] == "2025-26"]
    attack_ratios, defence_ratios, matches_seen = [], [], 0
    for team in promoted:
        if team not in set(prior_championship["home_team"]) | set(prior_championship["away_team"]):
            continue
        current_for, current_against, games = team_rates(current_results, team)
        championship_for, championship_against, _ = team_rates(prior_championship, team)
        if championship_for > 0 and championship_against > 0:
            attack_ratios.append(current_for / championship_for)
            defence_ratios.append(current_against / championship_against)
            matches_seen += games
    if not attack_ratios:
        return None, None, 0
    return (
        float(pd.Series(attack_ratios).median()),
        float(pd.Series(defence_ratios).median()),
        matches_seen,
    )


def form_rating(
    team: str,
    current_results: pd.DataFrame,
    current_league_rate: float,
) -> tuple[float, float, int]:
    """Return current attack/defence strength from the team's latest five games."""
    team_matches = current_results.loc[
        (current_results["home_team"] == team) | (current_results["away_team"] == team)
    ].tail(FORM_MATCHES)
    if team_matches.empty:
        return 1.0, 1.0, 0
    goals_for, goals_against, games = team_rates(team_matches, team)
    return goals_for / current_league_rate, goals_against / current_league_rate, games


def prior_rating(
    team: str,
    premier_league: pd.DataFrame,
    championship: pd.DataFrame,
    promoted: set[str],
    attack_factor: float,
    defence_factor: float,
) -> tuple[float, float]:
    """Get a PL prior, mapping promoted sides from the prior Championship season."""
    latest_season = max(premier_league["season"])
    latest_pl = premier_league.loc[premier_league["season"] == latest_season]
    if team in set(latest_pl["home_team"]) | set(latest_pl["away_team"]):
        goals_for, goals_against, _ = team_rates(latest_pl, team)
        baseline = league_goal_rate(latest_pl)
        return goals_for / baseline, goals_against / baseline

    if team in promoted:
        latest_championship = championship.loc[championship["season"] == latest_season]
        goals_for, goals_against, _ = team_rates(latest_championship, team)
        baseline = league_goal_rate(latest_championship)
        return (goals_for / baseline) * attack_factor, (goals_against / baseline) * defence_factor

    raise ValueError(f"{team} has no Premier League or promoted-team history.")


def poisson_probabilities(home_lambda: float, away_lambda: float) -> tuple[float, float, float, str]:
    """Calculate H/D/A probabilities and the most likely scoreline through six goals."""
    home_probs = [math.exp(-home_lambda) * home_lambda**goals / math.factorial(goals) for goals in range(7)]
    away_probs = [math.exp(-away_lambda) * away_lambda**goals / math.factorial(goals) for goals in range(7)]
    home_win = draw = away_win = 0.0
    score_probability, scoreline = -1.0, "0-0"
    for home_goals, home_probability in enumerate(home_probs):
        for away_goals, away_probability in enumerate(away_probs):
            probability = home_probability * away_probability
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
            if probability > score_probability:
                score_probability, scoreline = probability, f"{home_goals}-{away_goals}"
    return home_win, draw, away_win, scoreline


def build_predictions(
    premier_league: pd.DataFrame,
    championship: pd.DataFrame,
    current_results: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    historical_attack, historical_defence, historical_samples = transition_factors(
        premier_league, championship
    )
    promoted = current_promoted_teams(premier_league, current_results, fixtures)
    current_attack, current_defence, current_samples = current_transition_factors(
        promoted, current_results, championship
    )
    current_weight = min(current_samples / 36, MAX_CURRENT_PROMOTION_WEIGHT)
    attack_factor = historical_attack
    defence_factor = historical_defence
    if current_attack is not None and current_defence is not None:
        attack_factor = (1 - current_weight) * historical_attack + current_weight * current_attack
        defence_factor = (1 - current_weight) * historical_defence + current_weight * current_defence

    historical_latest = premier_league.loc[
        premier_league["season"] == max(premier_league["season"])
    ]
    home_baseline = historical_latest["home_goals"].mean()
    away_baseline = historical_latest["away_goals"].mean()
    current_rate = league_goal_rate(current_results)
    rows = []
    for fixture in fixtures.itertuples(index=False):
        home_prior = prior_rating(
            fixture.home_team, premier_league, championship, promoted, attack_factor, defence_factor
        )
        away_prior = prior_rating(
            fixture.away_team, premier_league, championship, promoted, attack_factor, defence_factor
        )
        home_form = form_rating(fixture.home_team, current_results, current_rate)
        away_form = form_rating(fixture.away_team, current_results, current_rate)
        home_weight = home_form[2] / (home_form[2] + FORM_PRIOR_GAMES)
        away_weight = away_form[2] / (away_form[2] + FORM_PRIOR_GAMES)
        home_attack = (1 - home_weight) * home_prior[0] + home_weight * home_form[0]
        home_defence = (1 - home_weight) * home_prior[1] + home_weight * home_form[1]
        away_attack = (1 - away_weight) * away_prior[0] + away_weight * away_form[0]
        away_defence = (1 - away_weight) * away_prior[1] + away_weight * away_form[1]
        home_lambda = home_baseline * home_attack * away_defence
        away_lambda = away_baseline * away_attack * home_defence
        home_win, draw, away_win, scoreline = poisson_probabilities(home_lambda, away_lambda)
        rows.append(
            {
                "kickoff_utc": fixture.kickoff_utc,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "expected_home_goals": round(home_lambda, 2),
                "expected_away_goals": round(away_lambda, 2),
                "home_win_probability": round(home_win, 3),
                "draw_probability": round(draw, 3),
                "away_win_probability": round(away_win, 3),
                "most_likely_score": scoreline,
            }
        )
    adjustment = {
        "promoted_teams": sorted(promoted),
        "historical_attack_factor": round(historical_attack, 3),
        "historical_defence_factor": round(historical_defence, 3),
        "historical_promotion_samples": historical_samples,
        "current_attack_factor": None if current_attack is None else round(current_attack, 3),
        "current_defence_factor": None if current_defence is None else round(current_defence, 3),
        "current_promoted_matches": current_samples,
        "current_season_weight": round(current_weight, 3),
        "applied_attack_factor": round(attack_factor, 3),
        "applied_defence_factor": round(defence_factor, 3),
    }
    return pd.DataFrame(rows), adjustment


def main() -> None:
    args = parse_arguments()
    premier_league = load_matches(PREMIER_LEAGUE_PATH)
    championship = load_matches(CHAMPIONSHIP_PATH)
    fixtures = pd.read_csv(args.fixtures)
    current_results = (
        load_matches(CURRENT_RESULTS_PATH)
        if args.no_refresh_current
        else refresh_current_results()
    )
    predictions, adjustment = build_predictions(
        premier_league, championship, current_results, fixtures
    )
    predictions.to_csv(args.output, index=False)
    ADJUSTMENT_PATH.write_text(json.dumps(adjustment, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(predictions)} predictions to {args.output}.")
    print(f"Promoted-team adjustment: {json.dumps(adjustment)}")


if __name__ == "__main__":
    main()