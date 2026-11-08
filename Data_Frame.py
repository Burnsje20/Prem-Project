"""Build a clean Premier League match-history dataset for model training."""

from pathlib import Path

import pandas as pd


CSV_PATH = Path("EPL_Match_Data_2000_2026.csv")
OUTPUT_PATH = Path("premier_league_match_history.csv")


def season_label(start_year: int) -> str:
    """Return a readable season label, for example 2026 becomes '2026-27'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def load_historical_matches() -> pd.DataFrame:
    """Load completed 2019-20 to 2025-26 matches from the supplied CSV."""
    source = pd.read_csv(CSV_PATH)
    season_codes = source["Season"].astype(str).str.zfill(4)
    start_year = 2000 + season_codes.str[:2].astype(int)

    historical = source.loc[start_year.between(2019, 2025)].copy()
    historical["date"] = pd.to_datetime(
        {
            "year": historical["Year"],
            "month": historical["Month"],
            "day": historical["Day"],
        },
        errors="coerce",
    )
    historical["season"] = start_year.loc[historical.index].map(season_label)
    historical["home_team"] = historical["HomeTeam"]
    historical["away_team"] = historical["AwayTeam"]
    historical["home_goals"] = historical["HomeGoal"]
    historical["away_goals"] = historical["AwayGoal"]
    historical["result"] = historical["Result"]
    historical["status"] = "FT"
    historical["source"] = "historical_csv"

    return historical[
        [
            "date",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "status",
            "source",
        ]
    ]


def build_match_history() -> pd.DataFrame:
    """Build training data through 2025-26; do not include target season 2026-27."""
    history = load_historical_matches().sort_values("date")
    history.to_csv(OUTPUT_PATH, index=False)
    return history


if __name__ == "__main__":
    matches = build_match_history()
    print(f"Saved {len(matches)} matches to {OUTPUT_PATH}.")
    print(matches.groupby("season").size())
