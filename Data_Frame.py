"""Build a clean Premier League match-history dataset for model training."""

from pathlib import Path

import pandas as pd


CSV_PATH = Path("EPL_Match_Data_2000_2026.csv")
OUTPUT_PATH = Path("premier_league_match_history.csv")
CHAMPIONSHIP_PATHS = sorted(Path(".").glob("E1*.csv"))
CHAMPIONSHIP_OUTPUT_PATH = Path("championship_match_history.csv")


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
    historical["competition"] = "Premier League"
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
            "competition",
            "source",
        ]
    ]


def load_championship_matches() -> pd.DataFrame:
    """Load completed 2019-20 to 2025-26 Championship matches from E1 CSVs."""
    matches = []
    for path in CHAMPIONSHIP_PATHS:
        source = pd.read_csv(path)
        source["date"] = pd.to_datetime(source["Date"], dayfirst=True, errors="coerce")
        # Use the first fixture to identify the source file's season. A date
        # cutoff would mislabel the July 2020 COVID-rearranged 2019-20 matches.
        start_year = source["date"].min().year
        if not 2019 <= start_year <= 2025:
            continue
        source["season"] = season_label(start_year)
        source["home_team"] = source["HomeTeam"]
        source["away_team"] = source["AwayTeam"]
        source["home_goals"] = source["FTHG"]
        source["away_goals"] = source["FTAG"]
        source["result"] = source["FTR"]
        source["status"] = "FT"
        source["competition"] = "Championship"
        source["source"] = "championship_csv"
        matches.append(
            source[
                [
                    "date",
                    "season",
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "result",
                    "status",
                    "competition",
                    "source",
                ]
            ]
        )

    if not matches:
        raise FileNotFoundError("No Championship files matching 'E1*.csv' were found.")
    return pd.concat(matches, ignore_index=True).sort_values("date")


def build_match_history() -> pd.DataFrame:
    """Build training data through 2025-26; do not include target season 2026-27."""
    history = load_historical_matches().sort_values("date")
    history.to_csv(OUTPUT_PATH, index=False)
    return history


def build_championship_history() -> pd.DataFrame:
    """Export Championship history separately for promoted-team form features."""
    history = load_championship_matches()
    history.to_csv(CHAMPIONSHIP_OUTPUT_PATH, index=False)
    return history


if __name__ == "__main__":
    premier_league_matches = build_match_history()
    championship_matches = build_championship_history()
    print(f"Saved {len(premier_league_matches)} matches to {OUTPUT_PATH}.")
    print(premier_league_matches.groupby("season").size())
    print(f"Saved {len(championship_matches)} matches to {CHAMPIONSHIP_OUTPUT_PATH}.")
    print(championship_matches.groupby("season").size())
