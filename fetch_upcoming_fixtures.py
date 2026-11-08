"""Download upcoming Premier League fixtures for score prediction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from fotmob_client import FotMobClient


OUTPUT_PATH = Path("upcoming_premier_league_fixtures.csv")
OUTPUT_COLUMNS = [
    "fixture_id",
    "kickoff_utc",
    "season",
    "home_team",
    "away_team",
    "status",
    "source",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download scheduled Premier League fixtures from FotMob."
    )
    parser.add_argument(
        "--days", type=int, default=14, help="Number of calendar days to fetch (default: 14)."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date.today(),
        help="First date to query in YYYY-MM-DD format (default: today).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    fixtures = FotMobClient().upcoming_fixtures(args.start_date, args.days)
    output = pd.DataFrame((asdict(fixture) for fixture in fixtures), columns=OUTPUT_COLUMNS)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output)} upcoming fixtures to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
