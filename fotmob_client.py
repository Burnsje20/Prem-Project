"""Minimal client for FotMob's public match-list feed.

FotMob does not provide an official public API. This client uses the same
read-only fixture feed used by its website and should be refreshed sparingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import requests


# The older /api/matches route now returns 404; this is FotMob's current
# public website feed for match listings.
FOTMOB_MATCHES_URL = "https://www.fotmob.com/api/data/matches"
PREMIER_LEAGUE_ID = 47

# FotMob's display names differ from the historical CSV's team names.
TEAM_NAME_ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham",
    "Sheffield United": "Sheffield",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


@dataclass(frozen=True)
class Fixture:
    fixture_id: int
    kickoff_utc: datetime
    season: str
    home_team: str
    away_team: str
    status: str
    source: str = "fotmob"


def normalize_team_name(name: str) -> str:
    """Map a FotMob display name to the naming convention in the training CSV."""
    return TEAM_NAME_ALIASES.get(name, name)


def season_for(kickoff: datetime) -> str:
    """Determine the English football season containing a fixture kickoff."""
    start_year = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


class FotMobClient:
    """Retrieve scheduled Premier League fixtures from FotMob's match feed."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "PremierLeaguePredictor/1.0",
            }
        )

    def matches_on(self, match_date: date) -> list[Fixture]:
        """Return unstarted, non-cancelled Premier League fixtures for one day."""
        response = self.session.get(
            FOTMOB_MATCHES_URL,
            params={"date": match_date.strftime("%Y%m%d")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._parse_fixtures(payload)

    def upcoming_fixtures(self, start_date: date, days: int = 14) -> list[Fixture]:
        """Return upcoming Premier League fixtures over the requested date range."""
        if days < 1:
            raise ValueError("days must be at least 1")

        fixtures: dict[int, Fixture] = {}
        for day_offset in range(days):
            for fixture in self.matches_on(start_date.fromordinal(start_date.toordinal() + day_offset)):
                fixtures[fixture.fixture_id] = fixture
        return sorted(fixtures.values(), key=lambda fixture: fixture.kickoff_utc)

    @staticmethod
    def _parse_fixtures(payload: dict[str, Any]) -> list[Fixture]:
        fixtures = []
        for league in payload.get("leagues", []):
            # Several countries call their top division "Premier League";
            # FotMob identifies England's competition with ID 47 and ENG.
            is_premier_league = (
                league.get("primaryId") == PREMIER_LEAGUE_ID
                and league.get("ccode") == "ENG"
            )
            if not is_premier_league:
                continue

            for match in league.get("matches", []):
                status = match.get("status", {})
                if status.get("finished") or status.get("cancelled") or status.get("started"):
                    continue

                kickoff_text = status.get("utcTime")
                if not kickoff_text:
                    continue
                kickoff = datetime.fromisoformat(kickoff_text.replace("Z", "+00:00"))
                kickoff = kickoff.astimezone(timezone.utc)
                fixtures.append(
                    Fixture(
                        fixture_id=match["id"],
                        kickoff_utc=kickoff,
                        season=season_for(kickoff),
                        home_team=normalize_team_name(match["home"]["name"]),
                        away_team=normalize_team_name(match["away"]["name"]),
                        status=status.get("reason", {}).get("short", "Scheduled"),
                    )
                )
        return fixtures
