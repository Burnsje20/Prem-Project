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
FOTMOB_LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"
FOTMOB_MATCH_DETAILS_URL = "https://www.fotmob.com/api/data/matchDetails"
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
    "Nottm Forest": "Nottingham",
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


@dataclass(frozen=True)
class CompletedMatch:
    fixture_id: int
    date: datetime
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
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

    def fetch_lineup(self, fixture_id: int) -> dict[str, Any]:
        """Fetch a timestamped lineup snapshot, never same-match performance data.

        This undocumented website feed may omit starters or change shape. A list
        of eleven players is not by itself evidence of a confirmed lineup.
        """
        if isinstance(fixture_id, bool) or not str(fixture_id).isdigit() or int(fixture_id) < 1:
            raise ValueError("fixture_id must be a positive integer")
        response = self.session.get(
            FOTMOB_MATCH_DETAILS_URL, params={"matchId": int(fixture_id)}, timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_lineup(response.json(), int(fixture_id), datetime.now(timezone.utc))

    @staticmethod
    def _parse_lineup(payload, fixture_id, fetched_at):
        general = payload["general"]
        if int(general["matchId"]) != fixture_id:
            raise ValueError("FotMob returned a different fixture")
        if int(general["leagueId"]) != PREMIER_LEAGUE_ID or general.get("gender") != "male":
            raise ValueError("Lineup is not from the men's Premier League")
        kickoff = datetime.fromisoformat(general["matchTimeUTCDate"].replace("Z", "+00:00"))
        if kickoff.tzinfo is None or fetched_at.tzinfo is None:
            raise ValueError("Lineup timestamps must include a timezone")
        lineup = payload.get("content", {}).get("lineup") or {}
        status = payload.get("header", {}).get("status") or {}
        blocked = (general.get("started") or general.get("finished") or
                   status.get("started") or status.get("finished") or
                   status.get("cancelled") or status.get("postponed") or fetched_at >= kickoff)
        result = {"fixture_id": fixture_id, "source": "fotmob",
                  "fetched_at": fetched_at.isoformat(), "kickoff_utc": kickoff.isoformat(),
                  "provider_lineup_type": lineup.get("lineupType"),
                  "status": "not_prematch" if blocked else "prematch"}
        for side in ["home", "away"]:
            team = general[f"{side}Team"]
            supplied = lineup.get(f"{side}Team") or {}
            if supplied and int(supplied["id"]) != int(team["id"]):
                raise ValueError("Lineup team identity mismatch")
            starters = supplied.get("starters") or []
            if not isinstance(starters, list) or len(starters) > 11:
                raise ValueError("Invalid starting lineup")
            # Deliberately whitelist identity fields: ratings, events, season
            # averages and other post-kickoff fields must never enter context.
            clean = [{"player_id": str(p["id"]), "name": p["name"]} for p in starters]
            ids = [p["player_id"] for p in clean]
            if any(not pid.isdigit() or int(pid) < 1 for pid in ids) or len(ids) != len(set(ids)):
                raise ValueError("Missing or duplicate starter identities")
            confirmation = "unavailable"
            if clean:
                confirmation = ("projected" if lineup.get("lineupType") in {"predicted", "projected"}
                                else "provided_unverified")
                if lineup.get("confirmed") is True:
                    confirmation = "confirmed"
            result[side] = {"team": normalize_team_name(team["name"]),
                            "team_id": int(team["id"]), "status": confirmation,
                            "starters": [] if blocked else clean}
        return result

    def upcoming_fixtures(self, start_date: date, days: int = 14) -> list[Fixture]:
        """Return upcoming Premier League fixtures over the requested date range."""
        if days < 1:
            raise ValueError("days must be at least 1")

        fixtures: dict[int, Fixture] = {}
        for day_offset in range(days):
            for fixture in self.matches_on(start_date.fromordinal(start_date.toordinal() + day_offset)):
                fixtures[fixture.fixture_id] = fixture
        return sorted(fixtures.values(), key=lambda fixture: fixture.kickoff_utc)

    def current_premier_league_results(self) -> list[CompletedMatch]:
        """Return completed matches from FotMob's currently selected PL season."""
        response = self.session.get(
            FOTMOB_LEAGUE_URL,
            params={"id": PREMIER_LEAGUE_ID},
            timeout=self.timeout,
        )
        response.raise_for_status()
        matches = response.json().get("overview", {}).get("leagueOverviewMatches", [])

        completed_matches = []
        for match in matches:
            status = match.get("status", {})
            home, away = match.get("home", {}), match.get("away", {})
            if not status.get("finished") or home.get("score") is None or away.get("score") is None:
                continue

            kickoff = datetime.fromisoformat(status["utcTime"].replace("Z", "+00:00"))
            kickoff = kickoff.astimezone(timezone.utc)
            completed_matches.append(
                CompletedMatch(
                    fixture_id=int(match["id"]),
                    date=kickoff,
                    season=season_for(kickoff),
                    home_team=normalize_team_name(home["name"]),
                    away_team=normalize_team_name(away["name"]),
                    home_goals=int(home["score"]),
                    away_goals=int(away["score"]),
                    status=status.get("reason", {}).get("short", "FT"),
                )
            )
        return sorted(completed_matches, key=lambda match: match.date)

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
