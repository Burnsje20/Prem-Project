"""Minimal client for FotMob's public match-list feed.

FotMob does not provide an official public API. This client uses the same
read-only fixture feed used by its website and should be refreshed sparingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


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

    def __init__(self, timeout: int = 20, max_retries: int = 3, backoff_factor: float = 0.5) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "PremierLeaguePredictor/1.0",
            }
        )
        if max_retries > 0:
            retry_strategy = Retry(
                total=max_retries,
                backoff_factor=backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

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

    def fetch_match_appearances(self, fixture_id: int) -> list[dict[str, Any]]:
        """Fetch player appearances (minutes, rating, position) for a completed match.

        Explicitly records fetched_at so that crawl time is never confused with
        historical match availability.
        """
        if isinstance(fixture_id, bool) or not str(fixture_id).isdigit() or int(fixture_id) < 1:
            raise ValueError("fixture_id must be a positive integer")
        response = self.session.get(
            FOTMOB_MATCH_DETAILS_URL, params={"matchId": int(fixture_id)}, timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_appearances(response.json(), int(fixture_id), datetime.now(timezone.utc))

    @staticmethod
    def _parse_appearances(payload: dict[str, Any], fixture_id: int, fetched_at: datetime) -> list[dict[str, Any]]:
        general = payload["general"]
        if int(general["matchId"]) != fixture_id:
            raise ValueError("FotMob returned a different fixture")
        if int(general["leagueId"]) != PREMIER_LEAGUE_ID or general.get("gender") != "male":
            raise ValueError("Match is not from the men's Premier League")
        status = payload.get("header", {}).get("status") or {}
        if not (general.get("finished") or status.get("finished")):
            raise ValueError("Match is not finished")
        kickoff = datetime.fromisoformat(general["matchTimeUTCDate"].replace("Z", "+00:00")).astimezone(timezone.utc)
        available_at = max(kickoff, (kickoff + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))

        red_cards = {}
        events = payload.get("content", {}).get("matchFacts", {}).get("events", {}).get("events", [])
        for ev in events:
            if ev.get("type") == "Card" and ev.get("card") == "Red":
                pid = str(ev.get("playerId") or ev.get("player", {}).get("id") or "")
                if pid and pid != "None":
                    red_cards[pid] = ev.get("time", 90)

        appearances = []
        lineup = payload.get("content", {}).get("lineup") or {}
        pos_map = {0: "GK", 1: "DEF", 2: "MID", 3: "ATT"}

        for side in ["homeTeam", "awayTeam"]:
            team_key = side[:-4] + "Team"
            team_name_norm = normalize_team_name(general[team_key]["name"])
            team_data = lineup.get(side) or {}
            starters = team_data.get("starters") or []
            subs = team_data.get("subs") or []

            for p in starters:
                pid = str(p["id"])
                perf = p.get("performance") or {}
                sub_events = perf.get("substitutionEvents") or []
                sub_out = None
                for ev in sub_events:
                    if ev.get("type") == "subOut":
                        sub_out = ev.get("time", 90)
                end_min = sub_out if sub_out is not None else red_cards.get(pid, 90)
                minutes = max(1, min(130, int(end_min)))
                rating_val = perf.get("rating")
                rating = float(rating_val) if rating_val is not None else None
                pos = pos_map.get(p.get("usualPlayingPositionId"), "MID")
                appearances.append({
                    "player_id": pid,
                    "player_name": p.get("name", ""),
                    "match_id": str(fixture_id),
                    "date": kickoff.isoformat(),
                    "available_at": available_at.isoformat(),
                    "retrieved_at": fetched_at.isoformat(),
                    "team": team_name_norm,
                    "position": pos,
                    "rating": rating,
                    "minutes": minutes,
                    "is_starter": True,
                    "is_home": (side == "homeTeam"),
                })

            for p in subs:
                pid = str(p["id"])
                perf = p.get("performance") or {}
                sub_events = perf.get("substitutionEvents") or []
                sub_in, sub_out = None, None
                for ev in sub_events:
                    if ev.get("type") == "subIn":
                        sub_in = ev.get("time", 0)
                    elif ev.get("type") == "subOut":
                        sub_out = ev.get("time", 90)
                if sub_in is None:
                    continue
                end_min = sub_out if sub_out is not None else red_cards.get(pid, 90)
                minutes = max(1, min(130, int(end_min) - int(sub_in)))
                rating_val = perf.get("rating")
                rating = float(rating_val) if rating_val is not None else None
                pos = pos_map.get(p.get("usualPlayingPositionId"), "MID")
                appearances.append({
                    "player_id": pid,
                    "player_name": p.get("name", ""),
                    "match_id": str(fixture_id),
                    "date": kickoff.isoformat(),
                    "available_at": available_at.isoformat(),
                    "retrieved_at": fetched_at.isoformat(),
                    "team": team_name_norm,
                    "position": pos,
                    "rating": rating,
                    "minutes": minutes,
                    "is_starter": False,
                    "is_home": (side == "homeTeam"),
                })

        return appearances


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
