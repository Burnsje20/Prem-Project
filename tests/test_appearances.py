from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from epl_predictor.appearances import collect_match_appearances
from epl_predictor.players import fotmob_units, player_form, validate_appearances
from fotmob_client import FotMobClient


def sample_match_payload(match_id=5795363, finished=True):
    return {
        "general": {
            "matchId": match_id,
            "leagueId": 47,
            "gender": "male",
            "matchTimeUTCDate": "2026-08-21T19:00:00.000Z",
            "finished": finished,
            "homeTeam": {"id": 9825, "name": "Arsenal"},
            "awayTeam": {"id": 8669, "name": "Coventry"},
        },
        "header": {
            "status": {"finished": finished}
        },
        "content": {
            "matchFacts": {
                "events": {
                    "events": [
                        {"type": "Card", "card": "Red", "time": 42, "playerId": 101}
                    ]
                }
            },
            "lineup": {
                "homeTeam": {
                    "starters": [
                        {"id": 101, "name": "Player RedCard", "usualPlayingPositionId": 1,
                         "performance": {"rating": 5.0}},
                        {"id": 102, "name": "Player SubOut", "usualPlayingPositionId": 2,
                         "performance": {"rating": 7.5, "substitutionEvents": [{"type": "subOut", "time": 68}]}},
                        {"id": 103, "name": "Player Full90", "usualPlayingPositionId": 0,
                         "performance": {"rating": 8.0}},
                    ],
                    "subs": [
                        {"id": 104, "name": "Sub Active", "usualPlayingPositionId": 3,
                         "performance": {"rating": 6.5, "substitutionEvents": [{"type": "subIn", "time": 68}]}},
                        {"id": 105, "name": "Sub Unused", "usualPlayingPositionId": 2,
                         "performance": None},
                    ]
                },
                "awayTeam": {
                    "starters": [
                        {"id": 201, "name": "Away GK", "usualPlayingPositionId": 0,
                         "performance": {"rating": 6.8}},
                        {"id": 202, "name": "Away Mid", "usualPlayingPositionId": 2,
                         "performance": {"rating": 6.2}},
                    ],
                    "subs": []
                }
            }
        }
    }


class MatchAppearancesParsingTests(unittest.TestCase):
    def test_parse_appearances_starters_and_subs(self):
        payload = sample_match_payload()
        apps = FotMobClient._parse_appearances(payload, 5795363, datetime.now(timezone.utc))

        df = pd.DataFrame(apps)
        # Should have 3 home starters + 1 active sub + 2 away starters = 6 players
        self.assertEqual(len(df), 6)

        # Unused sub (id 105) must be excluded
        self.assertNotIn("105", df["player_id"].values)

        # Player 101: Red card at minute 42 -> 42 minutes
        p101 = df[df.player_id == "101"].iloc[0]
        self.assertEqual(p101.minutes, 42)
        self.assertEqual(p101.rating, 5.0)
        self.assertEqual(p101.position, "DEF")
        self.assertTrue(p101.is_starter)
        self.assertTrue(p101.is_home)
        self.assertEqual(p101.team, "Arsenal")

        # Player 102: Subbed out at minute 68 -> 68 minutes
        p102 = df[df.player_id == "102"].iloc[0]
        self.assertEqual(p102.minutes, 68)
        self.assertEqual(p102.rating, 7.5)
        self.assertEqual(p102.position, "MID")

        # Player 103: Full 90 -> 90 minutes
        p103 = df[df.player_id == "103"].iloc[0]
        self.assertEqual(p103.minutes, 90)
        self.assertEqual(p103.position, "GK")

        # Player 104: Sub in at 68 -> 90 - 68 = 22 minutes
        p104 = df[df.player_id == "104"].iloc[0]
        self.assertEqual(p104.minutes, 22)
        self.assertEqual(p104.position, "ATT")
        self.assertFalse(p104.is_starter)

        # Next-day availability rule
        self.assertGreaterEqual(pd.Timestamp(p101.available_at), pd.Timestamp(p101.date))

        # Must pass validate_appearances contract
        validated = validate_appearances(df)
        self.assertEqual(len(validated), 6)

    def test_parse_appearances_unrated_player(self):
        payload = sample_match_payload()
        # Make player 103 unrated
        payload["content"]["lineup"]["homeTeam"]["starters"][2]["performance"]["rating"] = None
        apps = FotMobClient._parse_appearances(payload, 5795363, datetime.now(timezone.utc))
        df = pd.DataFrame(apps)
        p103 = df[df.player_id == "103"].iloc[0]
        self.assertTrue(pd.isna(p103.rating))
        self.assertEqual(p103.minutes, 90)
        validated = validate_appearances(df)
        self.assertTrue(pd.isna(validated[validated.player_id == "103"].iloc[0].rating))

    def test_parse_rejects_unfinished_or_wrong_fixture(self):
        payload = sample_match_payload(finished=False)
        with self.assertRaises(ValueError):
            FotMobClient._parse_appearances(payload, 5795363, datetime.now(timezone.utc))

        payload_finished = sample_match_payload(finished=True)
        with self.assertRaises(ValueError):
            FotMobClient._parse_appearances(payload_finished, 9999999, datetime.now(timezone.utc))

        payload_finished["general"]["leagueId"] = 100
        with self.assertRaises(ValueError):
            FotMobClient._parse_appearances(payload_finished, 5795363, datetime.now(timezone.utc))


class AppearancesCollectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results_csv = self.root / 'results.csv'
        self.output_csv = self.root / 'appearances.csv'

        pd.DataFrame([
            {'fixture_id': 1001, 'status': 'FT', 'home_team': 'Arsenal', 'away_team': 'Coventry'},
            {'fixture_id': 1002, 'status': 'FT', 'home_team': 'Chelsea', 'away_team': 'Leeds'},
        ]).to_csv(self.results_csv, index=False)

    def test_collect_match_appearances_end_to_end(self):
        mock_client = MagicMock()
        mock_client.fetch_match_appearances.side_effect = lambda fid: [
            {
                "player_id": f"{fid}_p1", "player_name": "P1", "match_id": str(fid),
                "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
                "retrieved_at": "2026-11-09T12:00:00+00:00", "team": "Arsenal",
                "position": "MID", "rating": 7.0, "minutes": 90, "is_starter": True, "is_home": True
            },
            {
                "player_id": f"{fid}_p2", "player_name": "P2", "match_id": str(fid),
                "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
                "retrieved_at": "2026-11-09T12:00:00+00:00", "team": "Arsenal",
                "position": "GK", "rating": 6.5, "minutes": 90, "is_starter": True, "is_home": True
            }
        ]

        # Initial collection
        report = collect_match_appearances(
            results_path=self.results_csv,
            output_path=self.output_csv,
            client=mock_client,
            delay=0.0
        )
        self.assertEqual(report['newly_collected_matches'], 2)
        self.assertEqual(report['total_appearances'], 4)
        self.assertEqual(report['total_matches'], 2)
        self.assertTrue(self.output_csv.exists())

        # Second collection should be incremental and skip already collected matches
        report2 = collect_match_appearances(
            results_path=self.results_csv,
            output_path=self.output_csv,
            client=mock_client,
            delay=0.0
        )
        self.assertEqual(report2['newly_collected_matches'], 0)
        self.assertEqual(report2['total_appearances'], 4)
        # Mock client should only have been called twice total (from first run)
        self.assertEqual(mock_client.fetch_match_appearances.call_count, 2)

    def test_appearances_integration_with_fotmob_units(self):
        apps = pd.DataFrame([
            {"player_id": "1", "player_name": "A", "match_id": "1",
             "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
             "team": "Arsenal", "position": "GK", "rating": 7.0, "minutes": 90},
            {"player_id": "2", "player_name": "B", "match_id": "1",
             "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
             "team": "Arsenal", "position": "DEF", "rating": 7.5, "minutes": 90},
            {"player_id": "3", "player_name": "C", "match_id": "1",
             "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
             "team": "Arsenal", "position": "MID", "rating": 8.0, "minutes": 90},
            {"player_id": "4", "player_name": "D", "match_id": "1",
             "date": "2026-08-21T19:00:00+00:00", "available_at": "2026-08-22T00:00:00+00:00",
             "team": "Arsenal", "position": "ATT", "rating": 8.5, "minutes": 90},
        ])
        validated = validate_appearances(apps)
        self.assertEqual(len(validated), 4)

        form = player_form(validated, '2026-09-01')
        self.assertEqual(len(form), 4)

        units = fotmob_units(validated, '2026-09-01')
        self.assertIn("Arsenal", units)
        self.assertAlmostEqual(units["Arsenal"]["fm_GK_form"], 7.0)
        self.assertAlmostEqual(units["Arsenal"]["fm_DEF_form"], 7.5)
        self.assertAlmostEqual(units["Arsenal"]["fm_MID_form"], 8.0)
        self.assertAlmostEqual(units["Arsenal"]["fm_ATT_form"], 8.5)


if __name__ == '__main__':
    unittest.main()
