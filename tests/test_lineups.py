from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from fotmob_client import FotMobClient
from epl_predictor.lineups import add_live_lineup_context
from epl_predictor.players import match_starting_xi_talent, validate_player_bridge


NOW = datetime(2026, 11, 21, 11, tzinfo=timezone.utc)
KICKOFF = '2026-11-21T12:30:00Z'


def payload():
    return {'general': {'matchId': '123', 'leagueId': 47, 'gender': 'male',
                       'matchTimeUTCDate': KICKOFF, 'started': False,
                       'homeTeam': {'id': 1, 'name': 'Arsenal'},
                       'awayTeam': {'id': 2, 'name': 'Chelsea'}},
            'content': {'lineup': {'lineupType': 'standard',
                'homeTeam': {'id': 1, 'starters': [
                    {'id': i, 'name': f'Player {i}', 'performance': {'rating': 10}}
                    for i in range(1, 12)]},
                'awayTeam': {'id': 2}}}}


def ea_data():
    return pd.DataFrame([{'player_id': i + 100, 'club': 'Arsenal',
                          'position': position, 'overall_rating': 80,
                          'snapshot_date': '2026-09-12', 'gender': "Men's Football",
                          'first_name': 'Player', 'last_name': str(i)}
                         for i, position in enumerate(['GK'] + ['CB'] * 4 + ['CM'] * 3 + ['ST'] * 3, 1)])


def bridge_data():
    return pd.DataFrame({'fotmob_id': [str(i) for i in range(1, 12)],
                         'ea_id': [str(i + 100) for i in range(1, 12)]})


def forecast():
    return {'fixture_id': 123, 'home_team': 'Arsenal', 'away_team': 'Chelsea',
            'kickoff_utc': KICKOFF, 'home_win_probability': .5,
            'draw_probability': .3, 'away_win_probability': .2,
            'most_likely_score': '1-0', 'expected_home_goals': 1.5}


class LineupClientTests(unittest.TestCase):
    def test_identity_only_and_confirmation_is_not_inferred(self):
        parsed = FotMobClient._parse_lineup(payload(), 123, NOW)
        self.assertEqual(parsed['home']['status'], 'provided_unverified')
        self.assertEqual(parsed['away']['status'], 'unavailable')
        self.assertEqual(parsed['home']['starters'][0], {'player_id': '1', 'name': 'Player 1'})
        self.assertNotIn('rating', json.dumps(parsed))
        self.assertEqual(parsed['fetched_at'], NOW.isoformat())

    def test_missing_projected_and_confirmed_lineups(self):
        d = payload()
        d['content']['lineup']['lineupType'] = 'projected'
        self.assertEqual(FotMobClient._parse_lineup(d, 123, NOW)['home']['status'], 'projected')
        d['content']['lineup']['confirmed'] = True
        self.assertEqual(FotMobClient._parse_lineup(d, 123, NOW)['home']['status'], 'confirmed')
        d['content'] = {}
        self.assertEqual(FotMobClient._parse_lineup(d, 123, NOW)['home']['starters'], [])

    def test_started_cancelled_and_post_kickoff_do_not_expose_starters(self):
        for field in ['started', 'finished', 'cancelled', 'postponed']:
            d = payload()
            d['header'] = {'status': {field: True}}
            parsed = FotMobClient._parse_lineup(d, 123, NOW)
            self.assertEqual(parsed['status'], 'not_prematch')
            self.assertEqual(parsed['home']['starters'], [])
        after = pd.Timestamp(KICKOFF).to_pydatetime()
        self.assertEqual(FotMobClient._parse_lineup(payload(), 123, after)['status'], 'not_prematch')

    def test_mismatched_ids_and_duplicate_starters_rejected(self):
        with self.assertRaises(ValueError): FotMobClient._parse_lineup(payload(), 999, NOW)
        d = payload()
        d['content']['lineup']['homeTeam']['id'] = 8
        with self.assertRaises(ValueError): FotMobClient._parse_lineup(d, 123, NOW)
        d = payload()
        d['content']['lineup']['homeTeam']['starters'][1]['id'] = 1
        with self.assertRaises(ValueError): FotMobClient._parse_lineup(d, 123, NOW)
        d = payload()
        d['general']['gender'] = 'female'
        with self.assertRaises(ValueError): FotMobClient._parse_lineup(d, 123, NOW)

    def test_fetch_uses_timeout_and_validates_id(self):
        client = FotMobClient(timeout=3)
        client.session = Mock()
        client.session.get.return_value.json.return_value = payload()
        parsed = client.fetch_lineup(123)
        self.assertEqual(parsed['fixture_id'], 123)
        self.assertEqual(client.session.get.call_args.kwargs, {'params': {'matchId': 123}, 'timeout': 3})
        for invalid in [None, True, 0, -1, '12.3']:
            with self.assertRaises(ValueError): client.fetch_lineup(invalid)


class StartingTalentTests(unittest.TestCase):
    def setUp(self):
        self.starters = FotMobClient._parse_lineup(payload(), 123, NOW)['home']['starters']

    def test_full_xi_and_same_unit_baseline(self):
        ea = ea_data()
        better = ea.iloc[[1]].copy()
        better['player_id'], better['overall_rating'] = 999, 90
        ea = pd.concat([ea, better], ignore_index=True)
        result = match_starting_xi_talent('Arsenal', self.starters, ea, NOW, bridge_data())
        self.assertEqual(result['xi_overall'], 80.)
        self.assertEqual(result['units']['GK']['matched_players'], 1)
        self.assertEqual(result['units']['DEF']['matched_players'], 4)
        self.assertAlmostEqual(result['rotation_delta'], -10 / 11)
        json.dumps(result, allow_nan=False)

    def test_partial_and_missing_mappings_never_claim_full_xi(self):
        partial = match_starting_xi_talent('Arsenal', self.starters, ea_data(), NOW, bridge_data().iloc[:5])
        self.assertEqual(partial['matched_count'], 5)
        self.assertIsNone(partial['xi_overall'])
        self.assertIsNone(partial['rotation_delta'])
        missing = match_starting_xi_talent('Arsenal', self.starters, ea_data(), NOW)
        self.assertEqual(missing['matched_count'], 0)
        self.assertEqual(missing['unmatched'][0]['candidate_ea_ids'], ['101'])
        self.assertIsNone(missing['xi_overall'])

    def test_ambiguous_names_are_only_candidates_and_duplicate_bridge_fails(self):
        ea = ea_data()
        duplicate = ea.iloc[[0]].copy()
        duplicate['player_id'] = 999
        ea = pd.concat([ea, duplicate])
        result = match_starting_xi_talent('Arsenal', self.starters, ea, NOW)
        self.assertEqual(result['unmatched'][0]['candidate_ea_ids'], ['101', '999'])
        with self.assertRaises(ValueError):
            validate_player_bridge(pd.concat([bridge_data(), bridge_data()]))
        with self.assertRaises(ValueError):
            validate_player_bridge(pd.DataFrame({'ea_id': [None], 'fotmob_id': ['1']}))

    def test_snapshots_transfers_gender_and_missing_ratings(self):
        ea = ea_data()
        self.assertEqual(match_starting_xi_talent('Arsenal', self.starters, ea, '2026-09-12', bridge_data())['matched_count'], 0)
        moved = ea.iloc[[0]].copy()
        moved['club'], moved['snapshot_date'] = 'Chelsea', '2026-10-01'
        result = match_starting_xi_talent('Arsenal', self.starters, pd.concat([ea, moved]), NOW, bridge_data())
        self.assertEqual(result['matched_count'], 10)
        ea.loc[0, 'gender'] = "Women's Football"
        self.assertEqual(match_starting_xi_talent('Arsenal', self.starters, ea, NOW, bridge_data())['matched_count'], 10)
        self.assertEqual(match_starting_xi_talent('Arsenal', self.starters, ea.drop(columns='gender'), NOW, bridge_data())['matched_count'], 0)
        self.assertEqual(match_starting_xi_talent('Arsenal', self.starters, None, NOW)['matched_count'], 0)


class LineupIntegrationTests(unittest.TestCase):
    def client(self):
        client = Mock()
        client.fetch_lineup.return_value = FotMobClient._parse_lineup(payload(), 123, NOW)
        return client

    def test_context_does_not_change_forecast_and_caches_duplicate_requests(self):
        client = self.client()
        initial = forecast()
        result = add_live_lineup_context([forecast(), forecast()], client, ea_data(), bridge_data(), now=NOW)
        self.assertEqual(client.fetch_lineup.call_count, 1)
        for key, value in initial.items():
            self.assertEqual(result[0][key], value)
        context = result[0]['live_lineup_context']
        self.assertFalse(context['used_in_model'])
        self.assertEqual(context['home']['xi_overall'], 80)
        json.dumps(result, allow_nan=False)

    def test_past_kickoff_and_missing_id_skip_network(self):
        client = self.client()
        for record in [dict(forecast(), kickoff_utc=NOW.isoformat()), dict(forecast(), fixture_id=None)]:
            result = add_live_lineup_context([record], client, now=NOW)[0]
            self.assertEqual(result['live_lineup_context']['status'], 'unavailable')
        client.fetch_lineup.assert_not_called()

    def test_mismatched_clubs_and_time_rejected(self):
        for record in [dict(forecast(), home_team='Everton'), dict(forecast(), kickoff_utc='2026-11-22T12:30:00Z')]:
            result = add_live_lineup_context([record], self.client(), now=NOW)[0]
            self.assertEqual(result['live_lineup_context']['status'], 'unavailable')
            self.assertIsNone(result['live_lineup_context']['snapshot'])

    def test_timeout_malformed_and_rate_limit_fallback(self):
        for error in [requests.Timeout('timeout'), ValueError('Bad JSON')]:
            client = self.client()
            client.fetch_lineup.side_effect = error
            result = add_live_lineup_context([forecast()], client, now=NOW)[0]
            self.assertEqual(result['home_win_probability'], .5)
            self.assertEqual(result['live_lineup_context']['status'], 'unavailable')
        response = requests.Response()
        response.status_code = 429
        client = self.client()
        client.fetch_lineup.side_effect = requests.HTTPError(response=response)
        records = [forecast(), dict(forecast(), fixture_id=124)]
        add_live_lineup_context(records, client, now=NOW)
        self.assertEqual(client.fetch_lineup.call_count, 1)
        self.assertIn('stopped', records[1]['live_lineup_context']['reason'])

    def test_default_cli_stays_offline_and_live_mode_is_explicit(self):
        from hybrid_predictor import main
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'forecast.json'
            args = ['hybrid_predictor.py', 'predict', '--home', 'Arsenal', '--away', 'Chelsea',
                    '--date', KICKOFF, '--fixture-id', '123', '--players', str(Path(folder) / 'absent.csv'),
                    '--output', str(output)]
            with patch('epl_predictor.predictor.Predictor') as model, patch('fotmob_client.FotMobClient') as client:
                model.return_value.manifest = {'selected_model': 'dixon_coles'}
                model.return_value.predict.return_value = [forecast()]
                with patch('sys.argv', args): main()
                client.assert_not_called()
                self.assertNotIn('live_lineup_context', json.loads(output.read_text())[0])
                client.return_value.fetch_lineup.return_value = FotMobClient._parse_lineup(payload(), 123, NOW)
                with patch('epl_predictor.lineups.datetime') as clock, patch('sys.argv', args + ['--use-lineups']):
                    clock.now.return_value = NOW
                    main()
                self.assertIn('live_lineup_context', json.loads(output.read_text())[0])
                client.return_value.fetch_lineup.assert_called_once_with(123)
                client.return_value.session.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
