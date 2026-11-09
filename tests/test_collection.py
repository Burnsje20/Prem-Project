from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from epl_predictor.collection import collect_lineups, recent_lineup_captures, select_collection_fixtures
from epl_predictor.prospective import archive_predictions, load_archive
from fotmob_client import FotMobClient
from tests.test_lineups import NOW, KICKOFF, payload, ea_data
from tests.test_prospective import record


def fixtures():
    return pd.DataFrame([{'fixture_id': '123', 'home_team': 'Arsenal', 'away_team': 'Chelsea',
                          'kickoff_utc': KICKOFF, 'status': 'Scheduled', 'source': 'fotmob'}])


class CollectionSelectionTests(unittest.TestCase):
    def test_window_boundary_is_inclusive_but_kickoff_is_not(self):
        due, skipped, next_at = select_collection_fixtures(fixtures(), NOW)
        self.assertEqual(due.fixture_id.tolist(), ['123'])
        self.assertEqual(skipped, [])
        self.assertIsNone(next_at)
        due, skipped, next_at = select_collection_fixtures(fixtures(), NOW - timedelta(seconds=1))
        self.assertTrue(due.empty)
        self.assertEqual(skipped[0]['reason'], 'outside_window')
        self.assertEqual(pd.Timestamp(next_at), NOW)
        due, skipped, _ = select_collection_fixtures(fixtures(), pd.Timestamp(KICKOFF))
        self.assertTrue(due.empty)
        self.assertEqual(skipped[0]['reason'], 'kickoff_passed')

    def test_cooldown_and_non_scheduled_status(self):
        due, skipped, next_at = select_collection_fixtures(fixtures(), NOW, recent={'123': NOW})
        self.assertTrue(due.empty)
        self.assertEqual(skipped[0]['reason'], 'refresh_cooldown')
        self.assertEqual(pd.Timestamp(next_at), NOW + timedelta(minutes=10))
        due, _, _ = select_collection_fixtures(fixtures(), NOW + timedelta(minutes=10), recent={'123': NOW})
        self.assertEqual(len(due), 1)
        for status in ['Cancelled', 'Postponed', 'FT', 'Unknown']:
            due, skipped, next_at = select_collection_fixtures(fixtures().assign(status=status), NOW)
            self.assertTrue(due.empty)
            self.assertEqual(skipped[0]['reason'], 'not_scheduled')
            self.assertIsNone(next_at)

    def test_unsupported_source_invalid_schema_and_ids(self):
        due, skipped, _ = select_collection_fixtures(fixtures().assign(source='other'), NOW)
        self.assertTrue(due.empty)
        self.assertEqual(skipped[0]['reason'], 'unsupported_source')
        for data in [fixtures().drop(columns='fixture_id'), pd.concat([fixtures(), fixtures()]),
                     fixtures().assign(kickoff_utc='2026-11-21'), fixtures().assign(fixture_id='bad'),
                     fixtures().assign(away_team='Arsenal')]:
            with self.assertRaises(ValueError): select_collection_fixtures(data, NOW)
        for window, refresh in [(0, 10), (1441, 10), (90, 0), (90, 91)]:
            with self.assertRaises(ValueError):
                select_collection_fixtures(fixtures(), NOW, window, refresh)


class CollectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures_path = self.root / 'fixtures.csv'
        fixtures().to_csv(self.fixtures_path, index=False)
        self.players_path = self.root / 'players.csv'
        ea_data().to_csv(self.players_path, index=False)
        self.archives = self.root / 'archives'
        self.output = self.root / 'collection'
        self.options = dict(fixtures_path=self.fixtures_path, artifact_dir=self.root / 'models',
                            players_path=self.players_path, archive_dir=self.archives, output_dir=self.output)
        self.model_patch = patch('epl_predictor.collection.Predictor')
        self.model = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.model.return_value.predict.side_effect = lambda *a, **kw: [record()]
        self.client_patch = patch('epl_predictor.collection.FotMobClient')
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.client.return_value.fetch_lineup.return_value = FotMobClient._parse_lineup(payload(), 123, NOW)
        for target in ['epl_predictor.collection.datetime', 'epl_predictor.prospective.datetime', 'epl_predictor.lineups.datetime']:
            patcher = patch(target)
            clock = patcher.start()
            self.addCleanup(patcher.stop)
            clock.now.return_value = NOW
            if target == 'epl_predictor.collection.datetime':
                self.clock = clock

    def test_collect_archive_review_and_second_run_cooldown(self):
        report, path = collect_lineups(**self.options)
        self.assertEqual(report['status'], 'collected')
        self.assertEqual(report['archived_count'], 1)
        self.assertEqual(report['lineups_available'], 1)
        self.assertEqual(report['unresolved_players'], 11)
        self.assertTrue(path.exists())
        archive = load_archive(report['archive'])
        self.assertEqual(archive['predictions'][0]['home_win_probability'], .5)
        decisions = json.loads(Path(report['review']).read_text())['decisions']
        self.assertTrue(all(d['approved'] is False for d in decisions))
        self.client.return_value.session.close.assert_called_once()
        self.client.return_value.fetch_lineup.assert_called_once_with(123)
        second, second_path = collect_lineups(**self.options)
        self.assertEqual(second['status'], 'idle')
        self.assertEqual(second['skipped'][0]['reason'], 'refresh_cooldown')
        self.assertNotEqual(path, second_path)
        self.assertEqual(self.client.call_count, 1)

    def test_outside_window_does_not_load_model_or_use_network(self):
        self.clock.now.return_value = NOW - timedelta(days=1)
        report, _ = collect_lineups(**self.options)
        self.assertEqual(report['status'], 'idle')
        self.assertIsNone(report['archive'])
        self.model.assert_not_called()
        self.client.assert_not_called()

    def test_provider_timeout_is_archived_as_missing_not_invented(self):
        self.client.return_value.fetch_lineup.side_effect = requests.Timeout('timeout')
        report, _ = collect_lineups(**self.options)
        self.assertEqual(report['archived_count'], 1)
        self.assertEqual(report['lineups_available'], 0)
        self.assertEqual(report['unresolved_players'], 0)
        self.assertIn('Timeout', report['lineup_statuses'][0]['reason'])
        self.assertEqual(json.loads(Path(report['review']).read_text())['decisions'], [])

    def test_crossing_kickoff_is_not_backdated(self):
        later = pd.Timestamp(KICKOFF).to_pydatetime() + timedelta(seconds=1)
        self.clock.now.side_effect = [NOW, later, later]
        report, _ = collect_lineups(**self.options)
        self.assertEqual(report['status'], 'expired_during_collection')
        self.assertEqual(report['archived_count'], 0)
        self.assertEqual(report['skipped'][-1]['reason'], 'kickoff_passed_during_collection')
        self.assertFalse(self.archives.exists())

    def test_offline_archives_do_not_trigger_lineup_cooldown(self):
        archive_predictions([record()], self.archives, {})
        self.assertEqual(recent_lineup_captures(self.archives, NOW), {})
        report, _ = collect_lineups(**self.options)
        self.assertEqual(report['archived_count'], 1)

    def test_invalid_reviewed_map_fails_before_network(self):
        mapping = self.root / 'invalid.csv'
        pd.DataFrame({'ea_id': ['101', '101'], 'fotmob_id': ['1', '2']}).to_csv(mapping, index=False)
        with self.assertRaises(ValueError): collect_lineups(**self.options, player_map_path=mapping)
        self.client.assert_not_called()

    def test_cli_idle_path(self):
        from hybrid_predictor import main
        self.clock.now.return_value = NOW - timedelta(days=1)
        with patch('sys.argv', ['hybrid_predictor.py', 'collect-lineups', '--fixtures', str(self.fixtures_path),
                               '--output-dir', str(self.output), '--archive-dir', str(self.archives),
                               '--players', str(self.players_path)]):
            main()
        self.assertEqual(len(list(self.output.glob('collection_*.json'))), 1)
        self.client.assert_not_called()

    def test_concurrent_collection_locking(self):
        lock_file = self.output / '.collector.lock'
        self.output.mkdir(parents=True, exist_ok=True)
        with lock_file.open('x', encoding='utf-8') as handle:
            json.dump({'pid': 99999, 'locked_at': NOW.isoformat()}, handle)
        import os
        os.utime(lock_file, (NOW.timestamp(), NOW.timestamp()))

        report, path = collect_lineups(**self.options)
        self.assertEqual(report['status'], 'locked')
        self.assertIn('Concurrent collection in progress', report['reason'])
        self.assertEqual(report['lock_info']['pid'], 99999)
        self.assertTrue(path.exists())
        self.client.assert_not_called()

    def test_stale_lock_is_cleared(self):
        lock_file = self.output / '.collector.lock'
        self.output.mkdir(parents=True, exist_ok=True)
        with lock_file.open('x', encoding='utf-8') as handle:
            json.dump({'pid': 99999, 'locked_at': (NOW - timedelta(minutes=30)).isoformat()}, handle)
        # Set mtime to 30 minutes in the past
        stale_time = (NOW - timedelta(minutes=30)).timestamp()
        import os
        os.utime(lock_file, (stale_time, stale_time))

        report, path = collect_lineups(**self.options)
        self.assertEqual(report['status'], 'collected')
        self.assertTrue(path.exists())

    def test_fotmob_client_retry_adapter(self):
        client = FotMobClient(max_retries=3, backoff_factor=0.5)
        self.assertIn('https://', client.session.adapters)
        adapter = client.session.adapters['https://']
        self.assertIsNotNone(adapter.max_retries)
        self.assertEqual(adapter.max_retries.total, 3)
        self.assertEqual(adapter.max_retries.backoff_factor, 0.5)
        self.assertEqual(adapter.max_retries.status_forcelist, [429, 500, 502, 503, 504])
        client.session.close()

    def test_collect_with_starter_form_adjustment(self):
        from tests.test_lineups import appearances_data
        apps_path = self.root / 'appearances.csv'
        appearances_data().to_csv(apps_path, index=False)
        self.model.return_value.dc.rho = -0.05
        report, path = collect_lineups(**self.options, appearances_path=apps_path, adjust_probabilities=True)
        self.assertEqual(report['status'], 'collected')
        self.assertTrue(report['adjust_with_lineup_form'])
        self.assertIn('FotMob starter form adjustment applied', report['forecast_policy'])
        archive = load_archive(report['archive'])
        saved = archive['predictions'][0]
        self.assertIn('starter_form_adjustment', saved)
        self.assertTrue(saved['starter_form_adjustment']['applied'])
        self.assertIn('baseline_forecast', saved)


if __name__ == '__main__':
    unittest.main()
