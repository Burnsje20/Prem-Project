import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from epl_predictor.player_review import create_review, export_reviewed_bridge
from epl_predictor.prospective import (
    archive_predictions, evaluate_archive, load_archive, source_hashes, write_json_new,
)
from tests.test_lineups import NOW, KICKOFF, forecast, payload, ea_data
from fotmob_client import FotMobClient
from epl_predictor.lineups import add_live_lineup_context
from unittest.mock import Mock


def record():
    return dict(forecast(), model='dixon_coles', data_cutoff='2026-11-09T00:00:00Z',
                trained_at='2026-11-09T00:00:00Z')


def results():
    return pd.DataFrame([{'fixture_id': 123, 'date': KICKOFF, 'season': '2026-27',
                          'home_team': 'Arsenal', 'away_team': 'Chelsea',
                          'home_goals': 2, 'away_goals': 0, 'status': 'FT'}])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def capture(self, rows=None, when=NOW, inputs=None):
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = when
            return archive_predictions(rows if rows is not None else [record()], self.folder, inputs or {})

    def test_roundtrip_preserves_predictions_and_never_overwrites(self):
        original = record()
        first = self.capture([original])
        second = self.capture([original])
        self.assertNotEqual(first, second)
        self.assertEqual(load_archive(first)['predictions'], [original])
        self.assertEqual(load_archive(first)['captured_at'], NOW.isoformat())
        with self.assertRaises(FileExistsError): write_json_new(first, {})
        self.assertEqual(load_archive(first)['predictions'], [original])

    def test_tampered_payload_rejected(self):
        path = self.capture()
        archive = json.loads(path.read_text())
        archive['predictions'][0]['home_win_probability'] = .99
        path.write_text(json.dumps(archive))
        with self.assertRaisesRegex(ValueError, 'checksum'): load_archive(path)

    def test_past_kickoff_future_cutoff_training_and_naive_dates_rejected(self):
        for changes in [
            {'kickoff_utc': NOW.isoformat()}, {'data_cutoff': '2026-12-01T00:00:00Z'},
            {'trained_at': '2026-12-01T00:00:00Z'}, {'kickoff_utc': '2026-11-21 12:30'},
            {'fixture_id': None}, {'fixture_id': 1.5}, {'home_win_probability': .9},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.capture([dict(record(), **changes)])
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_empty_batches_duplicate_ids_and_future_lineups_rejected(self):
        with self.assertRaises(ValueError): self.capture([])
        with self.assertRaises(ValueError): self.capture([record(), record()])
        with self.assertRaises(ValueError): self.capture([record(), dict(record(), fixture_id='00123')])
        row = record()
        snapshot = FotMobClient._parse_lineup(payload(), 123, NOW + timedelta(minutes=5))
        row['live_lineup_context'] = {'snapshot': snapshot}
        with self.assertRaisesRegex(ValueError, 'capture time'): self.capture([row])

    def test_hashes_recorded_and_source_changes_detected(self):
        source = self.folder / 'input.txt'
        source.write_text('original')
        inputs = source_hashes([source])
        path = self.capture(inputs=inputs)
        self.assertEqual(load_archive(path)['source_hashes'], inputs)
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'): self.capture(inputs=inputs)

    def test_latest_capture_selected_once_and_scores_correctly(self):
        self.capture()
        latest = dict(record(), home_win_probability=.7, draw_probability=.2, away_win_probability=.1)
        second = self.capture([latest], NOW + timedelta(hours=1))
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = NOW + timedelta(days=1)
            report = evaluate_archive(self.folder, results())
        self.assertEqual(report['forecast_count'], 1)
        self.assertEqual(report['metrics']['matches'], 1)
        self.assertEqual(report['metrics']['accuracy'], 1.)
        self.assertAlmostEqual(report['metrics']['brier'], .14)
        self.assertEqual(report['scored'][0]['archive'], str(second))
        self.assertEqual(report['scored'][0]['probabilities'], [.7, .2, .1])

    def test_result_changes_do_not_change_forecast_selection(self):
        self.capture()
        second = self.capture(when=NOW + timedelta(minutes=1))
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = NOW + timedelta(days=1)
            altered = results().assign(home_goals=0, away_goals=5)
            report = evaluate_archive(self.folder, altered)
        self.assertEqual(report['scored'][0]['archive'], str(second))
        self.assertEqual(report['scored'][0]['target'], 2)

    def test_same_day_results_stay_pending_and_schedule_changes_flagged(self):
        self.capture()
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = NOW + timedelta(hours=8)
            report = evaluate_archive(self.folder, results())
            self.assertIsNone(report['metrics'])
            self.assertEqual(report['pending_fixture_ids'], ['123'])
            clock.now.return_value = NOW + timedelta(days=2)
            report = evaluate_archive(self.folder, results().assign(date='2026-11-21T15:00:00Z'))
            self.assertEqual(report['mismatched_fixture_ids'], ['123'])
            self.assertEqual(report['scored'], [])

    def test_missing_archives_and_unfinished_results_fail_explicitly(self):
        with self.assertRaises(ValueError): evaluate_archive(self.folder, results())
        self.capture()
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = NOW + timedelta(days=1)
            with self.assertRaisesRegex(ValueError, 'completed'):
                evaluate_archive(self.folder, results().assign(status='Scheduled'))


class PlayerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        client = Mock()
        client.fetch_lineup.return_value = FotMobClient._parse_lineup(payload(), 123, NOW)
        self.predictions = add_live_lineup_context([record()], client, ea_data(), now=NOW)
        self.source = write_json_new(self.folder / 'predictions.json', self.predictions)
        self.review = self.folder / 'review.json'
        self.bridge = self.folder / 'bridge.csv'

    def approve(self, data, index=0, ea_id='101'):
        data['decisions'][index].update(approved=True, ea_id=ea_id, review_note='Verified both provider profiles')
        self.review.write_text(json.dumps(data), encoding='utf-8')

    def test_candidates_remain_unapproved_and_deduplicated(self):
        create_review([self.source, self.source], self.review)
        data = json.loads(self.review.read_text())
        self.assertEqual(len(data['decisions']), 11)
        self.assertEqual(data['decisions'][0]['candidate_ea_ids'], ['101'])
        self.assertTrue(all(r['approved'] is False and r['ea_id'] is None for r in data['decisions']))
        with self.assertRaisesRegex(ValueError, 'No approved'): export_reviewed_bridge(self.review, self.bridge)
        self.assertFalse(self.bridge.exists())

    def test_only_approved_identities_export_and_files_are_preserved(self):
        create_review([self.source], self.review)
        self.approve(json.loads(self.review.read_text()))
        self.assertEqual(export_reviewed_bridge(self.review, self.bridge), 1)
        self.assertEqual(pd.read_csv(self.bridge,dtype=str).to_dict('records'), [{'ea_id': '101', 'fotmob_id': '1'}])
        with self.assertRaises(FileExistsError): create_review([self.source], self.review)
        with self.assertRaises(FileExistsError): export_reviewed_bridge(self.review, self.bridge)

    def test_duplicate_mapping_and_missing_review_evidence_rejected(self):
        create_review([self.source], self.review)
        base = json.loads(self.review.read_text())
        self.approve(base)
        self.approve(base, index=1, ea_id='101')
        with self.assertRaisesRegex(ValueError, 'one-to-one'): export_reviewed_bridge(self.review, self.bridge)
        for approved, note in [('true', 'Checked'), (True, '')]:
            data = copy.deepcopy(base)
            data['decisions'][0].update(approved=approved, review_note=note)
            self.review.write_text(json.dumps(data))
            with self.assertRaises(ValueError): export_reviewed_bridge(self.review, self.bridge)
        self.assertFalse(self.bridge.exists())

    def test_archives_supported_and_missing_lineups_yield_no_fake_players(self):
        with patch('epl_predictor.prospective.datetime') as clock:
            clock.now.return_value = NOW
            archive = archive_predictions(self.predictions, self.folder, {})
        create_review([archive], self.review)
        self.assertEqual(len(json.loads(self.review.read_text())['decisions']), 11)
        empty_source = write_json_new(self.folder / 'empty.json', [record()])
        output = self.folder / 'empty_review.json'
        create_review([empty_source], output)
        self.assertEqual(json.loads(output.read_text())['decisions'], [])

    def test_review_and_export_cli(self):
        from hybrid_predictor import main
        with patch('sys.argv', ['hybrid_predictor.py', 'review-players', '--predictions', str(self.source),
                                '--output', str(self.review)]):
            main()
        self.approve(json.loads(self.review.read_text()))
        with patch('sys.argv', ['hybrid_predictor.py', 'export-player-map', '--review', str(self.review),
                                '--output', str(self.bridge)]):
            main()
        self.assertEqual(len(pd.read_csv(self.bridge)), 1)


if __name__ == '__main__':
    unittest.main()
