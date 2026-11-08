"""Regression checks for time isolation and probability validity (offline)."""

import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import poisson_distribution as model
from backtest_poisson import PROBABILITY_COLUMNS, run_backtest, validate_matches


def match(day, season, home, away, hg, ag):
    return dict(date=day, season=season, home_team=home, away_team=away,
                home_goals=hg, away_goals=ag, status="FT")


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.pl = pd.DataFrame([
            match("2024-08-01", "2024-25", "A", "B", 2, 1),
            match("2025-05-01", "2024-25", "B", "A", 0, 1),
            match("2025-08-01", "2025-26", "A", "C", 2, 1),
            match("2026-05-01", "2025-26", "C", "A", 1, 1),
        ])
        self.ch = pd.DataFrame([
            match("2024-08-01", "2024-25", "C", "D", 3, 1),
            match("2025-05-01", "2024-25", "D", "C", 1, 2),
            match("2025-08-01", "2025-26", "B", "D", 2, 1),
            match("2026-05-01", "2025-26", "D", "B", 1, 1),
        ])
        self.results = pd.DataFrame([
            match("2026-08-21T12:00:00Z", "2026-27", "A", "B", 1, 0),
            match("2026-08-21T20:00:00Z", "2026-27", "C", "A", 0, 0),
            match("2026-08-22T15:00:00Z", "2026-27", "B", "C", 2, 1),
            match("2026-08-23T15:00:00Z", "2026-27", "C", "B", 1, 2),
        ])
        self.prediction_columns = PROBABILITY_COLUMNS + [
            "expected_home_goals", "expected_away_goals", "most_likely_score",
        ]

    def run_model(self, results):
        return run_backtest(self.pl, self.ch, results)[0]

    def test_target_and_future_results_cannot_change_earlier_predictions(self):
        before = self.run_model(self.results)
        changed = self.results.copy()
        changed.loc[2:, ["home_goals", "away_goals"]] = [8, 7]
        after = self.run_model(changed)
        # Includes both the target match and all earlier fixtures.
        cutoff = pd.Timestamp("2026-08-22T15:00:00Z")
        pd.testing.assert_frame_equal(
            before.loc[before.kickoff_utc <= cutoff, self.prediction_columns],
            after.loc[after.kickoff_utc <= cutoff, self.prediction_columns],
        )

    def test_same_day_results_are_excluded_and_next_day_can_update(self):
        before = self.run_model(self.results)
        changed = self.results.copy()
        changed.loc[0, "home_goals"] = 7
        after = self.run_model(changed)
        mask = before.kickoff_utc < pd.Timestamp("2026-08-22T00:00:00Z")
        pd.testing.assert_frame_equal(before.loc[mask, self.prediction_columns],
                                      after.loc[mask, self.prediction_columns])
        walk = before.loc[before["mode"].eq("walk_forward")]
        self.assertEqual(walk.current_results_used.tolist(), [0, 0, 2, 3])
        self.assertFalse(before.loc[~mask & before["mode"].eq("walk_forward"), self.prediction_columns].equals(
            after.loc[~mask & after["mode"].eq("walk_forward"), self.prediction_columns]))

    def test_historical_only_never_uses_any_test_results(self):
        before = self.run_model(self.results)
        changed = self.results.copy()
        changed["home_goals"], changed["away_goals"] = 6, 4
        after = self.run_model(changed)
        mask = before["mode"].eq("historical_only")
        pd.testing.assert_frame_equal(before.loc[mask, self.prediction_columns],
                                      after.loc[mask, self.prediction_columns])
        self.assertTrue(before.loc[mask, "current_results_used"].eq(0).all())
        self.assertTrue(before.loc[mask, "current_promotion_weight"].eq(0).all())

    def test_opening_day_with_no_current_data_matches_historical_mode(self):
        predicted = self.run_model(self.results)
        for mode in ["historical_only", "walk_forward"]:
            selected = predicted.loc[predicted["mode"].eq(mode), self.prediction_columns].head(2).reset_index(drop=True)
            if mode == "historical_only":
                expected = selected
            else:
                pd.testing.assert_frame_equal(expected, selected)

    def test_probability_mass_and_high_scoring_mode(self):
        for home, away in [(0, 0), (1.5, 1.5), (9.2, 8.3), (100, 100)]:
            h, d, a, score = model.poisson_probabilities(home, away)
            self.assertAlmostEqual(h + d + a, 1, places=10)
            if home == away:
                self.assertAlmostEqual(h, a, places=10)
        self.assertEqual(model.poisson_probabilities(9.2, 8.3)[3], "9-8")
        self.assertEqual(model.poisson_probabilities(0, 0), (0, 1, 0, "0-0"))
        for rate in [-1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                model.poisson_probabilities(rate, 1)

    def test_duplicate_or_unfinished_results_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_matches(pd.concat([self.results, self.results.iloc[:1]]), "Test")
        changed = self.results.copy()
        changed.loc[0, "status"] = "NS"
        with self.assertRaisesRegex(ValueError, "completed"):
            validate_matches(changed, "Test")

    def test_training_season_overlap_rejected(self):
        self.pl.loc[3, "date"] = "2026-08-22"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_model(self.results)

    def test_omit_flag_does_not_fetch_or_read_current_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            pl_path, ch_path = folder / "pl.csv", folder / "ch.csv"
            fixtures_path = folder / "fixtures.csv"
            self.pl.to_csv(pl_path, index=False)
            self.ch.to_csv(ch_path, index=False)
            self.results.iloc[:1].rename(columns={"date": "kickoff_utc"}).to_csv(fixtures_path, index=False)
            args = argparse.Namespace(omit_current_season=True, no_refresh_current=False,
                                      fixtures=fixtures_path, output=folder / "predictions.csv")
            with patch.object(model, "parse_arguments", return_value=args), \
                 patch.object(model, "PREMIER_LEAGUE_PATH", pl_path), \
                 patch.object(model, "CHAMPIONSHIP_PATH", ch_path), \
                 patch.object(model, "CURRENT_RESULTS_PATH", folder / "nonexistent.csv"), \
                 patch.object(model, "ADJUSTMENT_PATH", folder / "adjustment.json"), \
                 patch.object(model, "refresh_current_results") as refresh:
                model.main()
                refresh.assert_not_called()
                self.assertEqual(len(pd.read_csv(args.output)), 1)


if __name__ == "__main__":
    unittest.main()
