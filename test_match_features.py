"""Run with: python3 -m unittest -v test_match_features"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from match_features import MATCH_COLUMNS, MatchFeatureBuilder


class MatchFeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        # Alpha alternates home/away. Dates cross a season boundary.
        self.matches = pd.DataFrame({
            "Date": pd.date_range("2025-06-01", periods=7, freq="7D"),
            "Time": ["15:00"] * 7,
            "HomeTeam": ["Alpha", "Beta", "Alpha", "Beta", "Alpha", "Beta", "Alpha"],
            "AwayTeam": ["Beta", "Alpha", "Beta", "Alpha", "Beta", "Alpha", "Beta"],
            "FTHG": [2, 1, 0, 0, 1, 2, 0],
            "FTAG": [0, 1, 1, 3, 1, 1, 0],
        })
        self.builder = MatchFeatureBuilder()

    def test_known_averages_include_both_venues_and_cross_seasons(self):
        result = self.builder.add_features(self.matches)
        self.assertEqual(len(result), 2)
        # Sixth match: Alpha's preceding points are [3, 1, 0, 3, 1].
        self.assertAlmostEqual(result.iloc[0]["away_points_last5"], 8 / 5)
        self.assertAlmostEqual(result.iloc[0]["away_goals_scored_last5"], 7 / 5)
        self.assertAlmostEqual(result.iloc[0]["away_goals_conceded_last5"], 3 / 5)
        self.assertAlmostEqual(result.iloc[0]["points_difference_last5"], -3 / 5)
        self.assertAlmostEqual(result.iloc[0]["goals_conceded_difference_last5"], 4 / 5)
        # Seventh match: the 20-game window includes all six earlier games.
        self.assertAlmostEqual(result.iloc[1]["home_points_last5"], 1)
        self.assertAlmostEqual(result.iloc[1]["home_points_last20"], 8 / 6)
        self.assertAlmostEqual(result.iloc[1]["home_goals_scored_last20"], 8 / 6)

    def test_current_and_future_scores_cannot_change_prior_features(self):
        original = self.builder.add_features(self.matches, drop_incomplete=False)
        changed = self.matches.copy()
        changed.loc[5:, "FTHG"] += 10
        updated = self.builder.add_features(changed, drop_incomplete=False)
        pd.testing.assert_frame_equal(
            original.loc[:5, self.builder.feature_columns],
            updated.loc[:5, self.builder.feature_columns],
        )
        self.assertNotEqual(
            original.iloc[6]["home_goals_conceded_last5"],
            updated.iloc[6]["home_goals_conceded_last5"],
        )

    def test_shuffled_input_is_sorted_and_input_is_not_changed(self):
        shuffled = self.matches.sample(frac=1, random_state=42)
        original = shuffled.copy(deep=True)
        pd.testing.assert_frame_equal(
            self.builder.add_features(shuffled), self.builder.add_features(self.matches)
        )
        pd.testing.assert_frame_equal(shuffled, original)

    def test_insufficient_history_is_optional_and_early_results_are_used(self):
        result = self.builder.add_features(self.matches, drop_incomplete=False)
        self.assertTrue(result.loc[:4, self.builder.feature_columns].isna().all().all())
        self.assertTrue(result.loc[5:, self.builder.feature_columns].notna().all().all())

    def test_configurable_windows(self):
        builder = MatchFeatureBuilder(windows=(2, 3), min_matches=2)
        result = builder.add_features(self.matches)
        self.assertEqual(len(builder.feature_columns), 18)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result.iloc[0]["home_points_last2"], 2)
        for kwargs in [{"windows": (2,)}, {"windows": (5, 5)}, {"min_matches": 0}]:
            with self.assertRaises(ValueError):
                MatchFeatureBuilder(**kwargs)

    def test_missing_results_are_rejected(self):
        incomplete = self.matches.copy()
        incomplete.loc[0, "FTHG"] = np.nan
        with self.assertRaisesRegex(ValueError, "completed matches"):
            self.builder.add_features(incomplete)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.builder.add_features(self.matches.drop(columns="HomeTeam"))

    def test_csv_loading_keeps_only_requested_odds_and_handles_bom(self):
        matches = self.matches.copy()
        for column in MATCH_COLUMNS:
            if column not in matches:
                matches[column] = 0
        matches["AvgH"] = 2.5
        matches["B365H"] = 2.6
        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory) / "a.csv", Path(directory) / "b.csv"]
            matches.iloc[:3].to_csv(files[0], index=False, encoding="utf-8-sig")
            matches.iloc[3:].to_csv(files[1], index=False)
            loaded = self.builder.load_csvs(files, extra_columns=["AvgH"])
        self.assertEqual(len(loaded), 7)
        self.assertIn("AvgH", loaded)
        self.assertNotIn("B365H", loaded)
        self.assertNotIn("AvgH", self.builder.feature_columns)
        featured = self.builder.add_features(loaded)
        self.assertTrue(featured["AvgH"].eq(2.5).all())
        self.assertEqual(featured["Date"].tolist(), self.matches["Date"].iloc[5:].tolist())


if __name__ == "__main__":
    unittest.main()
