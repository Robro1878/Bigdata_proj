"""Shared pre-match features for the Premier League model notebooks."""

import numpy as np
import pandas as pd


MATCH_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]


class MatchFeatureBuilder:
    """Build rolling averages and home-minus-away differences from match results.

    Call add_features on the full match history BEFORE splitting by date. Each
    row uses only earlier results, including earlier games in the test season.
    History includes home and away games and carries across season boundaries.
    This evaluates one-match-at-a-time predictions; it is not a forecast of an
    entire season before any of that season's results are known.

    Default windows (5, 20) produce the same 18 features, in the same order, as
    the original project notebook. Both windows require at least five matches.
    The builder does not learn scaling or model parameters.
    """

    stats = ("points", "goals_scored", "goals_conceded")

    def __init__(self, windows=(5, 20), min_matches=5):
        self.windows = tuple(windows)
        if not isinstance(min_matches, int) or isinstance(min_matches, bool) or min_matches < 1:
            raise ValueError("min_matches must be a positive integer.")
        if not self.windows or any(
            not isinstance(window, int) or isinstance(window, bool) or window < min_matches
            for window in self.windows
        ):
            raise ValueError("Each window must be an integer at least as large as min_matches.")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("Windows must be unique.")
        self.min_matches = min_matches
        self.feature_columns = [
            f"{side}_{stat}_last{window}"
            for side in ("home", "away")
            for window in self.windows
            for stat in self.stats
        ] + [
            f"{stat}_difference_last{window}"
            for window in self.windows
            for stat in self.stats
        ]

    def load_csvs(self, csv_files, extra_columns=None):
        """Combine season CSVs; optionally retain extra columns such as odds.

        Files must contain the standard MATCH_COLUMNS. Paths are relative to
        the notebook's working directory unless absolute paths are supplied.
        """
        columns = list(dict.fromkeys(MATCH_COLUMNS + list(extra_columns or [])))
        return pd.concat(
            [pd.read_csv(file, encoding="utf-8-sig", usecols=columns) for file in csv_files],
            ignore_index=True,
        )

    def add_features(self, matches, drop_incomplete=True):
        """Return a chronological copy with form features added.

        Input must contain completed matches. Extra columns are preserved but
        never become model inputs automatically. Set drop_incomplete=False to
        inspect rows whose teams have not yet accumulated enough history.
        Reuse feature_columns to select the same inputs in every model notebook.
        """
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing = [column for column in required if column not in matches.columns]
        if missing:
            raise ValueError(f"Missing required match columns: {missing}")

        df = matches.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            dates = df["Date"].astype("string")
            iso_dates = dates.str.match(r"^\d{4}-\d{2}-\d{2}").fillna(False)
            df["Date"] = pd.to_datetime(dates, format="mixed", dayfirst=True)
            # CSV exports may use ISO dates; preserve their year-month-day order.
            if iso_dates.any():
                df.loc[iso_dates, "Date"] = pd.to_datetime(dates.loc[iso_dates], format="ISO8601")
        if df[["Date", "HomeTeam", "AwayTeam"]].isna().any().any():
            raise ValueError("Every match needs a date and both team names.")
        goals = df[["FTHG", "FTAG"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(goals.to_numpy()).all() or (goals < 0).any().any() or (goals % 1 != 0).any().any():
            raise ValueError("Features require completed matches with nonnegative integer goals.")
        df[["FTHG", "FTAG"]] = goals
        sort_columns = ["Date", "Time"] if "Time" in df.columns else ["Date"]
        df = df.sort_values(sort_columns).reset_index(drop=True)

        # Make one history row for each team, combining its home and away games.
        home_matches = df[["HomeTeam", "FTHG", "FTAG"]].rename(columns={
            "HomeTeam": "team", "FTHG": "goals_scored", "FTAG": "goals_conceded",
        }).assign(match_id=df.index, side="home")
        away_matches = df[["AwayTeam", "FTAG", "FTHG"]].rename(columns={
            "AwayTeam": "team", "FTAG": "goals_scored", "FTHG": "goals_conceded",
        }).assign(match_id=df.index, side="away")
        team_matches = pd.concat([home_matches, away_matches], ignore_index=True)
        team_matches = team_matches.sort_values(["team", "match_id"])
        team_matches["points"] = np.select(
            [
                team_matches["goals_scored"] > team_matches["goals_conceded"],
                team_matches["goals_scored"] == team_matches["goals_conceded"],
            ],
            [3, 1],
            default=0,
        )

        rolling_columns = []
        for window in self.windows:
            columns = [f"{stat}_last{window}" for stat in self.stats]
            # Exclude the current match before calculating either window.
            team_matches[columns] = team_matches.groupby("team")[list(self.stats)].transform(
                lambda history: history.shift(1).rolling(window, min_periods=self.min_matches).mean()
            )
            rolling_columns.extend(columns)

        for side in ("home", "away"):
            form = team_matches.loc[team_matches["side"] == side].set_index("match_id")
            for column in rolling_columns:
                df[f"{side}_{column}"] = form[column]

        for window in self.windows:
            for stat in self.stats:
                df[f"{stat}_difference_last{window}"] = (
                    df[f"home_{stat}_last{window}"] - df[f"away_{stat}_last{window}"]
                )

        # Build all histories first so excluded early games still inform later games.
        if drop_incomplete:
            df = df.dropna(subset=self.feature_columns).reset_index(drop=True)
        return df
