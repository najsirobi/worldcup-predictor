#!/usr/bin/env python3
"""Controlled Phase 5 model comparison for player/coach feature lift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.backtest import realized_points_odds1, scoreline_metrics, time_split, most_probable_scores
from src.evaluation.metrics import all_wdl_metrics, calibration_table
from src.ingest.rules_and_scoring import load_scoring_rules
from src.models.baselines import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSES,
    NUMERIC_FEATURES,
    POISSON_AWAY_FEATURES,
    POISSON_HOME_FEATURES,
    PoissonScoreModel,
    make_hgb_pipeline,
    make_logit_pipeline,
    proba_in_class_order,
)
from src.models.baselines import wdl_from_matrix

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_plus.parquet"
REPORT = ROOT / "outputs" / "reports" / "player_coach_feature_lift_report.md"

PLUS_NUMERIC = [
    "squad_player_count_diff",
    "squad_avg_age_diff",
    "squad_median_age_diff",
    "squad_age_std_diff",
    "players_with_position_diff",
    "players_with_age_diff",
    "attackers_identified_diff",
    "coach_tenure_days_diff",
    "coach_matches_before_match_diff",
    "coach_winrate_before_match_diff",
    "coach_goal_diff_per_match_before_match_diff",
]
PLUS_BINARY = ["has_squad_features", "has_attacker_features", "has_wc2026_squad_features", "has_coach_features"]
ALL_BASELINE = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
ALL_PLUS = NUMERIC_FEATURES + PLUS_NUMERIC + CATEGORICAL_FEATURES + BINARY_FEATURES + PLUS_BINARY
PLUS_POISSON_HOME = POISSON_HOME_FEATURES + [
    "home_squad_squad_player_count",
    "home_squad_squad_avg_age",
    "home_squad_attackers_identified",
    "home_coach_coach_tenure_days",
    "home_coach_coach_matches_before_match",
    "home_coach_coach_winrate_before_match",
]
PLUS_POISSON_AWAY = POISSON_AWAY_FEATURES + [
    "away_squad_squad_player_count",
    "away_squad_squad_avg_age",
    "away_squad_attackers_identified",
    "away_coach_coach_tenure_days",
    "away_coach_coach_matches_before_match",
    "away_coach_coach_winrate_before_match",
]


def _preprocessor(numeric: list[str], categorical: list[str], binary: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("bin", "passthrough", binary), ("cat", cat, categorical)])


def make_plus_logit() -> Pipeline:
    return Pipeline([("pre", _preprocessor(NUMERIC_FEATURES + PLUS_NUMERIC, CATEGORICAL_FEATURES, BINARY_FEATURES + PLUS_BINARY)), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


class PlusPoissonModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.home_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("glm", PoissonRegressor(alpha=1e-3, max_iter=500))])
        self.away_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("glm", PoissonRegressor(alpha=1e-3, max_iter=500))])
        self.base = PoissonScoreModel(max_goals=max_goals)

    def fit(self, df: pd.DataFrame):
        self.home_pipe.fit(df[PLUS_POISSON_HOME], df["home_goals"])
        self.away_pipe.fit(df[PLUS_POISSON_AWAY], df["away_goals"])
        return self

    def predict_lambdas(self, df: pd.DataFrame):
        return (
            np.clip(self.home_pipe.predict(df[PLUS_POISSON_HOME]), 1e-3, 12),
            np.clip(self.away_pipe.predict(df[PLUS_POISSON_AWAY]), 1e-3, 12),
        )

    def score_matrix(self, lam_home: float, lam_away: float):
        return self.base.score_matrix(lam_home, lam_away)

    def predict_proba(self, df: pd.DataFrame):
        lh, la = self.predict_lambdas(df)
        out = np.zeros((len(df), 3))
        for idx in range(len(df)):
            out[idx] = wdl_from_matrix(self.score_matrix(lh[idx], la[idx]))
        return out


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in PLUS_BINARY:
        df[col] = df[col].fillna(False).astype(int)
    return df[(df["match_year"] >= 2000) & df["elo_diff"].notna()].copy()


def _high_conf_error_rate(y_true, proba, threshold=0.70):
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    mask = conf >= threshold
    if not mask.any():
        return np.nan
    return float(round((pred[mask] != np.array(y_true)[mask]).mean(), 4))


def _canonical_scores(proba):
    canonical = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}
    return [canonical[CLASSES[idx]] for idx in proba.argmax(axis=1)]


def _evaluate(label: str, train_end: str, test_mask: pd.Series, df: pd.DataFrame, rules: dict) -> list[dict]:
    train, test = time_split(df, train_end, test_mask)
    models = {
        "phase45_logit": make_logit_pipeline().fit(train[ALL_BASELINE], train["result_label"]),
        "phase45_hgb": make_hgb_pipeline().fit(train[ALL_BASELINE], train["result_label"]),
        "phase45_poisson": PoissonScoreModel().fit(train),
        "plus_logit": make_plus_logit().fit(train[ALL_PLUS], train["result_label"]),
        "plus_poisson": PlusPoissonModel().fit(train),
    }
    probs = {
        "phase45_logit": proba_in_class_order(models["phase45_logit"], test[ALL_BASELINE]),
        "phase45_hgb": proba_in_class_order(models["phase45_hgb"], test[ALL_BASELINE]),
        "phase45_poisson": models["phase45_poisson"].predict_proba(test),
        "plus_logit": proba_in_class_order(models["plus_logit"], test[ALL_PLUS]),
        "plus_poisson": models["plus_poisson"].predict_proba(test),
    }
    probs["phase45_ensemble"] = (probs["phase45_logit"] + probs["phase45_hgb"] + probs["phase45_poisson"]) / 3.0
    probs["plus_ensemble"] = (probs["plus_logit"] + probs["phase45_hgb"] + probs["plus_poisson"]) / 3.0

    actual_home = test["home_goals"].values
    actual_away = test["away_goals"].values
    rows = []
    for name, proba in probs.items():
        metrics = all_wdl_metrics(test["result_label"].values, proba)
        if "poisson" in name:
            model = models[name]
            lh, la = model.predict_lambdas(test)
            matrices = [model.score_matrix(lh[i], la[i]) for i in range(len(test))]
            score_metrics = scoreline_metrics(matrices, actual_home, actual_away)
            pred_scores = most_probable_scores(matrices)
        else:
            score_metrics = {"scoreline_log_loss": np.nan, "exact_score_hit_rate": np.nan, "goal_diff_hit_rate": np.nan, "outcome_hit_rate": np.nan}
            pred_scores = _canonical_scores(proba)
        rows.append(
            {
                "backtest": label,
                "model": name,
                **metrics,
                **score_metrics,
                "pts_odds1": realized_points_odds1(pred_scores, actual_home, actual_away, rules),
                "high_conf_error_rate_70": _high_conf_error_rate(test["result_label"].values, proba),
                "has_squad_coverage_test": float(test["has_squad_features"].mean()),
                "has_coach_coverage_test": float(test["has_coach_features"].mean()),
            }
        )
    return rows


def main() -> None:
    rules = load_scoring_rules()
    df = _prep(pd.read_parquet(MATRIX))
    backtests = [
        ("WC2018", "2014-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2018)),
        ("WC2022", "2018-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2022)),
    ]
    rows = []
    for label, train_end, mask in backtests:
        rows.extend(_evaluate(label, train_end, mask, df, rules))
    results = pd.DataFrame(rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as handle:
        handle.write("# Player / Coach Feature Lift Report\n\n")
        handle.write("- Plus features are sparse, World Cup-specific, and not available for WC2026 current squads.\n")
        handle.write("- Promotion requires metric lift, stable calibration, no suspicious high-confidence behavior, and WC2026 coverage.\n\n")
        handle.write("| Backtest | Model | Log loss | Brier | Accuracy | Avg p(actual) | Pts odds=1 | Exact score | Goal diff | Outcome hit | High-conf error |\n")
        handle.write("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, row in results.iterrows():
            handle.write(
                f"| {row['backtest']} | {row['model']} | {row['log_loss']} | {row['brier']} | {row['accuracy']} | "
                f"{row['avg_prob_on_actual']} | {row['pts_odds1']} | {row['exact_score_hit_rate']} | "
                f"{row['goal_diff_hit_rate']} | {row['outcome_hit_rate']} | {row['high_conf_error_rate_70']} |\n"
            )
        handle.write("\n## Coverage\n\n")
        for label in ["WC2018", "WC2022"]:
            sub = results[results["backtest"].eq(label)].iloc[0]
            handle.write(f"- {label}: squad coverage {sub['has_squad_coverage_test']:.3f}; coach coverage {sub['has_coach_coverage_test']:.3f}.\n")
        handle.write("\n## Promotion Decision\n\n")
        handle.write("Do not promote plus features into final WC2026 recommendations: WC2026 squad coverage is 0/48 teams, market-value/star-attacker features are unavailable, and the plus experiment is not a reliable production signal.\n")


if __name__ == "__main__":
    main()
