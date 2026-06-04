#!/usr/bin/env python3
"""Controlled backtest for rating momentum feature lift.

This script does not save final models and does not change prediction files. It
compares the Phase 4.5 baseline feature set with baseline + leak-free rating
momentum features on WC2018 and WC2022.
"""

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

from src.evaluation.backtest import most_probable_scores, realized_points_odds1, scoreline_metrics, time_split
from src.evaluation.metrics import all_wdl_metrics, calibration_table
from src.features.rating_momentum import MOMENTUM_FEATURES
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

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
REPORT = ROOT / "outputs" / "reports" / "rating_momentum_feature_lift_report.md"
SUMMARY = ROOT / "outputs" / "predictions" / "rating_momentum_backtest_summary.csv"

MOMENTUM_NUMERIC_FEATURES = NUMERIC_FEATURES + MOMENTUM_FEATURES
MOMENTUM_POISSON_HOME_FEATURES = POISSON_HOME_FEATURES + [
    "home_elo_change_6m",
    "away_elo_change_6m",
    "elo_change_6m_diff",
    "home_elo_change_12m",
    "away_elo_change_12m",
    "elo_change_12m_diff",
    "home_elo_change_24m",
    "away_elo_change_24m",
    "elo_change_24m_diff",
    "home_fifa_points_change_12m",
    "away_fifa_points_change_12m",
    "fifa_points_change_12m_diff",
    "home_fifa_rank_change_12m",
    "away_fifa_rank_change_12m",
    "fifa_rank_change_12m_diff",
    "rating_momentum_slope_12m",
]
MOMENTUM_POISSON_AWAY_FEATURES = POISSON_AWAY_FEATURES + [
    "home_elo_change_6m",
    "away_elo_change_6m",
    "elo_change_6m_diff",
    "home_elo_change_12m",
    "away_elo_change_12m",
    "elo_change_12m_diff",
    "home_elo_change_24m",
    "away_elo_change_24m",
    "elo_change_24m_diff",
    "home_fifa_points_change_12m",
    "away_fifa_points_change_12m",
    "fifa_points_change_12m_diff",
    "home_fifa_rank_change_12m",
    "away_fifa_rank_change_12m",
    "fifa_rank_change_12m_diff",
    "rating_momentum_slope_12m",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[(out["match_year"] >= 2000) & out["elo_diff"].notna()].copy()


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [
            ("num", num, numeric_features),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", cat, CATEGORICAL_FEATURES),
        ]
    )


def make_momentum_logit_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("pre", _preprocessor(MOMENTUM_NUMERIC_FEATURES)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def make_momentum_hgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
                    remainder="passthrough",
                ),
            ),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0)),
        ]
    )


class MomentumPoissonScoreModel(PoissonScoreModel):
    def fit(self, df: pd.DataFrame):
        self.home_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("glm", PoissonRegressor(alpha=1e-3, max_iter=500)),
            ]
        )
        self.away_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("glm", PoissonRegressor(alpha=1e-3, max_iter=500)),
            ]
        )
        self.home_pipe.fit(df[MOMENTUM_POISSON_HOME_FEATURES], df["home_goals"])
        self.away_pipe.fit(df[MOMENTUM_POISSON_AWAY_FEATURES], df["away_goals"])
        return self

    def predict_lambdas(self, df: pd.DataFrame):
        home = np.clip(self.home_pipe.predict(df[MOMENTUM_POISSON_HOME_FEATURES]), 1e-3, 12)
        away = np.clip(self.away_pipe.predict(df[MOMENTUM_POISSON_AWAY_FEATURES]), 1e-3, 12)
        return home, away


def _calibration_error(y_true: pd.Series, proba: np.ndarray) -> float:
    table = calibration_table(y_true, proba)
    if table.empty:
        return float("nan")
    return round(float((table["n"] * (table["mean_pred_conf"] - table["empirical_acc"]).abs()).sum() / table["n"].sum()), 4)


def _high_conf_error_rate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.80) -> float:
    pred = np.asarray(proba).argmax(axis=1)
    true = np.array([CLASSES.index(value) for value in y_true])
    high = np.asarray(proba).max(axis=1) >= threshold
    if not high.any():
        return 0.0
    return round(float((pred[high] != true[high]).mean()), 4)


def _evaluate_model(name: str, model, test: pd.DataFrame, rules: dict) -> dict:
    if name.endswith("poisson"):
        proba = model.predict_proba(test)
        lambdas_home, lambdas_away = model.predict_lambdas(test)
        matrices = [model.score_matrix(lambdas_home[idx], lambdas_away[idx]) for idx in range(len(test))]
        pred_scores = most_probable_scores(matrices)
        score_metrics = scoreline_metrics(matrices, test["home_goals"].values, test["away_goals"].values)
    else:
        features = MOMENTUM_NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES if name.startswith("momentum") else NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        proba = proba_in_class_order(model, test[features])
        pred_scores = [
            {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}[CLASSES[prediction.argmax()]]
            for prediction in proba
        ]
        score_metrics = {
            "scoreline_log_loss": np.nan,
            "exact_score_hit_rate": np.nan,
            "goal_diff_hit_rate": np.nan,
            "outcome_hit_rate": np.nan,
        }

    metrics = all_wdl_metrics(test["result_label"].values, proba)
    return {
        **metrics,
        "calibration_error": _calibration_error(test["result_label"], proba),
        "high_conf_error_rate": _high_conf_error_rate(test["result_label"], proba),
        "pts_odds1": realized_points_odds1(pred_scores, test["home_goals"].values, test["away_goals"].values, rules),
        **score_metrics,
    }


def _fit_models(train: pd.DataFrame) -> dict[str, object]:
    baseline_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    momentum_features = MOMENTUM_NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    return {
        "baseline_logit": make_logit_pipeline().fit(train[baseline_features], train["result_label"]),
        "baseline_hgb": make_hgb_pipeline().fit(train[baseline_features], train["result_label"]),
        "baseline_poisson": PoissonScoreModel().fit(train),
        "momentum_logit": make_momentum_logit_pipeline().fit(train[momentum_features], train["result_label"]),
        "momentum_hgb": make_momentum_hgb_pipeline().fit(train[momentum_features], train["result_label"]),
        "momentum_poisson": MomentumPoissonScoreModel().fit(train),
    }


def _split_rows(df: pd.DataFrame, train_end: str, year: int, rules: dict) -> list[dict]:
    test_mask = (df["tournament"].eq("FIFA World Cup")) & (df["match_year"].eq(year))
    train, test = time_split(df, train_end, test_mask)
    models = _fit_models(train)
    rows = []
    for model_name, model in models.items():
        metrics = _evaluate_model(model_name, model, test, rules)
        rows.append(
            {
                "split": f"WC{year}",
                "train_end": train_end,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                **metrics,
            }
        )
    return rows


def _promotion_decision(summary: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    baseline = summary[summary["model"].str.startswith("baseline")].copy()
    momentum = summary[summary["model"].str.startswith("momentum")].copy()
    for split in sorted(summary["split"].unique()):
        base_best = baseline[baseline["split"].eq(split)].sort_values("log_loss").iloc[0]
        mom_best = momentum[momentum["split"].eq(split)].sort_values("log_loss").iloc[0]
        if mom_best["log_loss"] < base_best["log_loss"] and mom_best["brier"] <= base_best["brier"] + 0.01:
            reasons.append(f"{split}: momentum best improves log loss without material Brier worsening.")
        else:
            reasons.append(
                f"{split}: momentum best does not clearly improve log loss/Brier "
                f"({mom_best['model']} vs {base_best['model']})."
            )
    promoted = all("improves" in reason for reason in reasons)
    if promoted:
        max_cal_worsening = 0.0
        for split in sorted(summary["split"].unique()):
            base_cal = baseline[baseline["split"].eq(split)]["calibration_error"].min()
            mom_cal = momentum[momentum["split"].eq(split)]["calibration_error"].min()
            max_cal_worsening = max(max_cal_worsening, float(mom_cal - base_cal))
        if max_cal_worsening > 0.025:
            promoted = False
            reasons.append(f"Calibration worsening too large: {max_cal_worsening:.4f}.")
    return promoted, reasons


def _df_to_md(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    df = _prep(pd.read_parquet(MATRIX))
    missing = [feature for feature in MOMENTUM_FEATURES if feature not in df.columns]
    if missing:
        raise SystemExit(f"Missing momentum features in model matrix: {missing}")

    rules = load_scoring_rules()
    rows = []
    rows.extend(_split_rows(df, "2014-12-31", 2018, rules))
    rows.extend(_split_rows(df, "2018-12-31", 2022, rules))
    summary = pd.DataFrame(rows)
    promoted, reasons = _promotion_decision(summary)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)

    display_cols = [
        "split",
        "model",
        "log_loss",
        "brier",
        "accuracy",
        "avg_prob_on_actual",
        "calibration_error",
        "high_conf_error_rate",
        "pts_odds1",
        "exact_score_hit_rate",
        "goal_diff_hit_rate",
        "outcome_hit_rate",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Rating Momentum Feature Lift Report",
                "",
                "- Purpose: test whether 6/12/24-month rating momentum improves Phase 4.5 baseline backtests.",
                "- Strict leakage rule: current ratings use latest rating strictly before match date; prior ratings use latest rating strictly before shifted cutoff date.",
                "- Final prediction files/models changed: **False**.",
                f"- Promotion decision: **{'promote' if promoted else 'do not promote'}**",
                "",
                "## Decision Reasons",
                "",
                *[f"- {reason}" for reason in reasons],
                "",
                "## Backtest Metrics",
                "",
                _df_to_md(summary[display_cols].round(4)),
                "",
                "## Momentum Features Tested",
                "",
                *[f"- `{feature}`" for feature in MOMENTUM_FEATURES],
                "",
                "## Output",
                "",
                f"- Summary CSV: `{SUMMARY.relative_to(ROOT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
