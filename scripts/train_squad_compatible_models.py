#!/usr/bin/env python3
"""Two-layer squad-compatible backtest (Phase 5C, Task G).

Per the agreed modelling preference we do NOT train one all-purpose model that
uses squad features as ordinary features everywhere. Instead:

  Layer 1 (primary): the full-history Phase 4.5 baseline (logit / hgb / poisson
           / ensemble), trained exactly as before. Squad features never touch it.
  Layer 2 (residual): a conservative *World-Cup-only* correction. It starts from
           the baseline log-probabilities as an offset and adds a strongly L2-
           regularised linear term in the comparable squad/coach difference
           features, fit only on World Cup training rows. With the L2 term it
           shrinks toward "no change", so it can only nudge the baseline.

Backtests: train through 2014 -> test WC2018; train through 2018 -> test WC2022.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.evaluation.backtest import most_probable_scores, realized_points_odds1, scoreline_metrics, time_split
from src.evaluation.metrics import all_wdl_metrics, calibration_table
from src.ingest.rules_and_scoring import load_scoring_rules
from src.models.baselines import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSES,
    NUMERIC_FEATURES,
    PoissonScoreModel,
    make_hgb_pipeline,
    make_logit_pipeline,
    proba_in_class_order,
)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_squad_compatible.parquet"
REPORT = ROOT / "outputs" / "reports" / "squad_compatible_feature_lift_report.md"

ALL_BASELINE = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Comparable squad/coach difference features for the residual layer.
RESIDUAL_FEATURES = [
    "squad_avg_age_diff",
    "squad_age_std_diff",
    "squad_player_count_diff",
    "squad_fw_share_diff",
    "squad_defensive_share_diff",
    "squad_midfield_share_diff",
    "coach_tenure_days_at_tournament_start_diff",
    "coach_matches_before_match_diff",
    "coach_winrate_before_match_diff",
]

EPS = 1e-9
L2 = 5.0          # strong shrinkage -> conservative residual
BLEND_W = 0.5     # weight on the residual correction for the blended variant


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["has_squad_features"] = df["has_squad_features"].fillna(False).astype(int)
    df["has_coach_features"] = df["has_coach_features"].fillna(False).astype(int)
    return df[(df["match_year"] >= 2000) & df["elo_diff"].notna()].copy()


def _onehot(y):
    return np.stack([(np.asarray(y) == c).astype(float) for c in CLASSES], axis=1)


class WCResidualModel:
    """Conservative WC-only residual on top of a fitted baseline classifier.

    final_logits = log(baseline_proba) + X @ W + b, with L2 on W,b.
    W=b=0 recovers the baseline exactly, so the layer can only nudge it.
    """

    def __init__(self, baseline_pipeline, features=RESIDUAL_FEATURES, l2=L2, scale=1.0):
        self.baseline = baseline_pipeline
        self.features = features
        self.l2 = l2
        self.scale = scale  # extra global shrink applied to the correction
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.W = None
        self.b = None

    def _design(self, df, fit=False):
        X = df[self.features].to_numpy(dtype=float)
        if fit:
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
        else:
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
        return X

    def fit(self, train_wc: pd.DataFrame):
        base_p = proba_in_class_order(self.baseline, train_wc[ALL_BASELINE])
        offset = np.log(np.clip(base_p, EPS, 1.0))
        X = self._design(train_wc, fit=True)
        Y = _onehot(train_wc["result_label"].values)
        n, k = X.shape

        def negll(theta):
            W = theta[: k * 3].reshape(k, 3)
            b = theta[k * 3:]
            logits = offset + X @ W + b
            P = softmax(logits, axis=1)
            ll = -(Y * np.log(np.clip(P, EPS, 1.0))).sum() / n
            reg = self.l2 * (np.sum(W * W) + np.sum(b * b)) / n
            return ll + reg

        theta0 = np.zeros(k * 3 + 3)
        res = minimize(negll, theta0, method="L-BFGS-B")
        self.W = res.x[: k * 3].reshape(k, 3)
        self.b = res.x[k * 3:]
        return self

    def predict_proba(self, df: pd.DataFrame):
        base_p = proba_in_class_order(self.baseline, df[ALL_BASELINE])
        offset = np.log(np.clip(base_p, EPS, 1.0))
        X = self._design(df, fit=False)
        logits = offset + self.scale * (X @ self.W + self.b)
        return softmax(logits, axis=1)


def _high_conf_error_rate(y_true, proba, threshold=0.70):
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    mask = conf >= threshold
    if not mask.any():
        return np.nan
    return float(round((pred[mask] != np.asarray(y_true)[mask]).mean(), 4))


def _canonical_scores(proba):
    canonical = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}
    return [canonical[CLASSES[i]] for i in proba.argmax(axis=1)]


def _evaluate(label, train_end, test_mask, df, rules):
    train, test = time_split(df, train_end, test_mask)
    train_wc = train[(train["tournament"] == "FIFA World Cup") & (train["has_squad_features"] == 1)].copy()

    # Layer 1: baseline models (full history).
    logit = make_logit_pipeline().fit(train[ALL_BASELINE], train["result_label"])
    hgb = make_hgb_pipeline().fit(train[ALL_BASELINE], train["result_label"])
    poisson = PoissonScoreModel().fit(train)

    p_logit = proba_in_class_order(logit, test[ALL_BASELINE])
    p_hgb = proba_in_class_order(hgb, test[ALL_BASELINE])
    p_poisson = poisson.predict_proba(test)
    p_ens = (p_logit + p_hgb + p_poisson) / 3.0

    # Layer 2: WC-only residual on top of the baseline logit.
    residual = WCResidualModel(logit).fit(train_wc)
    p_resid = residual.predict_proba(test)
    p_blend = (1 - BLEND_W) * p_logit + BLEND_W * p_resid

    probs = {
        "phase45_logit": p_logit,
        "phase45_ensemble": p_ens,
        "phase45_poisson": p_poisson,
        "squad_residual_logit": p_resid,
        "squad_residual_blend": p_blend,
    }

    actual_home = test["home_goals"].values
    actual_away = test["away_goals"].values
    rows = []
    cal = {}
    for name, proba in probs.items():
        metrics = all_wdl_metrics(test["result_label"].values, proba)
        if name == "phase45_poisson":
            lh, la = poisson.predict_lambdas(test)
            mats = [poisson.score_matrix(lh[i], la[i]) for i in range(len(test))]
            sm = scoreline_metrics(mats, actual_home, actual_away)
            pred_scores = most_probable_scores(mats)
        else:
            sm = {"exact_score_hit_rate": np.nan, "goal_diff_hit_rate": np.nan, "outcome_hit_rate": np.nan}
            pred_scores = _canonical_scores(proba)
        rows.append({
            "backtest": label, "model": name, **metrics, **sm,
            "pts_odds1": realized_points_odds1(pred_scores, actual_home, actual_away, rules),
            "high_conf_error_rate_70": _high_conf_error_rate(test["result_label"].values, proba),
        })
        cal[name] = calibration_table(test["result_label"].values, proba)
    return rows, cal, float(test["has_squad_features"].mean()), len(train_wc)


def main() -> None:
    rules = load_scoring_rules()
    df = _prep(pd.read_parquet(MATRIX))
    backtests = [
        ("WC2018", "2014-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2018)),
        ("WC2022", "2018-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2022)),
    ]
    all_rows, coverage, calibs = [], {}, {}
    for label, end, mask in backtests:
        rows, cal, cov, ntrain = _evaluate(label, end, mask, df, rules)
        all_rows.extend(rows)
        coverage[label] = (cov, ntrain)
        calibs[label] = cal
    res = pd.DataFrame(all_rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("# Squad-Compatible Feature Lift Report\n\n")
        fh.write("Two-layer design: full-history Phase 4.5 baseline (primary) + a conservative, "
                 "strongly-regularised **World-Cup-only residual** in the comparable squad/coach "
                 "difference features. The residual recovers the baseline exactly when its weights "
                 "are zero, so it can only nudge the baseline.\n\n")
        fh.write(f"- Residual features: {', '.join('`'+c+'`' for c in RESIDUAL_FEATURES)}\n")
        fh.write(f"- L2 (shrinkage): {L2}; blend weight on residual: {BLEND_W}\n\n")
        fh.write("| Backtest | Model | Log loss | Brier | Accuracy | Avg p(actual) | Pts odds=1 | "
                 "Exact | Goal diff | Outcome | High-conf err |\n")
        fh.write("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, r in res.iterrows():
            fh.write(f"| {r['backtest']} | {r['model']} | {r['log_loss']} | {r['brier']} | {r['accuracy']} | "
                     f"{r['avg_prob_on_actual']} | {r['pts_odds1']} | {r['exact_score_hit_rate']} | "
                     f"{r['goal_diff_hit_rate']} | {r['outcome_hit_rate']} | {r['high_conf_error_rate_70']} |\n")
        fh.write("\n## Lift vs baseline (phase45_logit)\n\n")
        fh.write("| Backtest | Model | dLogLoss | dBrier | dAcc | dPts | (negative logloss/brier = better) |\n")
        fh.write("|---|---|--:|--:|--:|--:|---|\n")
        for label in ["WC2018", "WC2022"]:
            base = res[(res.backtest == label) & (res.model == "phase45_logit")].iloc[0]
            for m in ["squad_residual_logit", "squad_residual_blend", "phase45_ensemble"]:
                row = res[(res.backtest == label) & (res.model == m)].iloc[0]
                fh.write(f"| {label} | {m} | {row['log_loss']-base['log_loss']:+.4f} | "
                         f"{row['brier']-base['brier']:+.4f} | {row['accuracy']-base['accuracy']:+.4f} | "
                         f"{row['pts_odds1']-base['pts_odds1']:+.4f} | |\n")
        fh.write("\n## Coverage\n\n")
        for label, (cov, ntrain) in coverage.items():
            fh.write(f"- {label}: test rows with both-team squad features = {cov:.3f}; "
                     f"WC training rows for residual = {ntrain}.\n")
        fh.write("\n## Calibration (residual model, test)\n\n")
        for label in ["WC2018", "WC2022"]:
            fh.write(f"\n**{label} — squad_residual_logit**\n\n")
            fh.write(calibs[label]["squad_residual_logit"].to_markdown(index=False))
            fh.write("\n")
        fh.write("\n## Promotion read\n\n")
        improved = []
        for label in ["WC2018", "WC2022"]:
            base = res[(res.backtest == label) & (res.model == "phase45_logit")].iloc[0]
            r = res[(res.backtest == label) & (res.model == "squad_residual_logit")].iloc[0]
            better = (r["log_loss"] < base["log_loss"]) and (r["brier"] <= base["brier"])
            improved.append(better)
            fh.write(f"- {label}: residual {'IMPROVES' if better else 'does NOT improve'} "
                     f"baseline on log loss+Brier.\n")
        fh.write(f"\n**Promotion gate (both/either WC improve without calibration damage): "
                 f"{'PASS' if any(improved) else 'FAIL'}**\n")

    res.to_csv(ROOT / "outputs" / "reports" / "squad_compatible_feature_lift_results.csv", index=False)
    print(res.to_string())
    print("\nReport:", REPORT)


if __name__ == "__main__":
    main()
