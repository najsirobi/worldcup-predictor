#!/usr/bin/env python3
"""Train + time-aware backtest baseline models; save final models.

Backtests:
  Split 1: train <= 2014-12-31, test = World Cup 2018.
  Split 2: train <= 2018-12-31, test = World Cup 2022.

Primary metrics: log loss, Brier, calibration, accuracy, scoreline accuracy,
expected scoring-game points. (R^2 only reported as a secondary goal-regression
metric.) Writes outputs/reports/model_backtest_report.md and saves final models.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.ingest.rules_and_scoring import load_scoring_rules
from src.models.baselines import (
    CLASSES, EmpiricalFavouriteModel, PoissonScoreModel,
    make_logit_pipeline, make_hgb_pipeline, proba_in_class_order,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    POISSON_HOME_FEATURES, POISSON_AWAY_FEATURES,
)
from src.evaluation.metrics import all_wdl_metrics, calibration_table, confusion
from src.evaluation.backtest import (
    time_split, scoreline_metrics, realized_points_odds1, most_probable_scores,
    CANONICAL_SCORE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
REPORT = ROOT / "outputs" / "reports" / "model_backtest_report.md"
MODELS_DIR = ROOT / "outputs" / "models"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def df_to_md(df):
    """Minimal DataFrame -> markdown (avoids a hard dependency on tabulate)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    rows = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + sep + rows


def _prep(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # primary training universe: 2000+, played, with a usable strength signal
    df = df[(df["match_year"] >= 2000) & df["elo_diff"].notna()].copy()
    return df


def _fit_all(train):
    models = {}
    models["m0_favourite"] = EmpiricalFavouriteModel("elo_diff").fit(train)
    models["m1_logit"] = make_logit_pipeline().fit(train[ALL_FEATURES], train["result_label"])
    models["m2_poisson"] = PoissonScoreModel().fit(train)
    models["m3_hgb"] = make_hgb_pipeline().fit(train[ALL_FEATURES], train["result_label"])
    return models


def _wdl_proba(name, model, test):
    if name == "m0_favourite":
        return model.predict_proba(test)
    if name == "m2_poisson":
        return model.predict_proba(test)
    return proba_in_class_order(model, test[ALL_FEATURES])


def _backtest(df, train_end, test_mask, rules, label, fh):
    train, test = time_split(df, train_end, test_mask)
    if len(test) == 0:
        fh.write(f"\n### {label}: no test rows\n")
        return {}
    models = _fit_all(train)
    fh.write(f"\n## Backtest: {label}\n\n")
    fh.write(f"- Train rows (≤ {train_end}): **{len(train)}** | Test rows: **{len(test)}**\n\n")
    fh.write("| Model | log_loss | Brier | accuracy | avg_p(actual) | pts(odds=1) |\n")
    fh.write("|---|--:|--:|--:|--:|--:|\n")

    results = {}
    poisson_model = models["m2_poisson"]
    lh, la = poisson_model.predict_lambdas(test)
    matrices = [poisson_model.score_matrix(lh[i], la[i]) for i in range(len(test))]
    aa, ab = test["home_goals"].values, test["away_goals"].values

    for name, model in models.items():
        proba = _wdl_proba(name, model, test)
        mets = all_wdl_metrics(test["result_label"].values, proba)
        # scoring-game points (odds=1): use canonical scores from argmax outcome,
        # except Poisson which uses its most-probable scoreline.
        if name == "m2_poisson":
            pred_scores = most_probable_scores(matrices)
        else:
            pred_scores = [CANONICAL_SCORE[CLASSES[p.argmax()]] for p in proba]
        pts = realized_points_odds1(pred_scores, aa, ab, rules)
        results[name] = {**mets, "pts_odds1": pts}
        fh.write(f"| {name} | {mets['log_loss']} | {mets['brier']} | {mets['accuracy']} | "
                 f"{mets['avg_prob_on_actual']} | {pts} |\n")

    # scoreline metrics + goal R^2 (secondary) for the Poisson model
    sm = scoreline_metrics(matrices, aa, ab)
    r2_h = round(r2_score(aa, lh), 4)
    r2_a = round(r2_score(ab, la), 4)
    fh.write(f"\n**Poisson scoreline metrics:** {sm}\n\n")
    fh.write(f"*Secondary goal-count R² (home/away): {r2_h} / {r2_a}*\n\n")

    # calibration + confusion for the best-log-loss classifier
    best = min([k for k in results], key=lambda k: results[k]["log_loss"])
    best_proba = _wdl_proba(best, models[best], test)
    fh.write(f"**Calibration (best by log loss: {best})**\n\n")
    fh.write(df_to_md(calibration_table(test["result_label"].values, best_proba)) + "\n")
    fh.write(f"**Confusion ({best})**\n\n")
    conf = confusion(test["result_label"].values, best_proba).reset_index(names="actual")
    fh.write(df_to_md(conf) + "\n")
    results["_scoreline"] = sm
    results["_best"] = best
    return results


def main():
    rules = load_scoring_rules()
    df = _prep(pd.read_parquet(MATRIX))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Model Backtest Report\n\n")
        f.write("Time-aware validation (train on past, test on future World Cups). "
                "Primary metrics: log loss, Brier, calibration, accuracy, scoreline accuracy, "
                "expected scoring-game points. R² is secondary (goal counts only).\n")

        wc2018 = (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2018)
        wc2022 = (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2022)
        r18 = _backtest(df, "2014-12-31", wc2018, rules, "train≤2014 → WC2018", f)
        r22 = _backtest(df, "2018-12-31", wc2022, rules, "train≤2018 → WC2022", f)

        # aggregate winner
        f.write("\n## Summary\n\n")
        for label, r in [("WC2018", r18), ("WC2022", r22)]:
            if not r:
                continue
            cls = {k: v for k, v in r.items() if not k.startswith("_")}
            best_ll = min(cls, key=lambda k: cls[k]["log_loss"])
            best_br = min(cls, key=lambda k: cls[k]["brier"])
            f.write(f"- {label}: best log loss = **{best_ll}** ({cls[best_ll]['log_loss']}), "
                    f"best Brier = **{best_br}** ({cls[best_br]['brier']}); "
                    f"Poisson exact-score hit {r['_scoreline']['exact_score_hit_rate']}, "
                    f"outcome hit {r['_scoreline']['outcome_hit_rate']}\n")
        f.write("\n- Scoreline distributions come from the **Poisson model** (Model 2); it is the "
                "engine for score predictions and group simulations.\n")
        f.write("- ⚠️ No tournament outcome has been simulated in this step; see prediction/simulation scripts.\n")

    # ---- fit FINAL models on all available 2000+ data and save ----
    final = _fit_all(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "final_models.pkl", "wb") as fh:
        pickle.dump({"poisson": final["m2_poisson"], "logit": final["m1_logit"],
                     "hgb": final["m3_hgb"], "favourite": final["m0_favourite"],
                     "trained_through": str(df["date"].max().date()),
                     "n_train": len(df)}, fh)
    logger.info(f"✓ Wrote {REPORT}")
    logger.info(f"✓ Saved final models to {MODELS_DIR/'final_models.pkl'} (trained on {len(df)} rows ≤ {df['date'].max().date()})")


if __name__ == "__main__":
    main()
