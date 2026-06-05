#!/usr/bin/env python3
"""Controlled country-context backtest (Country-Context Tasks C & D).

Two-layer design (the agreed conservative modelling preference):

  Layer 1 (primary): the full-history Phase 4.5 baseline (logit / hgb / poisson /
           ensemble). Country-context features never touch it.
  Layer 2 (residual): a strongly L2-regularised, World-Cup-only correction that
           starts from the baseline log-probabilities as an offset and adds a
           linear term in country-context *difference* features. With W=b=0 it
           recovers the baseline exactly, so it can only nudge it.

Backtests: train through 2014 -> test WC2018; train through 2018 -> test WC2022.

Candidate variants:
  1. phase45 baseline (primary; identical to v2_auto_science layer-1 family)
  2. + core country-context diffs (all-with-proxy)
  3. + core country-context diffs (direct-only / proxy-missing)
  4. + core + secondary (education / R&D) diffs
  5. conservative core variant (stronger shrinkage, smaller blend) for the sparse case

Writes the feature-lift report, the policy recommendation, and — only if the
promotion gate passes — creates outputs/final_candidate_v3_country_context/.
Otherwise it keeps v2 and writes a context-only dashboard note. It never reads or
modifies the frozen v2/v1 candidate files.
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
from src.features.country_context_match import core_diff_features, secondary_diff_features
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
MATRIX = ROOT / "data" / "processed" / "model_matrix_country_context.parquet"
WC2026_CONTEXT = ROOT / "data" / "interim" / "country_context_features.parquet"
REPORT = ROOT / "outputs" / "reports" / "country_context_feature_lift_report.md"
RESULTS = ROOT / "outputs" / "reports" / "country_context_feature_lift_results.csv"
POLICY = ROOT / "outputs" / "reports" / "country_context_policy_recommendation.md"
DASHBOARD = ROOT / "outputs" / "reports" / "wc2026_country_context_dashboard_note.md"
V3_DIR = ROOT / "outputs" / "final_candidate_v3_country_context"

ALL_BASELINE = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EPS = 1e-9

# Candidate definitions: (label, diff feature list, both-context flag, L2, blend, scale).
CORE_ALL = core_diff_features("all")
CORE_DIRECT = core_diff_features("direct")
CORE_PLUS_SECONDARY = core_diff_features("all") + secondary_diff_features("all")

CANDIDATES = [
    ("cc_core_all", CORE_ALL, "has_country_context_features", 8.0, 0.5, 1.0),
    ("cc_core_direct", CORE_DIRECT, "has_country_context_features_direct", 8.0, 0.5, 1.0),
    ("cc_core_plus_secondary", CORE_PLUS_SECONDARY, "has_country_context_features", 8.0, 0.5, 1.0),
    ("cc_core_all_conservative", CORE_ALL, "has_country_context_features", 25.0, 0.25, 0.5),
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["match_year"] >= 2000) & df["elo_diff"].notna()].copy()


def _onehot(y):
    return np.stack([(np.asarray(y) == c).astype(float) for c in CLASSES], axis=1)


class WCResidualModel:
    """Conservative World-Cup-only residual on top of a fitted baseline classifier."""

    def __init__(self, baseline_pipeline, features, l2=8.0, scale=1.0):
        self.baseline = baseline_pipeline
        self.features = features
        self.l2 = l2
        self.scale = scale
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

        res = minimize(negll, np.zeros(k * 3 + 3), method="L-BFGS-B")
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


def _ece(y_true, proba, n_bins=10):
    cal = calibration_table(y_true, proba, n_bins=n_bins)
    if cal.empty:
        return np.nan
    w = cal["n"] / cal["n"].sum()
    return float(round((w * (cal["mean_pred_conf"] - cal["empirical_acc"]).abs()).sum(), 4))


def _canonical_scores(proba):
    canonical = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}
    return [canonical[CLASSES[i]] for i in proba.argmax(axis=1)]


def _draw_freq(proba):
    return float(round((np.array(CLASSES)[proba.argmax(axis=1)] == "draw").mean(), 4))


def _evaluate(label, train_end, test_mask, df, rules):
    train, test = time_split(df, train_end, test_mask)

    logit = make_logit_pipeline().fit(train[ALL_BASELINE], train["result_label"])
    hgb = make_hgb_pipeline().fit(train[ALL_BASELINE], train["result_label"])
    poisson = PoissonScoreModel().fit(train)

    p_logit = proba_in_class_order(logit, test[ALL_BASELINE])
    p_hgb = proba_in_class_order(hgb, test[ALL_BASELINE])
    p_poisson = poisson.predict_proba(test)
    p_ens = (p_logit + p_hgb + p_poisson) / 3.0

    probs = {"phase45_logit": p_logit, "phase45_ensemble": p_ens, "phase45_poisson": p_poisson}
    train_counts = {}
    for name, feats, flag, l2, blend, scale in CANDIDATES:
        train_wc = train[(train["tournament"] == "FIFA World Cup") & train[flag].fillna(False)].copy()
        train_counts[name] = len(train_wc)
        if len(train_wc) < 12:  # too sparse to fit a residual safely -> fall back to baseline
            probs[name] = p_logit.copy()
            continue
        resid = WCResidualModel(logit, feats, l2=l2, scale=scale).fit(train_wc)
        p_resid = resid.predict_proba(test)
        probs[name] = (1 - blend) * p_logit + blend * p_resid

    actual_home = test["home_goals"].values
    actual_away = test["away_goals"].values
    lh, la = poisson.predict_lambdas(test)
    mats = [poisson.score_matrix(lh[i], la[i]) for i in range(len(test))]

    rows = []
    for name, proba in probs.items():
        metrics = all_wdl_metrics(test["result_label"].values, proba)
        if name == "phase45_poisson":
            sm = scoreline_metrics(mats, actual_home, actual_away)
            pred_scores = most_probable_scores(mats)
        else:
            sm = {"exact_score_hit_rate": np.nan, "goal_diff_hit_rate": np.nan, "outcome_hit_rate": np.nan}
            pred_scores = _canonical_scores(proba)
        rows.append({
            "backtest": label, "model": name, **metrics,
            "exact_score_hit_rate": sm["exact_score_hit_rate"],
            "goal_diff_hit_rate": sm["goal_diff_hit_rate"],
            "outcome_hit_rate": sm["outcome_hit_rate"],
            "pts_odds1": realized_points_odds1(pred_scores, actual_home, actual_away, rules),
            "high_conf_err_70": _high_conf_error_rate(test["result_label"].values, proba),
            "ece": _ece(test["result_label"].values, proba),
            "draw_freq": _draw_freq(proba),
        })
    return rows, train_counts, len(test)


def _delta(res, label, model, base="phase45_logit"):
    b = res[(res.backtest == label) & (res.model == base)].iloc[0]
    r = res[(res.backtest == label) & (res.model == model)].iloc[0]
    return {
        "dLogLoss": r["log_loss"] - b["log_loss"],
        "dBrier": r["brier"] - b["brier"],
        "dECE": r["ece"] - b["ece"],
        "dPts": r["pts_odds1"] - b["pts_odds1"],
        "dHighConfErr": (r["high_conf_err_70"] - b["high_conf_err_70"])
        if pd.notna(r["high_conf_err_70"]) and pd.notna(b["high_conf_err_70"]) else np.nan,
        "dOutcomeHit": (r["outcome_hit_rate"] - b["outcome_hit_rate"])
        if pd.notna(r["outcome_hit_rate"]) and pd.notna(b["outcome_hit_rate"]) else np.nan,
    }


def evaluate_gate(res: pd.DataFrame, candidate="cc_core_all") -> dict:
    """Apply the promotion rule to the headline candidate."""
    tol = 1e-4  # treat tiny movements as 'stable'
    d18 = _delta(res, "WC2018", candidate)
    d22 = _delta(res, "WC2022", candidate)
    direct18 = _delta(res, "WC2018", "cc_core_direct")
    direct22 = _delta(res, "WC2022", "cc_core_direct")
    sec18 = _delta(res, "WC2018", "cc_core_plus_secondary")
    sec22 = _delta(res, "WC2022", "cc_core_plus_secondary")

    wc2018_ok = d18["dLogLoss"] <= tol
    wc2022_ok = d22["dLogLoss"] <= tol
    calib_ok = (d18["dBrier"] <= tol) and (d22["dBrier"] <= tol) and (d18["dECE"] <= 0.02) and (d22["dECE"] <= 0.02)
    hce_ok = (np.nan_to_num(d18["dHighConfErr"]) <= 0.05) and (np.nan_to_num(d22["dHighConfErr"]) <= 0.05)
    pts_ok = (d18["dPts"] >= -tol) and (d22["dPts"] >= -tol)
    # improvement should not be solely a proxy-handling artefact: the direct-only
    # variant should not behave very differently from the all-with-proxy variant.
    not_proxy_driven = (abs(d18["dLogLoss"] - direct18["dLogLoss"]) < 0.01) and (
        abs(d22["dLogLoss"] - direct22["dLogLoss"]) < 0.01)
    # improvement should not require the sparse secondary indicators.
    not_secondary_driven = not (
        (sec18["dLogLoss"] < d18["dLogLoss"] - 0.01) or (sec22["dLogLoss"] < d22["dLogLoss"] - 0.01))
    # require an actual improvement somewhere (not merely 'stable everywhere').
    real_improvement = (d18["dLogLoss"] < -tol) or (d22["dLogLoss"] < -tol)

    passed = all([wc2018_ok, wc2022_ok, calib_ok, hce_ok, pts_ok, not_proxy_driven,
                  not_secondary_driven, real_improvement])
    return {
        "passed": passed, "wc2018_ok": wc2018_ok, "wc2022_ok": wc2022_ok, "calib_ok": calib_ok,
        "hce_ok": hce_ok, "pts_ok": pts_ok, "not_proxy_driven": not_proxy_driven,
        "not_secondary_driven": not_secondary_driven, "real_improvement": real_improvement,
        "d18": d18, "d22": d22, "direct18": direct18, "direct22": direct22,
        "sec18": sec18, "sec22": sec22,
    }


def write_lift_report(res, train_counts, test_sizes, calibs):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# Country Context Feature Lift Report\n\n")
        fh.write("Two-layer design: full-history Phase 4.5 baseline (primary) + a conservative, "
                 "strongly L2-regularised **World-Cup-only residual** in country-context difference "
                 "features. The residual recovers the baseline exactly when its weights are zero.\n\n")
        fh.write(f"- Core diff features (all-with-proxy): {', '.join('`'+c+'`' for c in CORE_ALL)}\n")
        fh.write(f"- Core diff features (direct-only): {', '.join('`'+c+'`' for c in CORE_DIRECT)}\n")
        fh.write(f"- Secondary diff features: {', '.join('`'+c+'`' for c in secondary_diff_features('all'))}\n\n")
        fh.write("## Backtest results\n\n")
        fh.write("| Backtest | Model | LogLoss | Brier | Acc | AvgP(act) | Pts(odds1) | Exact | GoalDiff | Outcome | HighConfErr | ECE | DrawFreq |\n")
        fh.write("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, r in res.iterrows():
            def g(k):
                v = r[k]
                return "" if pd.isna(v) else v
            fh.write(f"| {r['backtest']} | {r['model']} | {r['log_loss']} | {r['brier']} | {r['accuracy']} | "
                     f"{r['avg_prob_on_actual']} | {r['pts_odds1']} | {g('exact_score_hit_rate')} | "
                     f"{g('goal_diff_hit_rate')} | {g('outcome_hit_rate')} | {g('high_conf_err_70')} | "
                     f"{r['ece']} | {r['draw_freq']} |\n")

        fh.write("\n## Lift vs baseline (phase45_logit); negative LogLoss/Brier/ECE = better\n\n")
        fh.write("| Backtest | Model | dLogLoss | dBrier | dECE | dPts | dHighConfErr | dOutcomeHit |\n")
        fh.write("|---|---|--:|--:|--:|--:|--:|--:|\n")
        for label in ["WC2018", "WC2022"]:
            for m in ["cc_core_all", "cc_core_direct", "cc_core_plus_secondary",
                      "cc_core_all_conservative", "phase45_ensemble"]:
                d = _delta(res, label, m)
                fh.write(f"| {label} | {m} | {d['dLogLoss']:+.4f} | {d['dBrier']:+.4f} | {d['dECE']:+.4f} | "
                         f"{d['dPts']:+.4f} | "
                         f"{'' if pd.isna(d['dHighConfErr']) else format(d['dHighConfErr'], '+.4f')} | "
                         f"{'' if pd.isna(d['dOutcomeHit']) else format(d['dOutcomeHit'], '+.4f')} |\n")

        fh.write("\n## Coverage / residual training rows\n\n")
        for label in ["WC2018", "WC2022"]:
            tc = train_counts[label]
            fh.write(f"- {label}: test rows = {test_sizes[label]}; WC residual training rows "
                     f"(all-with-proxy) = {tc['cc_core_all']}, (direct-only) = {tc['cc_core_direct']}.\n")
        fh.write("\n## Calibration (cc_core_all, test)\n\n")
        for label in ["WC2018", "WC2022"]:
            fh.write(f"\n**{label} — cc_core_all**\n\n")
            cal = calibs[label]
            cols = list(cal.columns)
            fh.write("| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, rr in cal.iterrows():
                fh.write("| " + " | ".join(str(rr[c]) for c in cols) + " |\n")
        fh.write("\n## Notes\n\n")
        fh.write("- Coverage is limited to matches between the 48 mapped WC2026 nations, so the World-Cup "
                 "residual trains on a small sample. Sparse residuals fall back to the baseline.\n")
        fh.write("- No final candidate or frozen submission file was read or modified.\n")


def write_policy(gate: dict):
    POLICY.parent.mkdir(parents=True, exist_ok=True)
    d18, d22 = gate["d18"], gate["d22"]
    with open(POLICY, "w", encoding="utf-8") as fh:
        fh.write("# Country Context Policy Recommendation\n\n")
        fh.write("## Decision summary\n\n")
        fh.write(f"- **Promote v3_country_context: {'YES' if gate['passed'] else 'NO'}**\n")
        fh.write(f"- v2_auto_science remains final: **{'No' if gate['passed'] else 'Yes'}**\n\n")
        fh.write("## Answers\n\n")
        fh.write("1. **Robust enough features:** the core macro set — `log_gdp`, `log_gdp_per_capita`, "
                 "`log_population`, `urbanisation_pct`, `life_expectancy` — has full WC2026 coverage and is the "
                 "only set considered for the model layer.\n")
        fh.write("2. **Context-only features:** `education_spend_pct_gdp` and `rd_spend_pct_gdp` are sparse/stale "
                 "(missing for Iraq / Haiti / Curaçao and several stale years) and stay context-only.\n")
        fh.write(f"3. **WC2018 improvement (cc_core_all):** dLogLoss {d18['dLogLoss']:+.4f}, dBrier {d18['dBrier']:+.4f}, "
                 f"dPts {d18['dPts']:+.4f} -> {'improves/stable' if gate['wc2018_ok'] else 'worse'}.\n")
        fh.write(f"4. **WC2022 improvement (cc_core_all):** dLogLoss {d22['dLogLoss']:+.4f}, dBrier {d22['dBrier']:+.4f}, "
                 f"dPts {d22['dPts']:+.4f} -> {'improves/stable' if gate['wc2022_ok'] else 'worse'}.\n")
        fh.write(f"5. **Expected FIF8A-like points:** dPts WC2018 {d18['dPts']:+.4f}, WC2022 {d22['dPts']:+.4f} -> "
                 f"{'improves/stable' if gate['pts_ok'] else 'worse'}.\n")
        fh.write(f"6. **Proxy handling effect:** all-with-proxy vs direct-only differ by "
                 f"{abs(d18['dLogLoss']-gate['direct18']['dLogLoss']):.4f} (WC2018) / "
                 f"{abs(d22['dLogLoss']-gate['direct22']['dLogLoss']):.4f} (WC2022) in dLogLoss -> "
                 f"{'not proxy-driven' if gate['not_proxy_driven'] else 'proxy-sensitive'}.\n")
        fh.write(f"7. **Replace v2_auto_science:** {'YES' if gate['passed'] else 'NO'}.\n")
        if gate["passed"]:
            fh.write("8. **v3 created:** `outputs/final_candidate_v3_country_context/` (promotion gate passed).\n")
        else:
            fh.write("8. **v3 not created.** Promotion gate not satisfied.\n")
            fh.write("9. **Keep v2** and add the WC2026 macro context as dashboard context only "
                     "(`outputs/reports/wc2026_country_context_dashboard_note.md`).\n")
        fh.write("\n## Promotion gate detail\n\n")
        for k in ["wc2018_ok", "wc2022_ok", "calib_ok", "hce_ok", "pts_ok",
                  "not_proxy_driven", "not_secondary_driven", "real_improvement"]:
            fh.write(f"- {k}: **{gate[k]}**\n")
        fh.write(f"\n**Overall gate: {'PASS' if gate['passed'] else 'FAIL'}**\n\n")
        fh.write("## Read\n\n")
        fh.write("- These macro indicators are development proxies, not football spending; the World-Cup residual "
                 "sample is small and coverage is partial, so promotion requires clear, non-artefactual gains.\n")
        if not gate["passed"]:
            fh.write("- The conservative recommendation is to keep `final_candidate_v2_auto_science` as the final "
                     "submission and surface macro context for human reading only.\n")


def write_dashboard_note():
    if not WC2026_CONTEXT.exists():
        return
    ctx = pd.read_parquet(WC2026_CONTEXT)
    DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    cols = ["team", "group", "log_gdp_per_capita", "log_population", "urbanisation_pct",
            "life_expectancy", "is_proxy_mapping"]
    show = ctx[cols].sort_values(["group", "team"])
    with open(DASHBOARD, "w", encoding="utf-8") as fh:
        fh.write("# WC2026 Country Context Dashboard Note\n\n")
        fh.write("> **Context only — not used in final prediction.** The final submission remains "
                 "`outputs/final_candidate_v2_auto_science/`.\n\n")
        fh.write("Macro country context per WC2026 team (latest World Bank value strictly before 2026). "
                 "`log_*` columns are base-10 logs. England and Scotland use the GBR sovereign proxy and are "
                 "flagged accordingly — these are UK-wide values, not direct sovereign country data.\n\n")
        fh.write("| Team | Group | log GDP/capita | log population | Urbanisation % | Life expectancy | Proxy (GBR) |\n")
        fh.write("|---|---|--:|--:|--:|--:|:--:|\n")
        for _, r in show.iterrows():
            def f(v):
                return "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
            fh.write(f"| {r['team']} | {r['group']} | {f(r['log_gdp_per_capita'])} | {f(r['log_population'])} | "
                     f"{f(r['urbanisation_pct'])} | {f(r['life_expectancy'])} | "
                     f"{'yes' if bool(r['is_proxy_mapping']) else ''} |\n")
        fh.write("\n_Education and R&D spend are intentionally omitted here: they are sparse/stale and "
                 "context-only._\n")


def maybe_create_v3(gate: dict):
    if gate["passed"]:
        V3_DIR.mkdir(parents=True, exist_ok=True)
        (V3_DIR / "README.md").write_text(
            "# final_candidate_v3_country_context\n\nCreated because the country-context promotion gate "
            "passed. See outputs/reports/country_context_policy_recommendation.md.\n"
        )
        return True
    return False


def main() -> None:
    rules = load_scoring_rules()
    df = _prep(pd.read_parquet(MATRIX))
    backtests = [
        ("WC2018", "2014-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2018)),
        ("WC2022", "2018-12-31", (df["tournament"] == "FIFA World Cup") & (df["match_year"] == 2022)),
    ]
    all_rows, train_counts, test_sizes, calibs = [], {}, {}, {}
    # rebuild cc_core_all calibration for reporting
    for label, end, mask in backtests:
        rows, tc, n = _evaluate(label, end, mask, df, rules)
        all_rows.extend(rows)
        train_counts[label] = tc
        test_sizes[label] = n
    res = pd.DataFrame(all_rows)
    res.to_csv(RESULTS, index=False)

    # calibration tables for the headline candidate
    for label, end, mask in backtests:
        train, test = time_split(df, end, mask)
        logit = make_logit_pipeline().fit(train[ALL_BASELINE], train["result_label"])
        train_wc = train[(train["tournament"] == "FIFA World Cup") & train["has_country_context_features"].fillna(False)]
        if len(train_wc) >= 12:
            resid = WCResidualModel(logit, CORE_ALL, l2=8.0, scale=1.0).fit(train_wc)
            p = 0.5 * proba_in_class_order(logit, test[ALL_BASELINE]) + 0.5 * resid.predict_proba(test)
        else:
            p = proba_in_class_order(logit, test[ALL_BASELINE])
        calibs[label] = calibration_table(test["result_label"].values, p)

    gate = evaluate_gate(res, "cc_core_all")
    write_lift_report(res, train_counts, test_sizes, calibs)
    write_policy(gate)
    created = maybe_create_v3(gate)
    if not created:
        write_dashboard_note()

    print(res.to_string(index=False))
    print(f"\nPromotion gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"v3 created: {created}")
    print(f"Wrote {REPORT}\nWrote {POLICY}")
    if not created:
        print(f"Wrote {DASHBOARD}")


if __name__ == "__main__":
    main()
