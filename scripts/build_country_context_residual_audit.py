#!/usr/bin/env python3
"""Country-context residual audit (Country-Context Task B).

Analysis/reporting only. Does not modify production models, bracket mappings,
final prediction files, or final submission outputs.

Expectations come from expanding year-start Poisson baseline models (each match
in year Y is scored by a model fit only on matches before Jan 1 of Y). We then
ask whether macro country-context buckets explain the leftover residuals. These
are macro proxies, not football spending, so no causal claim is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.country_context_match import PRIMARY_FEATURES
from src.models.baselines import CLASSES, PoissonScoreModel

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_country_context.parquet"
REPORT = ROOT / "outputs" / "reports" / "country_context_residual_audit.md"

BUCKET_FEATURES = {
    "log_gdp_per_capita": "GDP per capita",
    "log_population": "population",
    "log_gdp": "total GDP",
    "urbanisation_pct": "urbanisation",
    "life_expectancy": "life expectancy",
}
RESIDUALS = ["points_residual", "goal_diff_residual", "goals_for_residual", "goals_against_residual", "outcome_surprise"]


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows (insufficient coverage)._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["match_year"] >= 2000)
        & df["elo_diff"].notna()
        & df["home_goals"].notna()
        & df["away_goals"].notna()
        & df["result_label"].isin(CLASSES)
    ].copy()
    return df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def predict_year_start(df: pd.DataFrame, min_train_rows: int = 500) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = np.arange(len(df))
    rows = []
    for year in sorted(df["match_year"].astype(int).unique()):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = df[df["date"] < cutoff]
        test = df[df["match_year"] == year]
        if len(train) < min_train_rows or test.empty:
            continue
        model = PoissonScoreModel().fit(train)
        lh, la = model.predict_lambdas(test)
        probs = model.predict_proba(test)
        pred = test[["match_id"]].copy()
        pred["lambda_home"], pred["lambda_away"] = lh, la
        pred["p_home_win"], pred["p_draw"], pred["p_away_win"] = probs[:, 0], probs[:, 1], probs[:, 2]
        rows.append(pred)
    return df.merge(pd.concat(rows, ignore_index=True), on="match_id", how="inner")


def build_team_rows(m: pd.DataFrame) -> pd.DataFrame:
    sides = {
        "home": dict(team="home_team", gf="home_goals", ga="away_goals", pts="home_points",
                     pwin="p_home_win", ploss="p_away_win", egf="lambda_home", ega="lambda_away", pre="home_cc_"),
        "away": dict(team="away_team", gf="away_goals", ga="home_goals", pts="away_points",
                     pwin="p_away_win", ploss="p_home_win", egf="lambda_away", ega="lambda_home", pre="away_cc_"),
    }
    out = []
    for side, s in sides.items():
        r = pd.DataFrame()
        r["match_year"] = m["match_year"].values
        r["is_world_cup"] = m["is_world_cup"].values
        r["team"] = m[s["team"]].values
        r["actual_points"] = m[s["pts"]].values
        r["expected_points"] = (3 * m[s["pwin"]] + m["p_draw"]).values
        r["points_residual"] = r["actual_points"] - r["expected_points"]
        gf, ga = m[s["gf"]].values, m[s["ga"]].values
        r["goal_diff_residual"] = (gf - ga) - (m[s["egf"]] - m[s["ega"]]).values
        r["goals_for_residual"] = gf - m[s["egf"]].values
        r["goals_against_residual"] = ga - m[s["ega"]].values
        p_actual = np.select(
            [r["actual_points"].eq(3), r["actual_points"].eq(1), r["actual_points"].eq(0)],
            [m[s["pwin"]].values, m["p_draw"].values, m[s["ploss"]].values], default=np.nan)
        r["outcome_surprise"] = 1 - p_actual
        r["has_context"] = m[f"{s['pre']}has_context"].values
        for f in PRIMARY_FEATURES:
            r[f] = m[f"{s['pre']}{f}"].values
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    return res[res["has_context"]].copy()


def bucket_summary(res: pd.DataFrame, feature: str, n_buckets: int = 4, min_n: int = 8) -> pd.DataFrame:
    sub = res[res[feature].notna()].copy()
    if len(sub) < n_buckets * min_n:
        return pd.DataFrame()
    try:
        sub["bucket"] = pd.qcut(sub[feature], q=n_buckets, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for bucket, g in sub.groupby("bucket", observed=True):
        if len(g) < min_n:
            continue
        row = {"bucket": str(bucket), "n": len(g)}
        for metric in RESIDUALS:
            row[metric] = float(g[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = _prep(pd.read_parquet(MATRIX))
    scored = predict_year_start(df)
    res = build_team_rows(scored)
    wc = res[res["is_world_cup"].eq(1)]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# Country Context Residual Audit\n\n")
        fh.write("Analysis only — no production model, bracket, or final prediction file was changed.\n\n")
        fh.write("## Method\n\n")
        fh.write("- One row per team per match (2000+) with valid Elo, goals and result.\n")
        fh.write("- Expectations from expanding year-start Poisson baselines (fit only on matches before Jan 1 of the match year).\n")
        fh.write("- Each row carries that team's leakage-safe macro context (latest World Bank value strictly before the match year).\n")
        fh.write("- Only team-rows with a mapped World Bank country are included.\n\n")
        fh.write(f"- Team-rows with context (all matches): **{len(res)}**\n")
        fh.write(f"- Team-rows with context (World Cup only): **{len(wc)}**\n\n")
        fh.write("Positive points/goal residual = the team did better than the baseline expected. "
                 "Outcome surprise = 1 - model probability on the realised result.\n\n")

        for feature, label in BUCKET_FEATURES.items():
            fh.write(f"## {label} buckets\n\n")
            fh.write("### All matches\n\n")
            fh.write(_md_table(bucket_summary(res, feature)))
            fh.write("\n### World Cup only\n\n")
            fh.write(_md_table(bucket_summary(wc, feature)))
            fh.write("\n")

        fh.write("## World Cup splits by year\n\n")
        for year in [2010, 2014, 2018, 2022]:
            sub = wc[wc["match_year"].eq(year)]
            fh.write(f"### WC{year} (team-rows with context = {len(sub)})\n\n")
            fh.write("Mean residuals by GDP-per-capita bucket:\n\n")
            fh.write(_md_table(bucket_summary(sub, "log_gdp_per_capita", n_buckets=2, min_n=4)))
            fh.write("\n")

        fh.write("## Interpretation\n\n")
        fh.write("- These macro indicators (GDP, GDP per capita, population, urbanisation, life expectancy) are "
                 "broad development proxies, **not** football investment. Any pattern is descriptive, not causal.\n")
        fh.write("- World Cup samples per bucket are small; treat them as flags for the backtest, not evidence on their own.\n")
        fh.write("- Coverage is restricted to the 48 mapped WC2026 nations, so historical buckets skew toward "
                 "established footballing nations.\n")

    print(f"Wrote {REPORT}")
    print(f"All-match team-rows with context: {len(res)}; World Cup: {len(wc)}")


if __name__ == "__main__":
    main()
