#!/usr/bin/env python3
"""Build the country-context model matrix (Country-Context Task A).

Preserves every Phase 4.5 baseline row, feature and target, then appends
leakage-safe World Bank country-context features and home-away difference
features (two proxy variants). No final candidate or frozen submission file is
read or modified by this script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.country_context_match import (
    ALL_FEATURES,
    DIFF_NAME,
    IDENTITY_COLUMNS,
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    add_country_context_features,
    feature_columns,
)

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
MAPPING = ROOT / "data" / "reference" / "country_code_map.csv"
WB_INTERIM = ROOT / "data" / "interim" / "world_bank_country_context.csv"
OUT = ROOT / "data" / "processed" / "model_matrix_country_context.parquet"
REPORT = ROOT / "outputs" / "reports" / "model_matrix_country_context_report.md"

TARGET_COLUMNS = ["home_score", "away_score", "result_label", "home_goals", "away_goals"]


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
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


def _coverage_rows(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segments = [
        ("all", out),
        ("2000+", out[out["match_year"] >= 2000]),
        ("world_cup", out[out["tournament"].eq("FIFA World Cup")]),
    ]
    wc = out[out["tournament"].eq("FIFA World Cup")]
    for year in [2010, 2014, 2018, 2022]:
        segments.append((f"WC{year}", wc[wc["match_year"].eq(year)]))
    for label, sub in segments:
        n = len(sub)
        rows.append(
            {
                "segment": label,
                "rows": n,
                "both_context_all_pct": round(float(sub["has_country_context_features"].mean()), 4) if n else None,
                "both_context_direct_pct": round(float(sub["has_country_context_features_direct"].mean()), 4) if n else None,
                "any_proxy_pct": round(float(sub["any_proxy_mapping_in_match"].mean()), 4) if n else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    baseline = pd.read_parquet(BASELINE).reset_index(drop=True)
    mapping = pd.read_csv(MAPPING)
    wb = pd.read_csv(WB_INTERIM)

    features = add_country_context_features(baseline, mapping, wb)
    if len(features) != len(baseline):
        raise ValueError(f"row count mismatch: baseline={len(baseline)} features={len(features)}")

    # Validate identity columns line up before concatenation.
    for col in IDENTITY_COLUMNS:
        left = baseline[col].astype(str).reset_index(drop=True)
        right = features[col].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"country-context identity column mismatch: {col}")

    add_cols = feature_columns()
    out = pd.concat([baseline, features[add_cols].reset_index(drop=True)], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    coverage = _coverage_rows(out)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# Model Matrix — Country Context Report\n\n")
        fh.write(f"- Baseline rows: **{len(baseline)}**, columns: **{len(baseline.columns)}**\n")
        fh.write(f"- Output rows: **{len(out)}**, columns: **{len(out.columns)}**\n")
        fh.write(f"- Row count preserved: **{len(out) == len(baseline)}**\n")
        for col in TARGET_COLUMNS:
            fh.write(f"- Preserves target `{col}`: **{col in out.columns}**\n")
        fh.write(f"- Country-context columns added: **{len(add_cols)}**\n\n")

        fh.write("## Primary features (per side, log/raw)\n\n")
        fh.write(", ".join(f"`{f}`" for f in PRIMARY_FEATURES) + "\n\n")
        fh.write("## Secondary / context-only features\n\n")
        fh.write(", ".join(f"`{f}`" for f in SECONDARY_FEATURES) + "\n\n")
        fh.write("## Match-level difference features\n\n")
        fh.write("All-with-proxy variant: ")
        fh.write(", ".join(f"`{DIFF_NAME[f]}`" for f in ALL_FEATURES))
        fh.write(", `proxy_mapping_flag_diff`, `any_proxy_mapping_in_match`.\n\n")
        fh.write("Direct-only / proxy-missing variant: ")
        fh.write(", ".join(f"`direct_{DIFF_NAME[f]}`" for f in ALL_FEATURES) + ".\n\n")

        fh.write("## Coverage (both teams mapped)\n\n")
        fh.write(_md_table(coverage))

        fh.write("\n## Leakage & proxy notes\n\n")
        fh.write("- For a match in year Y, the latest World Bank value strictly before Y is used; "
                 "post-tournament and same-year values are never used.\n")
        fh.write("- Missing macro values stay null and are exposed via `_missing` flags; they are not zero-filled.\n")
        fh.write("- England and Scotland use the GBR sovereign proxy in the all-with-proxy variant "
                 "(`*_cc_is_proxy = True`); in the direct-only variant their country-context values are dropped to null.\n")
        fh.write("- Only the 48 mapped WC2026 nations carry codes, so historical coverage is limited to matches "
                 "involving those nations. This is reported honestly rather than back-filled.\n")
        fh.write("- Education and R&D spend are secondary/context-only and are kept separate from the primary set.\n")
        fh.write("- No final candidate or frozen submission file is read or modified by this script.\n")

    print(f"Wrote {OUT} ({out.shape})")
    print(f"Wrote {REPORT}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
