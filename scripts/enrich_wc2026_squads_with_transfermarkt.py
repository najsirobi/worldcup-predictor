#!/usr/bin/env python3
"""Strictly enrich official WC2026 squads with Transfermarkt player values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.player_matching import match_official_to_transfermarkt

ROOT = Path(__file__).parent.parent
SQUADS = ROOT / "data" / "interim" / "wc2026_official_squads.parquet"
TM_DIR = ROOT / "data" / "raw" / "kaggle" / "transfermarkt_player_scores"
PLAYERS = TM_DIR / "players.csv"
VALUATIONS = TM_DIR / "player_valuations.csv"
OUT = ROOT / "data" / "interim" / "wc2026_squads_transfermarkt_enriched.parquet"
TEAM_FEATURES_OUT = ROOT / "data" / "interim" / "wc2026_transfermarkt_team_features.parquet"
CANDIDATES = ROOT / "data" / "reference" / "player_match_candidates.csv"
REPORT = ROOT / "outputs" / "reports" / "wc2026_transfermarkt_enrichment_report.md"


MARKET_FEATURE_COLUMNS = [
    "squad_total_market_value",
    "squad_top_11_market_value",
    "squad_top_15_market_value",
    "squad_median_market_value",
    "squad_market_value_depth_ratio",
    "gk_market_value",
    "df_market_value_total",
    "mf_market_value_total",
    "fw_market_value_total",
    "top_1_attacker_value",
    "top_3_attacker_value",
    "top_5_attacker_value",
    "top_3_attacker_share_of_squad_value",
    "attacker_depth_value",
    "has_transfermarkt_enrichment",
    "transfermarkt_match_coverage",
]


def sum_top(values: pd.Series, n: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return float(clean.head(n).sum()) if not clean.empty else np.nan


def sum_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.sum()) if not clean.empty else np.nan


def build_market_features(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = enriched.copy()
    frame["_value"] = pd.to_numeric(frame["transfermarkt_market_value_in_eur"], errors="coerce")
    for (team, fifa_code), group in frame.groupby(["team", "fifa_code"], dropna=False):
        values = group["_value"]
        coverage = float(values.notna().mean()) if len(group) else 0.0
        total = sum_or_nan(values)
        top15 = sum_top(values, 15)
        attackers = group.loc[group["position"].eq("FW"), "_value"]
        top3_attackers = sum_top(attackers, 3)
        rows.append(
            {
                "team": team,
                "fifa_code": fifa_code,
                "squad_total_market_value": total,
                "squad_top_11_market_value": sum_top(values, 11),
                "squad_top_15_market_value": top15,
                "squad_median_market_value": float(values.median()) if values.notna().any() else np.nan,
                "squad_market_value_depth_ratio": float(top15 / total) if total and not pd.isna(total) and not pd.isna(top15) else np.nan,
                "gk_market_value": sum_or_nan(group.loc[group["position"].eq("GK"), "_value"]),
                "df_market_value_total": sum_or_nan(group.loc[group["position"].eq("DF"), "_value"]),
                "mf_market_value_total": sum_or_nan(group.loc[group["position"].eq("MF"), "_value"]),
                "fw_market_value_total": sum_or_nan(attackers),
                "top_1_attacker_value": sum_top(attackers, 1),
                "top_3_attacker_value": top3_attackers,
                "top_5_attacker_value": sum_top(attackers, 5),
                "top_3_attacker_share_of_squad_value": float(top3_attackers / total) if total and not pd.isna(total) and not pd.isna(top3_attackers) else np.nan,
                "attacker_depth_value": sum_top(attackers, 8),
                "has_transfermarkt_enrichment": bool(values.notna().sum() >= 18),
                "transfermarkt_match_coverage": coverage,
                "transfermarkt_players_matched": int(values.notna().sum()),
                "transfermarkt_players_total": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    for column in MARKET_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out.sort_values("team").reset_index(drop=True)


def write_report(enriched: pd.DataFrame, team_features: pd.DataFrame, candidates: pd.DataFrame) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    accepted = enriched["transfermarkt_match_status"].eq("accepted")
    coverage_by_team = team_features[["team", "transfermarkt_players_matched", "transfermarkt_players_total", "transfermarkt_match_coverage", "has_transfermarkt_enrichment"]]
    robust_teams = int(team_features["has_transfermarkt_enrichment"].sum()) if not team_features.empty else 0
    with REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# WC2026 Transfermarkt Enrichment Report\n\n")
        handle.write(f"- Official squad source: `{SQUADS.relative_to(ROOT)}`\n")
        handle.write(f"- Transfermarkt players source: `{PLAYERS.relative_to(ROOT)}`\n")
        handle.write(f"- Enriched output: `{OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Team market feature output: `{TEAM_FEATURES_OUT.relative_to(ROOT)}`\n")
        handle.write(f"- Player match candidates: `{CANDIDATES.relative_to(ROOT)}`\n")
        handle.write(f"- Official player rows: **{len(enriched)}**\n")
        handle.write(f"- Accepted exact name+DOB matches: **{int(accepted.sum())}**\n")
        handle.write(f"- Overall match coverage: **{accepted.mean():.3f}**\n")
        handle.write(f"- Teams meeting 18-player suggested threshold: **{robust_teams} / {team_features['team'].nunique() if not team_features.empty else 0}**\n")
        handle.write(f"- Ambiguous candidate rows requiring review: **{len(candidates)}**\n")
        handle.write("- Auto-accepted matches require exact normalized official name key plus exact DOB.\n")
        handle.write("- Missing market values remain null; no zero fill is used.\n")
        handle.write("- Market-value/star-attacker features are not promoted to production predictions unless coverage is sufficiently even.\n\n")
        handle.write("## Team Coverage\n\n")
        handle.write("| Team | Matched | Total | Coverage | Meets 18-player threshold |\n|---|--:|--:|--:|---:|\n")
        for _, row in coverage_by_team.sort_values("team").iterrows():
            handle.write(
                f"| {row['team']} | {row['transfermarkt_players_matched']} | {row['transfermarkt_players_total']} | "
                f"{row['transfermarkt_match_coverage']:.3f} | {row['has_transfermarkt_enrichment']} |\n"
            )


def main() -> None:
    if not SQUADS.exists():
        raise FileNotFoundError(f"Missing official squads: {SQUADS}")
    if not PLAYERS.exists():
        raise FileNotFoundError(f"Missing Transfermarkt players file: {PLAYERS}")
    squads = pd.read_parquet(SQUADS)
    players = pd.read_csv(PLAYERS, low_memory=False)
    enriched, candidates = match_official_to_transfermarkt(squads, players)
    team_features = build_market_features(enriched)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(OUT, index=False)
    team_features.to_parquet(TEAM_FEATURES_OUT, index=False)
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        candidates = pd.DataFrame(
            columns=[
                "official_team",
                "official_player_name",
                "official_date_of_birth",
                "transfermarkt_player_id",
                "transfermarkt_name",
                "transfermarkt_date_of_birth",
                "match_method",
                "confidence",
                "needs_review",
                "evidence",
            ]
        )
    candidates.to_csv(CANDIDATES, index=False)
    write_report(enriched, team_features, candidates)


if __name__ == "__main__":
    main()
