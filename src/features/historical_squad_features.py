"""Historical squad features, restricted to fields comparable with WC2026.

Only features that can be computed from BOTH the historical squad sources and the
WC2026 official-PDF squad features are produced as usable signals. Age and
position-mix features are available in both. Height and club-country features are
unavailable historically, so their columns are emitted but stay null (and a
``*_available`` flag records that) — they must not be used in historical training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Position-mix + age features that are comparable across historical sources and
# the WC2026 official squad parse.
COMPARABLE_FEATURE_COLUMNS = [
    "squad_player_count",
    "squad_avg_age",
    "squad_median_age",
    "squad_age_std",
    "squad_oldest_player_age",
    "squad_youngest_player_age",
    "squad_gk_count",
    "squad_df_count",
    "squad_mf_count",
    "squad_fw_count",
    "squad_fw_share",
    "squad_defensive_share",
    "squad_midfield_share",
]

# Columns emitted for schema parity with WC2026 but unavailable historically.
UNAVAILABLE_FEATURE_COLUMNS = [
    "squad_avg_height_cm",
    "squad_domestic_club_share",
    "squad_foreign_club_share",
    "squad_top5_europe_club_country_share",
    "squad_club_country_diversity",
]


def aggregate_historical_squad_features(squads: pd.DataFrame) -> pd.DataFrame:
    """One row per (tournament_year, team) with comparable squad features."""
    required = {"tournament_year", "team", "player_name", "position", "age_at_tournament_start"}
    missing = required - set(squads.columns)
    if missing:
        raise ValueError(f"Missing required squad columns: {sorted(missing)}")

    frame = squads.copy()
    if "source" not in frame.columns:
        frame["source"] = ""
    frame["_age"] = pd.to_numeric(frame["age_at_tournament_start"], errors="coerce")
    frame["_pos"] = frame["position"].astype("string").str.upper()
    frame["_height"] = pd.to_numeric(frame.get("height_cm"), errors="coerce") if "height_cm" in frame else np.nan
    frame["_club_country"] = frame.get("club_country") if "club_country" in frame else pd.NA

    rows = []
    for (year, team), g in frame.groupby(["tournament_year", "team"], dropna=False):
        ages = g["_age"].dropna()
        n = int(g["player_name"].nunique(dropna=True))
        gk = int((g["_pos"] == "GK").sum())
        df_ = int((g["_pos"] == "DF").sum())
        mf = int((g["_pos"] == "MF").sum())
        fw = int((g["_pos"] == "FW").sum())
        n_pos = gk + df_ + mf + fw

        heights = g["_height"].dropna()
        club_countries = g["_club_country"].dropna() if hasattr(g["_club_country"], "dropna") else pd.Series([], dtype=object)

        row = {
            "tournament_year": int(year) if pd.notna(year) else pd.NA,
            "tournament_name": f"{int(year)} FIFA World Cup" if pd.notna(year) else "",
            "team": team,
            "source": g["source"].dropna().iloc[0] if g["source"].notna().any() else "",
            "squad_player_count": n,
            "squad_avg_age": float(ages.mean()) if not ages.empty else np.nan,
            "squad_median_age": float(ages.median()) if not ages.empty else np.nan,
            "squad_age_std": float(ages.std(ddof=0)) if not ages.empty else np.nan,
            "squad_oldest_player_age": float(ages.max()) if not ages.empty else np.nan,
            "squad_youngest_player_age": float(ages.min()) if not ages.empty else np.nan,
            "squad_gk_count": gk,
            "squad_df_count": df_,
            "squad_mf_count": mf,
            "squad_fw_count": fw,
            "squad_fw_share": float(fw / n_pos) if n_pos else np.nan,
            "squad_defensive_share": float((gk + df_) / n_pos) if n_pos else np.nan,
            "squad_midfield_share": float(mf / n_pos) if n_pos else np.nan,
            # Unavailable historically -> null, never zero.
            "squad_avg_height_cm": float(heights.mean()) if not heights.empty else np.nan,
            "squad_domestic_club_share": np.nan,
            "squad_foreign_club_share": np.nan,
            "squad_top5_europe_club_country_share": np.nan,
            "squad_club_country_diversity": float(club_countries.nunique()) if len(club_countries) else np.nan,
            "players_with_age": int(ages.shape[0]),
            "players_with_position": int(n_pos),
            "has_historical_squad_features": n > 0,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in COMPARABLE_FEATURE_COLUMNS + UNAVAILABLE_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out
