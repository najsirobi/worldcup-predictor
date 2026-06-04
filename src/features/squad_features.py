"""Squad feature aggregation from explicit player/squad columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

ATTACKING_POSITIONS = {"FW", "ST", "CF", "LW", "RW", "AM", "ATTACKING MIDFIELDER"}

SQUAD_FEATURE_COLUMNS = [
    "squad_player_count",
    "squad_total_value",
    "squad_top_11_value",
    "squad_top_15_value",
    "squad_avg_age",
    "squad_median_age",
    "squad_age_std",
    "squad_total_caps",
    "squad_total_goals",
    "share_players_top5_leagues",
    "squad_market_value_depth_ratio",
    "goalkeeper_value",
    "defender_value_total",
    "midfielder_value_total",
    "attacker_value_total",
    "top_1_attacker_value",
    "top_3_attacker_value",
    "top_5_attacker_value",
    "top_3_attacker_share_of_squad_value",
    "attacker_depth_value",
    "star_attacker_presence_flag",
    "attacking_quality_rank_within_team",
    "players_with_position",
    "players_with_market_value",
    "players_with_age",
    "attackers_identified",
    "has_squad_features",
    "has_attacker_features",
    "has_wc2026_squad_features",
]


def normalize_position(position: object) -> str:
    if pd.isna(position):
        return ""
    text = str(position).strip().upper()
    aliases = {
        "FORWARD": "FW",
        "FWD": "FW",
        "STRIKER": "ST",
        "CENTRE-FORWARD": "CF",
        "CENTER-FORWARD": "CF",
        "LEFT WINGER": "LW",
        "RIGHT WINGER": "RW",
        "ATTACKING MIDFIELD": "ATTACKING MIDFIELDER",
    }
    return aliases.get(text, text)


def is_attacking_position(position: object) -> bool:
    return normalize_position(position) in ATTACKING_POSITIONS


def _sum_top(values: pd.Series, n: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if clean.empty:
        return np.nan
    return float(clean.head(n).sum())


def _sum_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.sum())


def aggregate_squad_features(
    players: pd.DataFrame,
    *,
    team_col: str = "team",
    tournament_col: str = "tournament_name",
    player_col: str = "player",
    position_col: str | None = "position",
    age_col: str | None = "age",
    market_value_col: str | None = "market_value_eur",
    caps_col: str | None = "caps",
    goals_col: str | None = "goals",
    top5_league_col: str | None = "is_top5_league",
    source_col: str | None = "source",
) -> pd.DataFrame:
    """Aggregate one row per tournament/team from explicit player fields."""
    required = {team_col, tournament_col, player_col}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Missing required squad columns: {sorted(missing)}")

    frame = players.copy()
    frame["_position_norm"] = frame[position_col].map(normalize_position) if position_col in frame else ""
    frame["_is_attacker"] = frame["_position_norm"].map(lambda value: value in ATTACKING_POSITIONS)
    frame["_age"] = pd.to_numeric(frame[age_col], errors="coerce") if age_col in frame else np.nan
    frame["_value"] = pd.to_numeric(frame[market_value_col], errors="coerce") if market_value_col in frame else np.nan
    frame["_caps"] = pd.to_numeric(frame[caps_col], errors="coerce") if caps_col in frame else np.nan
    frame["_goals"] = pd.to_numeric(frame[goals_col], errors="coerce") if goals_col in frame else np.nan
    frame["_top5"] = pd.to_numeric(frame[top5_league_col], errors="coerce") if top5_league_col in frame else np.nan

    rows = []
    for (tournament, team), group in frame.groupby([tournament_col, team_col], dropna=False):
        values = group["_value"]
        attackers = group[group["_is_attacker"]]
        total_value = _sum_or_nan(values)
        top_3_attackers = _sum_top(attackers["_value"], 3)
        top_15 = _sum_top(values, 15)
        player_count = int(group[player_col].nunique(dropna=True))
        value_count = int(values.notna().sum())
        attacker_count = int(group["_is_attacker"].sum())
        has_values = value_count > 0

        row = {
            "tournament_name": tournament,
            "team": team,
            "source": group[source_col].dropna().iloc[0] if source_col in group and group[source_col].notna().any() else "",
            "squad_player_count": player_count,
            "squad_total_value": total_value,
            "squad_top_11_value": _sum_top(values, 11),
            "squad_top_15_value": top_15,
            "squad_avg_age": float(group["_age"].mean()) if group["_age"].notna().any() else np.nan,
            "squad_median_age": float(group["_age"].median()) if group["_age"].notna().any() else np.nan,
            "squad_age_std": float(group["_age"].std(ddof=0)) if group["_age"].notna().any() else np.nan,
            "squad_total_caps": _sum_or_nan(group["_caps"]),
            "squad_total_goals": _sum_or_nan(group["_goals"]),
            "share_players_top5_leagues": float(group["_top5"].mean()) if group["_top5"].notna().any() else np.nan,
            "squad_market_value_depth_ratio": float(top_15 / total_value) if has_values and total_value else np.nan,
            "goalkeeper_value": _sum_or_nan(group.loc[group["_position_norm"].eq("GK"), "_value"]),
            "defender_value_total": _sum_or_nan(group.loc[group["_position_norm"].isin({"DF", "CB", "LB", "RB", "DEFENDER"}), "_value"]),
            "midfielder_value_total": _sum_or_nan(group.loc[group["_position_norm"].isin({"MF", "CM", "DM", "AM", "MIDFIELDER", "ATTACKING MIDFIELDER"}), "_value"]),
            "attacker_value_total": _sum_or_nan(attackers["_value"]),
            "top_1_attacker_value": _sum_top(attackers["_value"], 1),
            "top_3_attacker_value": top_3_attackers,
            "top_5_attacker_value": _sum_top(attackers["_value"], 5),
            "top_3_attacker_share_of_squad_value": float(top_3_attackers / total_value) if has_values and total_value and not pd.isna(top_3_attackers) else np.nan,
            "attacker_depth_value": _sum_top(attackers["_value"], 8),
            "star_attacker_presence_flag": bool((attackers["_value"] >= 50_000_000).any()) if has_values else pd.NA,
            "attacking_quality_rank_within_team": np.nan,
            "players_with_position": int(group["_position_norm"].ne("").sum()),
            "players_with_market_value": value_count,
            "players_with_age": int(group["_age"].notna().sum()),
            "attackers_identified": attacker_count,
            "has_squad_features": player_count > 0,
            "has_attacker_features": has_values and attacker_count > 0,
            "has_wc2026_squad_features": False,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    for column in SQUAD_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out


def world_cup_database_players(raw_dir: str) -> pd.DataFrame:
    squads = pd.read_csv(f"{raw_dir}/squads.csv")
    players = pd.read_csv(f"{raw_dir}/players.csv")
    merged = squads.merge(players[["player_id", "birth_date"]], on="player_id", how="left")
    year = merged["tournament_name"].str.extract(r"(\d{4})")[0].astype(float)
    birth_year = pd.to_datetime(merged["birth_date"], errors="coerce").dt.year
    return pd.DataFrame(
        {
            "tournament_name": merged["tournament_name"],
            "team": merged["team_name"],
            "player": (merged["given_name"].fillna("") + " " + merged["family_name"].fillna("")).str.strip(),
            "position": merged["position_code"],
            "age": year - birth_year,
            "market_value_eur": np.nan,
            "caps": np.nan,
            "goals": np.nan,
            "source": "world_cup_database",
        }
    )


def world_cup_2022_players(raw_dir: str) -> pd.DataFrame:
    players = pd.read_csv(f"{raw_dir}/player_playingtime.csv")
    return pd.DataFrame(
        {
            "tournament_name": "2022 FIFA World Cup",
            "team": players["team"],
            "player": players["player"],
            "position": players["position"],
            "age": players["age"],
            "market_value_eur": np.nan,
            "caps": np.nan,
            "goals": np.nan,
            "source": "world_cup_2022_player_data",
        }
    )
