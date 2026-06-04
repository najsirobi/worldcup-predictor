#!/usr/bin/env python3
"""Run full tournament simulation with official FIFA bracket mapping."""

from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from src.features.template_features import build_team_snapshots, match_lambdas_matrix, resolve
from src.simulation.full_tournament import simulate_full_tournament_official
from src.simulation.knockout_bracket import (
    MissingBracketMappingError,
    load_round_of_32_mapping,
    load_round_progression,
    load_third_place_annex,
)

ROOT = Path(__file__).parent.parent
R32_MAPPING = ROOT / "data" / "reference" / "round_of_32_mapping.csv"
PROGRESSION = ROOT / "data" / "reference" / "knockout_round_progression.csv"
ANNEX = ROOT / "data" / "reference" / "third_place_assignment_annex_c.csv"
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
MODELS = ROOT / "outputs" / "models" / "final_models.pkl"
GROUP_VIEW = ROOT / "outputs" / "predictions" / "group_submission_view.csv"
OUT = ROOT / "outputs" / "predictions" / "full_tournament_simulation_summary.csv"
REPORT = ROOT / "outputs" / "reports" / "full_tournament_simulation_report.md"
N_SIMS = 5_000


def write_missing_report(error: Exception) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Full Tournament Simulation Report",
                "",
                "- Full path-aware tournament simulation created: **False**",
                f"- Reason: `{error}`",
                f"- Required Round-of-32 mapping: `{R32_MAPPING.relative_to(ROOT)}`",
                f"- Required progression mapping: `{PROGRESSION.relative_to(ROOT)}`",
                f"- Required Annexe C mapping: `{ANNEX.relative_to(ROOT)}`",
                "- No simulation CSV was written because bracket placement would be invented.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    try:
        r32_mapping = load_round_of_32_mapping(R32_MAPPING)
        progression = load_round_progression(PROGRESSION)
        annex = load_third_place_annex(ANNEX)
    except MissingBracketMappingError as exc:
        write_missing_report(exc)
        raise SystemExit(2)

    model_matrix = pd.read_parquet(MATRIX)
    model_matrix["date"] = pd.to_datetime(model_matrix["date"])
    snapshots = build_team_snapshots(model_matrix)
    with MODELS.open("rb") as handle:
        poisson = pickle.load(handle)["poisson"]
    template = pd.read_csv(TEMPLATE)

    matrices, match_rows, unresolved = {}, [], set()
    for _, row in template.iterrows():
        result = match_lambdas_matrix(poisson, snapshots, row["team_a"], row["team_b"], row["date"])
        if result is None:
            unresolved.add(row["team_a"] if resolve(row["team_a"], snapshots) is None else row["team_b"])
            continue
        _, _, matrix = result
        match_number = int(row["match_number"])
        matrices[match_number] = matrix
        match_rows.append(
            {
                "group": row["group"],
                "match_number": match_number,
                "team_a": row["team_a"],
                "team_b": row["team_b"],
            }
        )

    if unresolved:
        error = ValueError(f"Unresolved teams for scoreline matrices: {sorted(unresolved)}")
        write_missing_report(error)
        raise SystemExit(2)

    group_matches = pd.DataFrame(match_rows)
    group_view = pd.read_csv(GROUP_VIEW)
    summary = simulate_full_tournament_official(
        group_matches,
        matrices,
        r32_mapping,
        progression,
        annex,
        group_view,
        n_sims=N_SIMS,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("p_win_world_cup", ascending=False).to_csv(OUT, index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Tournament Simulation Report",
        "",
        "- Full path-aware tournament simulation created: **True**",
        f"- Simulations: **{N_SIMS:,}**",
        f"- Round-of-32 mapping: `{R32_MAPPING.relative_to(ROOT)}`",
        f"- Knockout progression mapping: `{PROGRESSION.relative_to(ROOT)}`",
        f"- Annexe C mapping: `{ANNEX.relative_to(ROOT)}`",
        f"- Group matches simulated: **{len(group_matches)}**",
        "- Knockout match probabilities use existing group-simulation strength fallback; no model retraining.",
        "- Group-stage tables are sampled from current Phase 4.5 Poisson scoreline distributions.",
        "- Draws after 90 minutes are resolved by strength-adjusted advancement share; no precise penalty model is claimed.",
        "- Runtime note: 20,000 simulations exceeded the practical sandbox runtime, so this run uses the allowed 5,000 simulations.",
        "",
        "## Top World Cup Win Probabilities",
        "",
        "| Team | P QF | P SF | P Final | P Winner |",
        "|---|--:|--:|--:|--:|",
    ]
    for _, row in summary.sort_values("p_win_world_cup", ascending=False).head(12).iterrows():
        lines.append(
            f"| {row['team']} | {row['p_reach_qf']:.3f} | {row['p_reach_sf']:.3f} | "
            f"{row['p_reach_final']:.3f} | {row['p_win_world_cup']:.3f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
