#!/usr/bin/env python3
"""Monte Carlo group-stage simulation (Phase 4, Task E)."""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.template_features import build_team_snapshots, match_lambdas_matrix, resolve
from src.simulation.group_stage import simulate_groups

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MATRIX = ROOT / "data" / "processed" / "model_matrix_baseline.parquet"
TEMPLATE = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"
MODELS = ROOT / "outputs" / "models" / "final_models.pkl"
OUT_CSV = ROOT / "outputs" / "predictions" / "group_stage_simulation_summary.csv"
REPORT = ROOT / "outputs" / "reports" / "group_stage_simulation_report.md"

N_SIMS = 20000


def main():
    mm = pd.read_parquet(MATRIX); mm["date"] = pd.to_datetime(mm["date"])
    snaps = build_team_snapshots(mm)
    with open(MODELS, "rb") as f:
        poisson = pickle.load(f)["poisson"]
    tmpl = pd.read_csv(TEMPLATE)

    matrices, rows, unresolved = {}, [], set()
    for _, t in tmpl.iterrows():
        res = match_lambdas_matrix(poisson, snaps, t["team_a"], t["team_b"], t["date"])
        if res is None:
            unresolved.add(t["team_a"] if resolve(t["team_a"], snaps) is None else t["team_b"])
            continue
        _, _, M = res
        matrices[int(t["match_number"])] = M
        rows.append({"group": t["group"], "match_number": int(t["match_number"]),
                     "team_a": t["team_a"], "team_b": t["team_b"]})

    gm = pd.DataFrame(rows)
    logger.info(f"Simulating {gm['group'].nunique()} groups, {len(gm)} matches, {N_SIMS} sims...")
    summary = simulate_groups(gm, matrices, n_sims=N_SIMS)
    summary = summary.sort_values(["group", "expected_points"], ascending=[True, False])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    logger.info(f"✓ Wrote {OUT_CSV} ({len(summary)} teams)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Group-Stage Simulation Report\n\n")
        f.write(f"- Monte Carlo simulations: **{N_SIMS:,}** per (joint) run, seed=42\n")
        f.write(f"- Groups: {gm['group'].nunique()} | matches: {len(gm)} | "
                f"unresolved teams: {sorted(unresolved) or 'none'}\n")
        f.write("- Scoreline source: Poisson model matrices (0–10 goals/side).\n")
        f.write("- Table sorting: points → goal difference → goals for → random tie-break.\n")
        f.write("- Best-thirds: the 8 best of the 12 third-placed teams advance (joint sim).\n\n")
        f.write("> **Tie-break limitation:** real FIFA rules also use head-to-head, disciplinary "
                "and drawing of lots; we approximate with GD→GF→random. Documented, not exact.\n\n")
        f.write("## Projected qualifiers (top 2 per group by P(top2))\n\n")
        f.write("| Group | Team | E[pts] | P(1st) | P(2nd) | P(top2) | P(advance incl. best-3rd) |\n")
        f.write("|---|---|--:|--:|--:|--:|--:|\n")
        for g in sorted(summary["group"].unique()):
            sub = summary[summary["group"] == g].sort_values("p_top2", ascending=False)
            for _, r in sub.head(2).iterrows():
                f.write(f"| {g} | {r['team']} | {r['expected_points']} | {r['p_finish_1st']} | "
                        f"{r['p_finish_2nd']} | {r['p_top2']} | {r['p_advance_with_best_thirds']} |\n")
        f.write("\n## Uncertainty / variance note (Hypothesis 3)\n\n")
        f.write("- Probabilities come from full scoreline distributions + Monte Carlo, i.e. luck is "
                "modelled as **variance**, not as a deterministic skill term.\n")
        f.write("- Teams with `p_top2` near 0.5 and several plausible ranks are the high-variance groups.\n")
        f.write("- ⚠️ No knockout bracket simulated yet; penalty/shoot-out randomness deferred to knockout phase.\n")

    logger.info(f"✓ Wrote {REPORT}")


if __name__ == "__main__":
    main()
