#!/usr/bin/env python3
"""Recalculate live group advancement probabilities (Travel Mode).

Pins played matches to their entered scores and samples unplayed matches from
the frozen active candidate's submitted scorelines. Does NOT change submitted
predictions and does NOT train or regenerate first-round picks from actual
results.

Outputs:
    outputs/live/live_group_stage_simulation_summary.csv
    outputs/live/live_group_stage_simulation_summary.json
    outputs/reports/live_group_stage_simulation_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.live.active_candidate import load_active_candidate
from src.live.live_simulation import (
    GOAL_FLOOR,
    N_ADVANCING_THIRDS,
    load_predictions,
    simulate_live,
)
from src.live.scores_override import OVERRIDE_PATH, load_override, utc_now_iso
from src.live.submission_guard import guard_frozen_submission
from src.live.tournament_state import LIVE_STATE_SEMANTICS

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
REPORTS_DIR = ROOT / "outputs" / "reports"

TIE_BREAK_NOTE = (
    "Ranking within a group uses points -> goal difference -> goals for -> "
    "uniform random draw. The 8 best third-placed teams are chosen by the same "
    "ladder. Head-to-head and fair-play tie-breakers are not modelled."
)


def write_report(summary, n_sims, elapsed, candidate, path: Path) -> None:
    lines = [
        "# Live Group-Stage Simulation Report",
        "",
        f"_Generated: {utc_now_iso()}_",
        "",
        f"- Active candidate: **{candidate['name']}** (`{candidate['active_candidate_dir']}`)",
        f"- Simulations: **{n_sims:,}** (runtime {elapsed:.1f}s)",
        f"- Format: top 2 per group + {N_ADVANCING_THIRDS} best third-placed teams advance",
        "- Played matches are pinned to entered scores; unplayed matches sampled "
        f"from frozen recommended scorelines (Poisson, goal floor {GOAL_FLOOR}).",
        "",
        "## Method & limitations",
        "",
        "This does not change submitted predictions. Actual results from "
        "`data/live/scores_override.csv` pin played group matches only; unplayed "
        f"fixtures reuse `{candidate['score_predictions_file']}` from the active "
        "frozen candidate.",
        "",
        "## Live semantics",
        "",
        *[f"- {text}" for text in LIVE_STATE_SEMANTICS.values()],
        "",
        TIE_BREAK_NOTE,
        "",
        "## Advancement probabilities by group",
        "",
    ]
    for group, sub in summary.groupby("group"):
        lines.append(f"### Group {group}")
        lines.append("")
        lines.append("| Team | MP | Pts | P(win) | P(top2) | P(advance) | xPts | Status |")
        lines.append("|------|----|-----|--------|---------|------------|------|--------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['team']} | {r['matches_played']} | {r['current_points']} | "
                f"{r['p_rank1']:.2f} | {r['p_top2']:.2f} | {r['p_advance']:.2f} | "
                f"{r['expected_points']:.2f} | {r['status']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sims",
        type=int,
        default=20000,
        help="Number of Monte Carlo simulations (use 5000 for a faster run).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores", type=Path, default=OVERRIDE_PATH)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Score predictions CSV (defaults to the active candidate's file).",
    )
    args = parser.parse_args()

    with guard_frozen_submission("recalculate_live_simulations.py"):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        candidate_obj = load_active_candidate()
        candidate = candidate_obj.as_dict()
        predictions_path = args.predictions or candidate_obj.score_predictions_path

        scores = load_override(args.scores)
        predictions = load_predictions(predictions_path)

        start = time.perf_counter()
        summary = simulate_live(scores, predictions, n_sims=args.sims, seed=args.seed)
        elapsed = time.perf_counter() - start

        summary.to_csv(LIVE_DIR / "live_group_stage_simulation_summary.csv", index=False)
        payload = {
            "generated_at": utc_now_iso(),
            "active_candidate": candidate,
            "n_sims": int(args.sims),
            "seed": int(args.seed),
            "goal_floor": GOAL_FLOOR,
            "n_advancing_thirds": N_ADVANCING_THIRDS,
            "tie_break_note": TIE_BREAK_NOTE,
            "semantics": LIVE_STATE_SEMANTICS,
            "teams": summary.to_dict(orient="records"),
        }
        (LIVE_DIR / "live_group_stage_simulation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        write_report(
            summary, args.sims, elapsed, candidate, REPORTS_DIR / "live_group_stage_simulation_report.md"
        )
        print(
            f"Ran {args.sims:,} live simulations in {elapsed:.1f}s. "
            f"Wrote summary CSV/JSON + report."
        )


if __name__ == "__main__":
    main()
