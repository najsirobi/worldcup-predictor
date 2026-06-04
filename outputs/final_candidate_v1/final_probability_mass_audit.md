# Final Probability Mass Audit

Generated: 2026-06-04
Author: Claude QA layer (Phase 7C)
Scope: read-only sanity check of tournament probability mass. No predictions modified.

Source files inspected:
- `outputs/predictions/full_tournament_simulation_summary.csv` (48 teams)
- `outputs/predictions/last8_recommendations.csv`
- `outputs/predictions/final_last8_predictions.csv`

Simulation basis: 5,000 full-tournament simulations, official FIFA bracket + Annexe C,
group stage sampled from Phase 4.5 Poisson scoreline matrices.

---

## 1. Probability mass checks

| # | Check | Expected | Observed | Result |
|---|-------|----------|----------|--------|
| 1 | sum(p_win_world_cup) | ≈ 1.0 | 1.0 | PASS |
| 2 | sum(p_reach_final) | ≈ 2.0 | 2.0 | PASS |
| 3 | sum(p_reach_sf) | ≈ 4.0 | 4.0 | PASS |
| 4 | sum(p_reach_qf) | ≈ 8.0 | 8.0 | PASS |
| — | sum(p_reach_r16) | ≈ 16.0 | 16.0 | PASS |
| — | sum(p_reach_r32) | ≈ 32.0 | 32.0 | PASS |

The round sums equal the exact number of bracket slots at each stage (32 → 16 → 8 → 4 → 2 → 1).
This is the correct invariant for a knockout: across all simulations, exactly N teams occupy
each round, so the per-team probabilities sum to the slot count. Exact integers indicate the
simulation counts are internally consistent (no double-counting or dropped teams).

---

## 2. Structural checks

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 5 | All probabilities in [0, 1] | PASS | No value < 0 or > 1 across all probability columns |
| 6 | No duplicated teams | PASS | 0 duplicate team rows |
| 7 | 48 teams in tournament summary | PASS | 48 rows, 48 unique team names |
| 8 | p_reach_qf ≥ p_reach_sf ≥ p_reach_final ≥ p_win_world_cup (per team) | PASS | 0 violations across all 48 teams |

Additional (stronger) monotonicity check — full chain
`p_reach_r32 ≥ p_reach_r16 ≥ p_reach_qf ≥ p_reach_sf ≥ p_reach_final ≥ p_win_world_cup` —
holds for all 48 teams with 0 violations.

---

## 3. Cross-file consistency (last-8 files)

`final_last8_predictions.csv` and `last8_recommendations.csv` draw their probabilities directly
from the tournament summary columns:
- QF probabilities match `p_reach_qf` (e.g. Spain 0.6524, Switzerland 0.503).
- SF probabilities match `p_reach_sf` (e.g. Spain 0.5164).
- Finalist probabilities match `p_reach_final` (e.g. Spain 0.3888, Argentina 0.1904).
- Winner probability matches `p_win_world_cup` (Spain 0.2804).

No inconsistency found between the recommendation files and the simulation summary.

---

## 4. Verdict

**ALL probability-mass and structural checks PASS.** The tournament simulation output is
internally consistent and the last-8 recommendation files are faithful projections of it.
No anomalies, no out-of-range values, no duplicate teams, and monotonicity holds for every team.
