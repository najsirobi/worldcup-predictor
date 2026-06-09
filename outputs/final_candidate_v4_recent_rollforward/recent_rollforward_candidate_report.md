# v4 Recent Rollforward Candidate Report

**Generated:** 2026-06-09
**Rule applied:** `R1_only_diff_5_0`

## Provenance

- Base: `final_candidate_v2_auto_science` form features updated with 35 June 3–8 matches appended to the backbone
- Rolling form re-computed via `build_model_matrix()` — no model retraining
- ELO / FIFA ratings frozen (max 2025-12-13 / 2024-06-20)
- Auto-consensus policy applied (same config as v2)
- R1_only_diff_5_0 applied on top of v4 auto score
- Overlay: `data/reference/wc2026_human_upside_overlay.csv`

## Constraints satisfied

- broad_human_overlay_used: **false**
- manual_approval_used: **false**
- subjective_override_used: **false**
- rolling_forward_update: **true**
- model_retrained: **false**

## R1 adjustment summary

- Group matches: **72**
- R1-adjusted scores: **4**

| match | group | fixture | v4_auto | adjusted | change type | overlay_diff |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | L | England vs Croatia | 1-0 | 2-0 | favourite_strengthened | +6.25 |
| 23 | K | Portugal vs Congo DR | 1-0 | 2-0 | favourite_strengthened | +6.5 |
| 50 | C | Morocco vs Haiti | 1-0 | 2-0 | favourite_strengthened | +5 |
| 64 | G | Egypt vs IR Iran | 0-1 | 1-1 | decisive_to_draw | +5.5 |

## Lineup distortion caveat

Argentina/Messi (LA002, active_monitoring) June 6 match included at full weight in rolling form. Distortion documented in outputs/predictions/recent_international_lineup_distortion_audit.csv but custom weighting not implemented (would alter model architecture).

## Frozen candidate integrity

- v2_auto_science byte-identical: **True**
- v3_objective_residual byte-identical: **True**
