# Final Submission Pack

- Group score predictions: `outputs/predictions/final_group_score_predictions.csv` (72 matches)
- Group standing predictions: `outputs/predictions/final_group_standing_predictions.csv` (12 groups)
- Last-8 predictions: `outputs/predictions/final_last8_predictions.csv`
- Combined CSV: `outputs/predictions/final_submission_pack.csv`
- Safe-vs-EV policy: `outputs/reports/final_safe_vs_ev_policy.md`

## Status

- Group-stage score predictions are ready, with manual-review flags preserved.
- Group standings are ready from current group simulation summary.
- Last-8 path-aware recommendations are ready from official FIFA bracket mapping and Annexe C.
- Bracket audit: `outputs/reports/bracket_structure_audit.md`

## Manual Review

- Group matches requiring human decision: **21**
- Low-confidence groups: `['D']`
- Last-8 borderline teams are listed in `last8_recommendation_report.md`.

## First Manual-Review Matches

| # | Match | Safe | EV | Default | Reason |
|---:|---|---|---|---|---|
| 2 | Korea Republic vs Czechia | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 12 | Sweden vs Tunisia | 0-1 | 1-1 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 16 | IR Iran vs New Zealand | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 21 | England vs Croatia | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 23 | Portugal vs Congo DR | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 25 | Mexico vs Korea Republic | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 28 | Czechia vs South Africa | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 30 | Scotland vs Morocco | 0-1 | 1-0 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 32 | Türkiye vs Paraguay | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 40 | New Zealand vs Egypt | 0-1 | 1-0 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 44 | Jordan vs Algeria | 0-1 | 1-1 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 48 | Colombia vs Congo DR | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 49 | Scotland vs Brazil | 0-1 | 1-1 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 50 | Morocco vs Haiti | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 51 | Switzerland vs Canada | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 53 | Czechia vs Mexico | 0-1 | 1-1 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 55 | Ecuador vs Germany | 0-1 | 1-0 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 61 | Norway vs France | 0-1 | 1-0 | 0-1 | safe_ev_disagreement_requires_manual_review |
| 66 | Cabo Verde vs Saudi Arabia | 1-0 | 1-1 | 1-0 | safe_ev_disagreement_requires_manual_review |
| 71 | Colombia vs Portugal | 0-1 | 1-0 | 0-1 | safe_ev_disagreement_requires_manual_review |

## Last-8 Report

- `outputs/reports/last8_recommendation_report.md`
