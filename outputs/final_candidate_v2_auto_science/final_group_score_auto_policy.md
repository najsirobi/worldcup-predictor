# Final Group Score Auto Policy

This file documents the deterministic science-only score selector. No manual review override is used.

## Thresholds

- `min_ev_uplift_to_override_safe`: **0.25 expected points**
- `max_allowed_variance_flag_for_ev`: **False**
- `contrarian_ev_allowed_by_default`: **False**

## Selection Rule

1. If all scientific sources agree, choose that score.
2. Otherwise choose the modal score across candidate sources and deterministic seed views.
3. If tied, choose the score with highest average expected FIF8A points.
4. If still tied, choose `safe_score`.
5. If the selected score is the EV score while safe and EV disagree, keep EV only when uplift exceeds the threshold and the row is neither high-variance nor contrarian.
6. Otherwise choose `safe_score`; no row requires manual input.

## Sources Used

- `ensemble_score`: **72** candidate rows
- `ev_score`: **72** candidate rows
- `expected_points_max_score`: **72** candidate rows
- `most_probable_score`: **72** candidate rows
- `safe_score`: **72** candidate rows

## Source Availability

- Skipped source entries: **0**
- All configured sources were available for all matches.

## Outcomes

- Final score rows: **72**
- Original manual-review rows auto-resolved: **21**
- Safe-vs-EV disagreements: **27**
- EV overrides accepted: **0**
- EV overrides rejected: **27**
- Manual review still required: **0**

- Candidate CSV: `outputs/predictions/auto_score_candidates.csv`
- Auto score CSV: `outputs/predictions/final_group_score_predictions_auto.csv`
- Seed stability report: `outputs/reports/auto_consensus_seed_stability_report.md`
