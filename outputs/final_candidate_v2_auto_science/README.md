# Final Candidate v2 Auto Science

This frozen candidate uses a fully automatic science-only score policy.

- No manual overrides used.
- Manual-review flags were auto-resolved: **21** rows.
- Score policy: source agreement, modal consensus, expected-points tie-break, safe-score final fallback, and gated EV override.
- Thresholds: `min_ev_uplift_to_override_safe=0.25`, `max_allowed_variance_flag_for_ev=false`, `contrarian_ev_allowed_by_default=false`.
- Sources used: safe score, EV score, most probable score, ensemble score, expected-points-max score.
- Last-8 source: copied unchanged from `outputs/final_candidate_v1/final_last8_predictions.csv`.
- Tests passed: `.venv/bin/python -m pytest` -> 223 passed.

## Final Files To Submit

- `final_group_score_predictions_auto.csv`
- `final_group_standing_predictions_auto.csv`
- `final_last8_predictions_auto.csv`
- `final_submission_pack_auto.csv`
