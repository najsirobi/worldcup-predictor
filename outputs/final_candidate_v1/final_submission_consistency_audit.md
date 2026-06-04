# Final Submission Consistency Audit

Generated: 2026-06-04
Author: Claude QA layer (Phase 7C)
Scope: read-only consistency check across submission files. Nothing modified.

Files inspected:
- `outputs/predictions/final_submission_pack.csv`
- `outputs/predictions/final_group_score_predictions.csv`
- `outputs/predictions/final_group_standing_predictions.csv`
- `outputs/predictions/final_last8_predictions.csv`
- `outputs/predictions/submission_decision_table.csv`

---

## Checks

| # | Check | Expected | Observed | Result |
|---|-------|----------|----------|--------|
| 1 | Group score rows | 72 | 72 (also 72 `group_score` rows in pack) | PASS |
| 2 | Group standing rows | 12 | 12 | PASS |
| 3a | Quarter-finalists | 8 | 8 | PASS |
| 3b | Semi-finalists | 4 | 4 | PASS |
| 3c | Finalists | 2 | 2 | PASS |
| 3d | Winner | 1 | 1 (Spain) | PASS |
| 4 | Last-8 teams exist in tournament teams | all | all present, 0 missing | PASS |
| 5 | Winner ∈ finalists | yes | Spain ∈ {Spain, Argentina} | PASS |
| 6 | Finalists ⊆ SF list | yes | yes | PASS |
| 7 | SF teams ⊆ QF list | yes | yes | PASS |
| 8 | No missing manual-review flags | 0 nulls | 0 nulls (values ∈ {True, False}) | PASS |
| 9 | final_group_score consistent with safe/EV policy | consistent | consistent (see below) | PASS |

---

## Last-8 nesting (checks 5–7)

- Winner: **Spain**
- Finalists: **Spain, Argentina** → winner is one of them.
- Semi-finalists: **Spain, Switzerland, Argentina, England** → both finalists included.
- Quarter-finalists: **Spain, Switzerland, Argentina, England, Belgium, France, Netherlands, Germany**
  → all four semi-finalists included.

The Last-8 set is strictly nested (Winner ⊆ Finalists ⊆ SF ⊆ QF), as required.

---

## Safe/EV policy consistency (check 9)

The submission uses a **safe-default policy**:

- `final_recommended_score == safe_score` for **all 72** group matches (0 mismatches on unflagged
  rows; flagged rows also retain the safe score).
- **21 of 72** matches carry `manual_review_flag = True` with reason
  `safe_ev_disagreement_requires_manual_review`. These are matches where the EV-maximising score
  differs from the safe score.
- These 21 flagged matches correspond exactly to the **21** `manual_review` entries in
  `submission_decision_table.csv` (`suggested_submission_score == "manual_review"`). The two files
  agree perfectly.

Interpretation: the frozen candidate plays the **safe score everywhere** and surfaces the 21
safe/EV disagreements as optional manual-review items — it does not silently switch to EV scores.
This is internally consistent and matches the documented safe-vs-EV policy.

---

## Manual-review items

- **Group-stage matches flagged for manual review: 21** (out of 72).
- Last-8 selections: none require manual review (all `safe_highest_probability`).

---

## Verdict

**ALL consistency checks PASS.** The five submission files are mutually consistent, the Last-8
hierarchy is correctly nested, the safe/EV policy is applied uniformly (safe-default with 21
flagged disagreements), and there are no missing flags or orphaned teams.
