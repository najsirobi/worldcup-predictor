# Final Candidate v1 — World Cup 2026 Submission Pack

Frozen: 2026-06-04 (Phase 7C QA freeze)
Status: **QA-passed candidate, ready for submission**

This folder is a frozen snapshot of the recommended World Cup 2026 prediction submission,
together with the QA reports that validate it. Nothing in this folder should be edited; regenerate
a new candidate folder (v2, …) if upstream predictions change.

---

## Provenance

| Item | Value |
|---|---|
| Model version / source | Phase 4.5 Poisson scoreline model — `outputs/models/final_models.pkl` (key `poisson`) |
| Group prediction source | Phase 4.5 Poisson scoreline matrices (group stage **not** modified in Phase 7) |
| Bracket mapping source | Official **FIFA World Cup 26 Regulations, Articles 12.6–12.11 and Annexe C** (cached PDF), extracted to `data/reference/round_of_32_mapping.csv`, `knockout_round_progression.csv`, `third_place_assignment_annex_c.csv` |
| Simulation count | **5,000** full-tournament simulations (path-aware, official bracket) |
| Selection policy | Safe-default; last-8 = `safe_highest_probability` |

### Features explicitly NOT promoted into this candidate

- **Squad / coach features** — kept experimental; not used in group or tournament predictions.
- **Rating-momentum features** — tested in a backtest but not promoted; no final prediction file
  depends on them.

These remain experimental layers only, consistent with the project's two-layer modelling decision.

---

## Files in this candidate

### Prediction files (the submission)
- `final_submission_pack.csv` — combined pack (group scores + standings + last-8).
- `final_group_score_predictions.csv` — 72 group-match score predictions.
- `final_group_standing_predictions.csv` — 12 group final standings.
- `final_last8_predictions.csv` — QF/SF/finalist/winner picks.
- `submission_decision_table.csv` — per-match safe/EV decision detail.

### QA / supporting reports
- `final_submission_pack.md` — human-readable pack summary.
- `final_safe_vs_ev_policy.md` — safe-vs-EV policy used.
- `final_probability_mass_audit.md` — probability-mass sanity (Task B).
- `final_submission_consistency_audit.md` — cross-file consistency (Task D).
- `switzerland_path_audit.md` — Switzerland deep-run explanation (Task C).
- `annex_c_qa_report_claude.md` — Annexe C / bracket QA (Task A).

---

## QA status (Phase 7C)

| Audit | Result |
|---|---|
| Annexe C table (495 rows, unique keys, valid assignments) | PASS |
| R32 mapping (16 matches, 8 best-third slots) | PASS |
| Round progression (R16→Final, resolves to Winner) | PASS |
| Probability mass (win≈1, final≈2, sf≈4, qf≈8; ranges; monotonic; 48 teams) | PASS |
| Submission consistency (72/12 rows, nested last-8, policy) | PASS |
| pytest | see final summary |

---

## Known manual-review items

- **21 group matches** carry `manual_review_flag = True` (safe/EV disagreement,
  `safe_ev_disagreement_requires_manual_review`). The candidate plays the **safe score** for these
  by default; a human may optionally switch any of them to the EV score. See
  `submission_decision_table.csv` (21 `manual_review` rows) for detail.
- **Low-confidence group: D** (Türkiye / Paraguay / Australia / USA) — close 2nd/3rd race and high
  exact-standing uncertainty. It is the only group flagged `low` confidence; A, E, K are `medium`.
- **Switzerland** is recommended at QF and SF on a `safe_highest_probability` basis. The
  `switzerland_path_audit.md` finds this **defensible** (weak Group B + favourable early bracket),
  not an artefact. No override applied.

---

## Final recommended files to submit

1. `final_submission_pack.csv` — primary, single-file submission.

If individual files are required instead of the combined pack, submit:
2. `final_group_score_predictions.csv`
3. `final_group_standing_predictions.csv`
4. `final_last8_predictions.csv`

`submission_decision_table.csv` and the `*.md` reports are **supporting evidence only** — not part
of the graded submission.
