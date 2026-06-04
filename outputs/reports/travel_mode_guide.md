# Travel Mode Guide

Travel Mode is a **free, phone-friendly, static** system for tracking the World
Cup 2026 group stage while you are away from your computer. You enter actual
scores from your phone; GitHub Actions recomputes the live tables, simulations
and your earned points, and republishes a mobile dashboard on GitHub Pages.

**It never retrains a model, never calls a live-score API, and never changes any
prediction.** It only ingests scores, scores them against the active candidate,
and rebuilds the dashboard.

---

## 0. The active candidate

Every Travel Mode output is computed against one frozen "active candidate" set
of predictions, selected by `data/live/active_candidate.yml`:

```yaml
active_candidate_dir: outputs/final_candidate_v1
score_predictions_file: final_group_score_predictions.csv
standing_predictions_file: final_group_standing_predictions.csv
last8_predictions_file: final_last8_predictions.csv
submission_pack_file: final_submission_pack.csv
```

The dashboard always shows which candidate is active (header + "Active candidate"
section). The frozen candidate folders themselves are **never modified**.

---

## 1. Initialise `scores_override.csv`

Once, before the tournament:

```bash
python scripts/init_scores_override.py
```

This creates `data/live/scores_override.csv` with all 72 group matches set to
`scheduled` and empty goals. It refuses to overwrite a file that already has
played matches unless you pass `--force`.

Columns: `match_number, group, date, team_a, team_b, team_a_goals,
team_b_goals, status, source, updated_at, notes`. Status is one of
`scheduled | played | postponed | void`.

---

## 2. Update ONE match from your phone (GitHub Actions)

1. Open the repo on GitHub → **Actions** tab → **Travel Mode update**.
2. Tap **Run workflow**.
3. Fill in `match_number`, `team_a_goals`, `team_b_goals`; leave `status` as
   `played`. Add `notes` if you like.
4. **Run workflow**. After ~1 minute it has updated the score, recomputed
   everything (tables, simulations, prediction-vs-actual points), rebuilt the
   dashboard and committed `docs/index.html`.
5. Open your GitHub Pages URL (or reload it) to see the update.

This runs `scripts/update_score_override.py` under the hood.

---

## 3. Update MULTIPLE matches via a GitHub Issue comment

Post a comment on any issue containing a `/WK-SCORES` block:

```
/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
/END-WK-SCORES
```

The **Score comment update** workflow triggers automatically, validates the
whole batch, applies it, rebuilds the dashboard, commits the result, and replies
to the issue with a success/failure summary. If *any* row is invalid the whole
batch is rejected and nothing is applied — fix the comment and post again.

A ChatGPT helper prompt for converting raw results into this format is in
[github_issue_score_update_guide.md](github_issue_score_update_guide.md).

---

## 4. Update multiple matches via `scores_batch_update.csv`

If you prefer a file (e.g. editing on a laptop), fill in
`data/live/scores_batch_update.csv`:

```csv
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
```

Then run:

```bash
python scripts/apply_scores_batch_update.py
```

Same all-or-nothing validation: one bad row rejects the whole batch. A report is
written to `outputs/reports/scores_batch_update_report.md`. Changed rows are
stamped `source=batch_update`.

---

## 5. Fix a wrong score

Re-enter the **same match number** with the corrected goals — any of the three
paths (Actions, issue comment, batch CSV) overwrites the existing row. To clear a
result entirely, set its `status` to `scheduled` (this wipes the goals so the
match is treated as unplayed again).

---

## 6. Switch the active candidate

Edit `data/live/active_candidate.yml`. To use the auto-science candidate:

```yaml
active_candidate_dir: outputs/final_candidate_v2_auto_science
score_predictions_file: final_group_score_predictions_auto.csv
standing_predictions_file: final_group_standing_predictions_auto.csv
last8_predictions_file: final_last8_predictions_auto.csv
submission_pack_file: final_submission_pack_auto.csv
```

Re-run the pipeline (or trigger any Travel Mode workflow). All scoring, tables,
simulations and the dashboard immediately reflect the new candidate. If the file
is missing, Travel Mode falls back to `final_candidate_v1`. If a configured file
does not exist, the scripts fail clearly naming the missing path.

---

## 7. Open the dashboard on your phone

- **GitHub Pages:** enable Pages for the repo (`docs/` folder on the default
  branch). Your dashboard lives at the Pages URL and updates on every score
  entry. Bookmark it on your home screen.
- **Offline:** `outputs/live/mobile_dashboard.html` is fully self-contained (data
  embedded inline) — you can open it straight off the filesystem with no server.

---

## 8. Where to see predicted vs actual & points

On the dashboard:

- **🏅 Points earned** — total points, possible points, points missed, average
  per match, and outcome / goal-difference / exact-score hit rates, plus a
  points-by-group table.
- **🎯 Prediction vs actual** — per match: predicted score, actual score,
  outcome ✅/❌, goal difference ✅/❌, exact score ✅/❌, points earned, and an
  expandable scoring explanation.

Raw data: `outputs/live/prediction_vs_actual.csv` / `.json` and
`outputs/live/scoring_summary.csv` / `.json`. Report:
`outputs/reports/prediction_vs_actual_report.md`.

### How points are calculated (RULES_AND_SCORING.md §2)

- **Correct outcome** (home win / draw / away win): `6 × template odd` of the
  predicted outcome (Rate A for team_a win, Rate Draw for a draw, Rate B for
  team_b win).
- **+2** flat for exact goal difference, **+3** flat for exact score.
- Bonuses **only stack on a correct outcome**. A wrong outcome scores **0**.

---

## 9. The full local pipeline

```bash
python scripts/init_scores_override.py             # one time
# ... enter scores via any of the three paths ...
python scripts/update_live_tournament_state.py     # live group tables
python scripts/recalculate_live_simulations.py     # advancement probabilities (20k sims)
python scripts/score_predictions_vs_actuals.py     # predicted vs actual points
python scripts/build_mobile_dashboard.py           # mobile_dashboard.html + data JSON
cp outputs/live/mobile_dashboard.html docs/index.html
cp outputs/live/mobile_dashboard_data.json docs/mobile_dashboard_data.json
```

The two GitHub workflows (`travel_mode_update.yml`, `score_comment_update.yml`)
run exactly these steps for you and publish to `docs/`.

---

## 10. What NOT to do

- ❌ **Do not retrain** any model — Travel Mode is ingestion + scoring + dashboard
  only.
- ❌ **Do not hand-edit generated outputs** in `outputs/live/` or `docs/` — they
  are overwritten on every run. Edit only the inputs (`scores_override.csv` via
  the scripts, `scores_batch_update.csv`, `active_candidate.yml`).
- ❌ **Do not edit the frozen candidate folders** (`final_candidate_v1`,
  `final_candidate_v2_auto_science`).
- ❌ **No API / live-score fetch is needed or used** — scores are entered
  manually.
- ❌ **Do not edit raw data** under `data/reference/` or `data/raw/`.
