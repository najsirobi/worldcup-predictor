# GitHub Issue Score Update Guide

Update multiple World Cup 2026 scores at once — **group (1-72) and knockout
(73-104)** — by posting a **GitHub Issue comment**. No app, no API, no laptop
required.

---

## How it works

1. You post a comment containing a `/WK-SCORES … /END-WK-SCORES` block.
2. The **Score comment update** workflow (`.github/workflows/score_comment_update.yml`)
   fires automatically because the comment contains `/WK-SCORES`.
3. It parses the block, validates every row, and applies the whole batch
   atomically to `data/live/scores_override.csv` (`source=github_issue_comment`).
4. It recomputes live tables, simulations and prediction-vs-actual points,
   rebuilds the mobile dashboard, commits the outputs, and **replies to the
   issue** with a success/failure summary.

If **any** row is invalid, the whole update is rejected and nothing is applied —
the reply tells you exactly which rows failed so you can repost.

---

## The exact format

Post this as an issue comment (the surrounding text can be anything):

```
/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
/END-WK-SCORES
```

Rules:

- The header line must be **exactly**
  `match_number,team_a_goals,team_b_goals,status,notes`.
- `match_number` must be a real fixture number (**1–72** group, **73–104**
  knockout).
- `team_a_goals` / `team_b_goals` must be non-negative integers.
- `status` is one of `played | scheduled | postponed | void`;
  `played` requires both goals.
- `notes` is free text (used only for your own reference).
- Re-posting the same `match_number` with corrected goals **fixes** a wrong score.

### Knockout matches (73–104) and `advanced_team`

Knockout scores use the same block. You don't type team names — the dashboard
resolves each knockout match's participants from the results so far. If a
knockout match is **level after extra time** (decided on penalties), use the
**6-column** header and name the side that went through in `advanced_team` so the
next round can be filled:

```
/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes,advanced_team
73,1,1,played,R32 decided on penalties,Canada
74,2,0,played,R32,
/END-WK-SCORES
```

- The 5-column header still works for decisive knockout scores (winner = higher
  score) and for all group matches — it is fully backward compatible.
- With the 6-column header, `advanced_team` is the **last** field; `notes` may
  still contain commas.
- A **level** knockout score **requires** `advanced_team`, and it must be one of
  the two teams in that match.

---

## ChatGPT helper prompt (Dutch)

Paste this into ChatGPT, then paste the day's results where indicated. It returns
exactly the block you post as an issue comment:

```
Zet onderstaande WK-uitslagen om naar mijn GitHub score-update format.

Gebruik exact dit format:

/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
<match_number>,<team_a_goals>,<team_b_goals>,played,<team_a> v <team_b>
/END-WK-SCORES

Regels:
- Gebruik match_number uit mijn WK-template als je die weet.
- Als je match_number niet zeker weet, zet NEEDS_MATCH_NUMBER in plaats van een nummer.
- Gebruik alleen eindstanden.
- Geen extra uitleg, alleen het tekstblok.

Uitslagen:
[plak hier de wedstrijden]
```

If ChatGPT outputs `NEEDS_MATCH_NUMBER` for a row, look the fixture up in
`data/live/scores_override.csv` (or the dashboard's remaining-matches list),
replace it with the real number, and then post the comment.

---

## Troubleshooting

| Reply says | Cause | Fix |
|------------|-------|-----|
| `Missing start/end marker` | The `/WK-SCORES` or `/END-WK-SCORES` line is missing | Add both markers on their own lines |
| `Invalid header …` | Header line differs from the accepted columns | Copy the 5-column (or 6-column knockout) header exactly |
| `match N does not exist` | `match_number` not in 1–104 | Use a valid fixture number (1–72 group, 73–104 knockout) |
| `… is not an integer` | A goal cell isn't a whole number | Use integers like `0,1,2` |
| `status 'played' requires both goals` | A `played` row is missing a goal | Fill both goals or use `scheduled` |
| `… is level …; set advanced_team` | A level knockout score has no shoot-out winner | Use the 6-column header and set `advanced_team` |
| `advanced_team … must be one of …` | The named team isn't in that match | Use one of the match's two teams |
| `advanced_team is only valid for knockout` | `advanced_team` set on a group match (1–72) | Remove it for group matches |

A full machine-readable report is also written to
`outputs/reports/score_comment_ingestion_report.md` on every run.
