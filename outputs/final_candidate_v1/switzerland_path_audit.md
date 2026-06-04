# Switzerland Path Audit

Generated: 2026-06-04
Author: Claude QA layer (Phase 7C)
Scope: explain why Switzerland is recommended as a QF and SF team. Read-only. No override applied.

---

## 1. Why Switzerland is recommended

The last-8 selection policy is `safe_highest_probability`: for each stage it picks the teams with
the highest simulated probability of *reaching* that stage. Switzerland ranks **2nd of 48** for both:

- **Quarter-finalist**: `p_reach_qf = 0.503` (rank 2, behind Spain 0.652)
- **Semi-finalist**: `p_reach_sf = 0.3084` (rank 2, behind Spain 0.516)

So Switzerland is selected purely because the simulation gives it the 2nd-highest reach
probability for those two rounds — not because of any manual preference.

---

## 2. Switzerland's full probability profile

| Metric | Value | Rank (of 48) |
|---|--:|--:|
| Group | B | — |
| p_win_group | 0.6116 | high |
| p_finish_top2 | 0.8936 | high |
| p_advance_group | 0.9732 | high |
| p_reach_r16 | 0.7500 | 2nd |
| p_reach_qf | 0.5030 | 2nd |
| p_reach_sf | 0.3084 | 2nd |
| p_reach_final | 0.1810 | ~4th |
| p_win_world_cup | 0.0950 | ~5th |

Group-stage detail (Group B): expected points **6.59**, expected GD **+4.18**, p_finish_1st **0.606**.
Group B opponents are Canada (5.43 xpts), Bosnia and Herzegovina (3.33), Qatar (1.44) — a weak group.

---

## 3. Bracket path (from official R32 mapping + progression)

Switzerland's group feeds these knockout slots:
- **1B → R32 match 85** vs *best 3rd from E/F/G/I/J* (a third-placed team).
- **2B → R32 match 73** vs *2A* (another group runner-up).

Both possible R32 opponents are beatable: the group winner draws one of the easiest opponent
classes in the bracket (a best-placed third), and the runner-up draws another runner-up.

### Modal (most-likely) knockout path — analytic trace

The full simulation stores only aggregate per-round probabilities, not per-simulation paths.
The table below is a lightweight **analytic modal trace**: each team is placed in its most-likely
group rank and meets the strongest modal opponent from each feeding slot. It is illustrative of the
central path, not a frequency distribution.

| Team | R32 | R16 | QF | SF | Final | Modal exit |
|---|---|---|---|---|---|---|
| **Switzerland** | Senegal (3rd) | Portugal | Argentina | — | — | loses QF |
| Spain | Austria | Croatia | Belgium | Netherlands | Argentina | reaches Final |
| Argentina | Uruguay | Türkiye | Switzerland | England | Spain | reaches Final |
| England | Senegal (3rd) | Mexico | Brazil | Argentina | — | loses SF |

Switzerland's modal path: **soft best-third in R32 → beatable R16 (Portugal-tier) → eliminated by
an elite team (Argentina) at QF.** This matches the numbers exactly: high reach to QF, then the
probabilities collapse from the SF onward.

---

## 4. What drives the high QF/SF probability

| Driver | Contribution | Notes |
|---|---|---|
| **Group strength (weak Group B)** | **Primary** | p_advance 0.97, p_win_group 0.61 — Switzerland is the clear group favourite over Canada/Bosnia/Qatar. |
| **Favourable early bracket path** | **Primary** | 1B draws a best-third in R32, then a non-elite R16 opponent. Two soft knockout rounds before meeting an elite side. |
| **Third-place assignment** | Contributing | Switzerland (as 1B) is scheduled against a 3rd-placed team in R32 — one of the easiest R32 opponent classes. Annexe C QA passed, so this assignment is valid, not an error. |
| **Model strength** | Minor caveat | Phase 4.5 rates Switzerland strongly (6.59 xpts). This is somewhat above betting-market consensus, but within reason given Switzerland's FIFA-ranking strength. It is a model-rating question, not a bracket/simulation defect. |
| **Artefact?** | **No** | Deep-run probabilities are appropriately modest (final 0.18, win 0.095, both well below Spain). Monotonicity holds; probability-mass checks pass exactly. The simulation does not over-project Switzerland into the final/winner. |

---

## 5. Verdict: **DEFENSIBLE**

Switzerland's QF/SF recommendation is a defensible consequence of (1) a weak Group B and
(2) a favourable early bracket draw (group winner faces a best-third in R32 and a beatable R16
opponent). The simulation correctly discounts Switzerland once it reaches elite opposition — its
final and winner probabilities are modest and rank it 4th–5th, not 2nd. There is no sign of a
simulation artefact, double-counting, or bracket-mapping error.

The only mild caveat is that the Phase 4.5 model rates Switzerland's raw strength a little above
market consensus; this affects the magnitude but not the structural conclusion. Per Phase 7C
instructions, **no manual override is applied** — Switzerland remains as the simulation places it.
