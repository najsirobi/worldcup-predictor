# FIF8A World Cup 2026 — Rules, Group Round & Scoring

> Summary extracted from the two source files in this folder:
> - `FIF8A World Cup 2026 - Player guide.pdf` (authoritative rules & points)
> - `FIF8A World Cup 2026_Player_template.xlsx` (groups, fixtures, betting odds/“staffels”)
>
> Purpose: define **exactly what a model must predict and how each prediction is
> scored**, so the prediction model can be built to maximise expected points.
> Where the PDF and the Excel disagree, the **PDF is treated as authoritative** and
> the discrepancy is flagged in [§7](#7-source-discrepancies--assumptions).

---

## 1. Tournament format (what gets predicted)

48 teams, 12 groups of 4 (**A–L**), played as a group stage followed by a knockout
bracket (the new 48-team format):

```
Group Stage (12 groups × 6 matches = 72 matches)
        │  top 2 per group (24) + 8 best third-placed teams (8) = 32
        ▼
Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final (+ 3rd-place match)
```

- **Group stage:** 11 Jun 2026 → 28 Jun 2026 (per the template fixture dates).
- **Best third-placed teams:** the 8 best of the 12 third-placed teams advance
  (handled in the workbook’s `Best 3rds calculation` tab).
- Predictions are made in **two blocks**, both submitted *before* the group stage:
  1. a predicted score for **every** group-stage match, and
  2. the **last-standing teams** (who reaches QF / SF / Final / wins) — see [§4](#4-tournament-progression-scoring-last-8-block).

---

## 2. Group-stage scoring — per match

For each of the 72 group matches you predict an **exact score** (e.g. `2–1`). Points:

| What you got right | Points | Odd-multiplied? |
|---|---|---|
| **Correct outcome** (home win / draw / away win) | **6 × betting odd** of that outcome | ✅ yes |
| **Exact goal difference** (e.g. you said 1–0, result 2–1) | **+2** | ❌ flat |
| **Exact score** (e.g. you said 2–1, result 2–1) | **+3** | ❌ flat |

- The three awards **stack**. A perfect exact-score prediction earns
  `6 × odd + 2 + 3`.
- **The base outcome reward is identical for a win, a draw or a loss — always 6.**
  Predicting the harder result is *not* rewarded by a bigger base; the **only**
  thing that changes the outcome points is the **odd** of the result you picked.
- The “6” is multiplied by the match’s **betting odd / rate** for the predicted
  outcome (see [§5](#5-betting-odds--staffels-kansen-dat-een-team-wint)). Higher
  odd = bigger underdog = more points if you call it correctly.
- With an odd of 1.0 the per-match maximum is **11 points** (this is the
  “11 points per game” figure in the Excel `Rules` tab); with real odds the
  ceiling is higher for upset calls.

> **Note:** goal difference and exact score are only meaningful on top of a correct
> outcome. If the outcome is wrong, the score/difference bonuses do not apply.

## 3. Group-stage scoring — per group (global performance)

Awarded once per group, based on the final group table your predictions produce:

| What you got right | Points |
|---|---|
| **Correct top 2 teams** of the group (the 2 qualifiers, in **any** order) | **30** |
| **Exact full group standing** (positions 1, 2, 3, 4 in exact order) | **+60** |

- These **stack**: a perfectly ordered group = `30 + 60` = **90 points**.
- Getting the exact standing necessarily implies getting the top 2, so the exact
  standing pays the full 90.
- **These points are flat — they are NOT multiplied by any odd / FIFA ranking.**
  The world ranking only feeds the per-match `6 × odd` term (§2 / §4b). The group
  bonuses, and the progression bonuses (§4), are fixed amounts that simply reward
  predicting **what actually happens**. ⇒ maximise them by predicting the **most
  probable** qualifiers and ordering (see [§5 practical recommendation](#practical-recommendation-what-to-fill-in)).

---

## 4. Tournament-progression scoring (“Last 8” block)

Predicted **at the same time as the group-stage predictions** (workbook tab
`Last 8 teams predictions`). You name the teams that reach each late stage:

| Prediction | Slots | Points each | Block max |
|---|---|---|---|
| Team reaches the **quarter-finals** | 8 (Q1–Q8) | **20** | 160 |
| Team reaches the **semi-finals** | 4 (S1–S4) | **40** | 160 |
| Team reaches the **final** | 2 (F1–F2) | **60** | 120 |
| **Winner** of the World Cup | 1 (W) | **100** | 100 |

Maximum from this block = **540 points**.

## 4b. Knockout-stage match scoring (filled in later via MS Forms)

During the knockout rounds you additionally predict individual match scores. Per
knockout match:

| What you got right | Points | Odd-multiplied? |
|---|---|---|
| **Correct qualified team** (who advances) | **6 × betting odd** | ✅ yes |
| **Exact score** (at end of extra time, if any) | **+2** | ❌ flat |
| **Correct guess the game goes to a penalty shoot-out** | **+2** | ❌ flat |

> The knockout *match* predictions are entered along the way (more flexible timing);
> only the **Last 8** progression block (§4) is locked in up front with the group
> predictions.

---

## 5. Betting odds / “staffels” (kansen dat een team wint)

Every match in the template carries three odds — **Rate A** (Team A wins),
**Rate Draw**, **Rate B** (Team B wins). These are the multipliers on the 6
outcome points. They are derived, **not** bookmaker-sourced:

1. Take each team’s **FIFA ranking points** (workbook `FIFA rankings` tab,
   last updated 2026-05-21).
2. Convert the **points gap** into an **expected goals** value per team
   (baseline ≈ **1.4 goals per team per match**, scaled by the ranking gap; the
   gap is normalised against “4× the points gap between the top and bottom
   ranked team”).
3. Feed the two expected-goals values into a **Poisson distribution** over each
   team’s goal count (0–10) and sum the joint probabilities into
   `P(A win)`, `P(draw)`, `P(B win)` (these sum to 1).
4. **Odd = 1 / probability** of that outcome.

**Worked example from the guide — Norway vs France:**

| Outcome | Probability | Odd (= 1/p) |
|---|---|---|
| France win | 0.5823 | 1.72 |
| Draw | 0.2197 | 4.55 |
| Norway win | 0.1971 | 5.07 |

So a lower odd = the model thinks that outcome is likely (few points); a higher
odd = unlikely (many points if you call it).

### Modelling implication (what to actually predict)

Because outcome points are `6 × (1 / p_template)`, the **expected** outcome points
for any pick equal `6 × p_yourModel / p_template`:

- If you trust the template’s probabilities, every outcome pick has the same
  expected value (≈6) — so picking the favourite is *not* inherently better.
- **Edge therefore comes from three places:**
  1. **Disagreeing with the template** — predict an outcome where *your* model’s
     probability exceeds the template’s implied probability (i.e. find
     mispriced odds / value upsets).
  2. **Exact scoreline & goal-difference bonuses** (+3 / +2) — these are **flat**
     (not odd-weighted), so accurate *scoreline* prediction is pure additive value
     on every game.
  3. **Group-table (§3, up to 90/group) and progression (§4, up to 540) bonuses**
     — large, lump-sum, and reward getting the *qualifiers, ordering, and deep-run
     teams* right rather than individual scorelines.

A points-maximising model should therefore output, per match, a full
**scoreline probability distribution** (e.g. from a bivariate/independent Poisson
goals model), then choose the predicted score that **maximises expected points**
given the match odds — and separately optimise the group-standing and last-8
selections.

### Practical recommendation: what to fill in

**Default to the most probable outcome almost everywhere.** Reasoning:

- **The big points are flat (not odd-weighted).** Group tables pay up to
  `12 × 90 = 1,080` and progression up to `540`, versus only `~72 × 6 = ~432` of
  base outcome points. Those large blocks are **independent of the FIFA ranking /
  odds** — they reward predicting *what actually happens*, so they are maximised
  by picking the **most likely** qualifiers, group order, and deep-run teams.
- **The odd-weighted term is expectation-neutral.** Because the base reward is the
  same 6 for any result and the odd is `1 / p_template`, the expected outcome
  points are ≈6 *whatever* you pick (favourite or upset). Chasing high odds does
  **not** raise your expected score — it only raises variance. So there is no
  expected-value reason to be “pulled” toward underdogs by the odds.
- **Therefore: do not let the odds tempt you into upset picks.** Fill in the most
  probable result for each match (and the most probable *scoreline* for the +2/+3),
  and the most probable qualifiers/finalists/winner for the flat bonuses.
- **The only justified exception** is a match where *your own* model gives a
  result a meaningfully higher probability than the template’s odds imply (a
  genuinely mispriced game). Only then does deviating add expected value — and the
  template’s odds come from the same FIFA-ranking Poisson model, so such edges
  will be rare and should be deliberate, not odds-chasing.
- **Caveat (winning the pool vs. expected points):** if the aim is to *finish
  first* among many players who all pick favourites, a few contrarian high-odd
  calls can decorrelate your entry and help you win on rank — but that is a
  higher-variance bet, separate from maximising expected points, and not the
  recommended baseline.

**Bottom line:** yes — don’t be swayed by the odds for the group-standing and
progression points; fill in the most probable outcomes. Reserve odd-driven
deviations for the rare games where the model genuinely disagrees with the FIFA
ranking.

---

## 6. The group round — teams & fixtures (with odds)

**Group assignments** below are taken from the template’s `Base qualification`
tab (the seeded/assumed draw used in this workbook). Team A / Team B are the
fixture designations from the template; **Rate A / Rate Draw / Rate B** are the
odds defined in [§5](#5-betting-odds--staffels-kansen-dat-een-team-wint).
`#` is the template’s match number.

### Groups (4 teams each, seeds 1–4)

| Grp | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|
| A | Mexico | South Africa | Korea Republic | Czechia |
| B | Canada | Bosnia and Herzegovina | Qatar | Switzerland |
| C | Brazil | Morocco | Haiti | Scotland |
| D | USA | Paraguay | Australia | Türkiye |
| E | Germany | Curaçao | Côte d'Ivoire | Ecuador |
| F | Netherlands | Japan | Sweden | Tunisia |
| G | Belgium | Egypt | IR Iran | New Zealand |
| H | Spain | Cabo Verde | Saudi Arabia | Uruguay |
| I | France | Senegal | Iraq | Norway |
| J | Argentina | Algeria | Austria | Jordan |
| K | Portugal | Congo DR | Uzbekistan | Colombia |
| L | England | Croatia | Ghana | Panama |

### Fixtures & odds (72 matches)
<!-- Tables generated from the template; Rate A = Team A win, Rate B = Team B win. -->
#### Group A
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 1 | 2026-06-11 | Mexico | 1.87 | 4.30 | 4.30 | South Africa |
| 2 | 2026-06-12 | Korea Republic | 2.34 | 4.00 | 3.11 | Czechia |
| 25 | 2026-06-19 | Mexico | 2.32 | 4.00 | 3.14 | Korea Republic |
| 28 | 2026-06-18 | Czechia | 2.39 | 3.98 | 3.03 | South Africa |
| 53 | 2026-06-25 | Czechia | 3.71 | 4.13 | 2.05 | Mexico |
| 54 | 2026-06-25 | South Africa | 3.56 | 4.09 | 2.11 | Korea Republic |

#### Group B
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 3 | 2026-06-12 | Canada | 2.08 | 4.11 | 3.64 | Bosnia and Herzegovina |
| 8 | 2026-06-13 | Qatar | 3.82 | 4.16 | 2.01 | Switzerland |
| 26 | 2026-06-19 | Canada | 2.29 | 4.01 | 3.19 | Qatar |
| 27 | 2026-06-18 | Switzerland | 1.85 | 4.33 | 4.41 | Bosnia and Herzegovina |
| 51 | 2026-06-24 | Switzerland | 2.32 | 4.00 | 3.14 | Canada |
| 52 | 2026-06-24 | Bosnia and Herzegovina | 3.01 | 3.98 | 2.40 | Qatar |

#### Group C
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 5 | 2026-06-14 | Brazil | 2.65 | 3.96 | 2.70 | Morocco |
| 7 | 2026-06-14 | Haiti | 3.91 | 4.18 | 1.98 | Scotland |
| 29 | 2026-06-20 | Brazil | 1.49 | 5.30 | 7.21 | Haiti |
| 30 | 2026-06-20 | Scotland | 4.35 | 4.32 | 1.86 | Morocco |
| 49 | 2026-06-25 | Scotland | 4.40 | 4.33 | 1.85 | Brazil |
| 50 | 2026-06-25 | Morocco | 1.50 | 5.27 | 7.11 | Haiti |

#### Group D
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 4 | 2026-06-13 | USA | 2.08 | 4.11 | 3.63 | Paraguay |
| 6 | 2026-06-14 | Australia | 2.76 | 3.96 | 2.60 | Türkiye |
| 31 | 2026-06-19 | USA | 2.32 | 4.00 | 3.14 | Australia |
| 32 | 2026-06-20 | Türkiye | 2.31 | 4.00 | 3.16 | Paraguay |
| 59 | 2026-06-26 | Türkiye | 3.04 | 3.98 | 2.38 | USA |
| 60 | 2026-06-26 | Paraguay | 3.06 | 3.99 | 2.37 | Australia |

#### Group E
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 9 | 2026-06-14 | Germany | 1.54 | 5.09 | 6.61 | Curaçao |
| 10 | 2026-06-15 | Côte d'Ivoire | 2.97 | 3.98 | 2.43 | Ecuador |
| 33 | 2026-06-20 | Germany | 2.00 | 4.16 | 3.84 | Côte d'Ivoire |
| 34 | 2026-06-21 | Ecuador | 1.77 | 4.45 | 4.78 | Curaçao |
| 55 | 2026-06-25 | Ecuador | 3.40 | 4.05 | 2.18 | Germany |
| 56 | 2026-06-25 | Curaçao | 4.18 | 4.26 | 1.90 | Côte d'Ivoire |

#### Group F
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 11 | 2026-06-14 | Netherlands | 2.30 | 4.01 | 3.17 | Japan |
| 12 | 2026-06-15 | Sweden | 2.54 | 3.96 | 2.82 | Tunisia |
| 35 | 2026-06-20 | Netherlands | 1.89 | 4.27 | 4.22 | Sweden |
| 36 | 2026-06-21 | Tunisia | 3.69 | 4.12 | 2.06 | Japan |
| 57 | 2026-06-26 | Tunisia | 4.52 | 4.37 | 1.82 | Netherlands |
| 58 | 2026-06-26 | Japan | 2.15 | 4.07 | 3.47 | Sweden |

#### Group G
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 15 | 2026-06-15 | Belgium | 2.07 | 4.11 | 3.65 | Egypt |
| 16 | 2026-06-16 | IR Iran | 1.70 | 4.58 | 5.16 | New Zealand |
| 39 | 2026-06-21 | Belgium | 2.23 | 4.03 | 3.30 | IR Iran |
| 40 | 2026-06-22 | New Zealand | 4.59 | 4.39 | 1.81 | Egypt |
| 63 | 2026-06-27 | New Zealand | 6.91 | 5.20 | 1.51 | Belgium |
| 64 | 2026-06-27 | Egypt | 2.92 | 3.97 | 2.46 | IR Iran |

#### Group H
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 13 | 2026-06-15 | Spain | 1.44 | 5.60 | 8.05 | Cabo Verde |
| 14 | 2026-06-16 | Saudi Arabia | 4.30 | 4.30 | 1.87 | Uruguay |
| 37 | 2026-06-21 | Spain | 1.51 | 5.21 | 6.94 | Saudi Arabia |
| 38 | 2026-06-22 | Uruguay | 1.75 | 4.48 | 4.85 | Cabo Verde |
| 65 | 2026-06-27 | Uruguay | 3.89 | 4.18 | 1.99 | Spain |
| 66 | 2026-06-27 | Cabo Verde | 2.94 | 3.97 | 2.45 | Saudi Arabia |

#### Group I
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 17 | 2026-06-16 | France | 2.03 | 4.14 | 3.77 | Senegal |
| 18 | 2026-06-17 | Iraq | 3.21 | 4.01 | 2.28 | Norway |
| 41 | 2026-06-22 | France | 1.55 | 5.06 | 6.52 | Iraq |
| 42 | 2026-06-23 | Norway | 3.42 | 4.06 | 2.17 | Senegal |
| 61 | 2026-06-26 | Norway | 5.07 | 4.55 | 1.72 | France |
| 62 | 2026-06-26 | Senegal | 1.89 | 4.27 | 4.21 | Iraq |

#### Group J
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 19 | 2026-06-17 | Argentina | 1.75 | 4.49 | 4.89 | Algeria |
| 20 | 2026-06-17 | Austria | 1.99 | 4.17 | 3.88 | Jordan |
| 43 | 2026-06-22 | Argentina | 1.81 | 4.39 | 4.58 | Austria |
| 44 | 2026-06-23 | Jordan | 3.66 | 4.11 | 2.07 | Algeria |
| 69 | 2026-06-28 | Jordan | 7.48 | 5.40 | 1.47 | Argentina |
| 70 | 2026-06-28 | Algeria | 2.81 | 3.96 | 2.55 | Austria |

#### Group K
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 23 | 2026-06-17 | Portugal | 1.80 | 4.40 | 4.63 | Congo DR |
| 24 | 2026-06-18 | Uzbekistan | 4.09 | 4.23 | 1.93 | Colombia |
| 47 | 2026-06-23 | Portugal | 1.77 | 4.45 | 4.76 | Uzbekistan |
| 48 | 2026-06-24 | Colombia | 1.96 | 4.20 | 3.98 | Congo DR |
| 71 | 2026-06-28 | Colombia | 3.02 | 3.98 | 2.39 | Portugal |
| 72 | 2026-06-28 | Congo DR | 2.62 | 3.96 | 2.74 | Uzbekistan |

#### Group L
| # | Date | Home (Team A) | Rate A | Rate Draw | Rate B | Away (Team B) |
|---|------|------|-------:|----------:|-------:|------|
| 21 | 2026-06-17 | England | 2.26 | 4.02 | 3.24 | Croatia |
| 22 | 2026-06-18 | Ghana | 3.82 | 4.16 | 2.01 | Panama |
| 45 | 2026-06-23 | England | 1.48 | 5.37 | 7.41 | Ghana |
| 46 | 2026-06-24 | Panama | 3.68 | 4.12 | 2.06 | Croatia |
| 67 | 2026-06-27 | Panama | 4.62 | 4.40 | 1.80 | England |
| 68 | 2026-06-27 | Croatia | 1.64 | 4.74 | 5.63 | Ghana |

---

## 7. Source discrepancies & assumptions

1. **Group-standing bonus wording.** The Excel `Rules` tab says “60 *additional*
   points … *ie you win 15 points in total*”. The “15 in total” is a stale
   leftover from an earlier scoring version; the PDF (authoritative) lists **30**
   (top 2) and **60** (exact standing). This summary uses **30 / +60 (= 90 max)**.
2. **Final & Winner points.** The Excel `Rules` tab only lists QF (20) and SF (40).
   The PDF adds **Final = 60** and **Winner = 100**, matching the `Last 8 teams
   predictions` tab’s F1/F2 and W slots. These are included here.
3. **Knockout match scoring** (correct team `6×odd`, exact score `+2`, shoot-out
   `+2`) comes from the PDF; it is not in the Excel `Rules` tab.
4. **Group assignments / teams** are the template’s assumed/seeded draw as of the
   file date (FIFA rankings updated 2026-05-21). Re-verify against the official
   final draw before locking in predictions.
5. **“Home/Away”** is only the template’s Team A / Team B fixture designation;
   World Cup group games are at neutral/host venues, so no home advantage is
   implied by the labels (the odds are FIFA-ranking based only).

---

## 8. Quick reference — point values

| Stage | Prediction | Points |
|---|---|---|
| Group match | Correct outcome | 6 × odd |
| Group match | Exact goal difference | +2 |
| Group match | Exact score | +3 |
| Group (per group) | Correct top 2 (any order) | 30 |
| Group (per group) | Exact standing 1–4 | +60 |
| Knockout match | Correct qualified team | 6 × odd |
| Knockout match | Exact score (incl. extra time) | +2 |
| Knockout match | Correct penalty shoot-out call | +2 |
| Progression | Team into quarter-finals (×8) | 20 each |
| Progression | Team into semi-finals (×4) | 40 each |
| Progression | Team into final (×2) | 60 each |
| Progression | Correct champion | 100 |
