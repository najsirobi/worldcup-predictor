# Score selection policy — dashboard note

**Why there is only one score to fill in per match, and why no "median of 5 runs" is needed.**

## The one number that matters

For every group-stage match there is exactly one score to submit:
`final_recommended_score`, surfaced in the dashboard and the fill-only export as
**"Score to fill in"**. Everything else on the row is audit context.

| Column | Dashboard label | Role |
| --- | --- | --- |
| `final_recommended_score` | **Score to fill in** | The score you submit. |
| `safe_score` | Safe alternative | Audit/alternative only. |
| `ev_score` | EV alternative | Audit/alternative only. |
| `auto_consensus_score` | Consensus/modal score | Audit/alternative only. |
| `auto_policy_decision` | Policy decision | Explains *why* the score above was selected. |

The safe, EV and consensus/modal columns are **not** four equally valid picks.
They are the inputs the policy already weighed before committing to a single
recommended score.

## The model already compared the alternatives

The deterministic science-only selector compared, per match:

- the **safe** score,
- the **EV** (expected-value) score,
- the **consensus / modal** score across candidate sources, and
- the **expected-FIF8A-points** of each candidate.

The selection rule:

1. If all scientific sources agree, take that score.
2. Otherwise take the modal score across candidate sources and deterministic
   seed views.
3. Break ties by highest average expected FIF8A points, then by `safe_score`.
4. Keep the EV score over the safe score only when its expected-points uplift
   clears a strict threshold and the row is neither high-variance nor
   contrarian.
5. Otherwise keep `safe_score`. No row ever needs manual input.

## Stability — five seed checks

Five deterministic seed views were run as a stability check. The candidate
sources and the resulting selections were **deterministic / stable** across all
five seeds, so the recommended score does not move from run to run.

## Why the safe score won, and what that means for "median"

The final policy selected `safe_score` unless an EV override passed the strict
thresholds. Across the 72 matches:

- **EV overrides accepted = 0.**
- The safe score was kept for every match.

Because the output is deterministic and stable across seeds, there is no
run-to-run spread to average over. A **"median of 5 runs" adds nothing** — five
runs produce the same numbers.

## If stochastic score samples are introduced later

If a future version draws stochastic score *samples* (rather than deterministic
candidates), do **not** take a naive median of the sampled scorelines — the
median of scorelines is not itself a sensible scoreline. Instead prefer:

- the **modal score** (most frequently sampled scoreline), or
- the score with the **highest expected FIF8A points**,

which is exactly what the current deterministic policy already optimises for.
