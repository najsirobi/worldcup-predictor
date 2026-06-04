# Final Safe vs EV Policy

- Safe score is the default final score.
- EV score is used only if the existing decision table marks it as a non-suspicious, meaningful-uplift choice.
- High-variance contrarian picks remain optional/manual-review, not default.
- Manual-review flags are preserved in `final_group_score_predictions.csv`.
- Safe-vs-EV disagreements: **27**
- Manual-review group matches: **21**
- In this run, manual-review rows retain the safe score as the machine default until a human overrides them.
