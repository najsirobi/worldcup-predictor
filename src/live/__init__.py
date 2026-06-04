"""Travel Mode live tournament tracking.

Self-contained, offline-only helpers that take manually entered group-stage
scores and recompute live group tables and advancement probabilities on top of
the frozen ``final_candidate_v1`` predictions.

Nothing in this package retrains a model, fetches an API, or modifies the
baseline predictions. Unplayed matches fall back to the published recommended
scorelines so the live state degrades gracefully to the pre-tournament picks
when no scores have been entered yet.
"""
