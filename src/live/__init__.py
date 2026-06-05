"""Travel Mode live tournament tracking.

Self-contained, offline-only helpers that take manually entered group-stage
scores and recompute live group tables and advancement probabilities on top of
the frozen active candidate predictions.

Nothing in this package retrains a model, fetches an API, or modifies the
submitted predictions. Unplayed matches fall back to the submitted recommended
scorelines so the live state degrades gracefully to the pre-tournament picks
when no scores have been entered yet.
"""
