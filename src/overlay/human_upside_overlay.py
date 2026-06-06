"""Shared helpers for the Human Upside + Current-State + Chemistry analyst overlay.

This is a *limited analyst overlay*, not a model replacement. Nothing in this
module retrains models, touches submitted predictions, or modifies the frozen
``final_candidate_v2_auto_science`` artifacts. It only resolves the analyst seed
file against the official WC2026 squad list and recomputes the documented
overlay arithmetic so downstream reports stay internally consistent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEED_PATH = ROOT / "data" / "reference" / "wc2026_human_upside_overlay_seed.csv"
OFFICIAL_SQUADS_PATH = ROOT / "data" / "interim" / "wc2026_official_squads.parquet"
V2_SCORES_PATH = (
    ROOT
    / "outputs"
    / "final_candidate_v2_auto_science"
    / "final_group_score_predictions_auto.csv"
)

OVERLAY_LABEL = "Context only — not used in final prediction."

# Tokens that carry no identifying information on their own. A candidate made up
# only of these (e.g. "young starter") cannot be resolved to a squad player.
_FILLER_TOKENS = {"young", "starter", "none", "clear", "tbd", "unknown"}

# Fuzzy-match acceptance threshold on collapsed (space-free) name strings. Kept
# deliberately high so we only accept obvious spelling variants and never make a
# silent, low-confidence substitution.
_FUZZY_THRESHOLD = 0.86

VALID_CATEGORIES = {
    "elite_upside",
    "positive",
    "useful_context",
    "high_variance",
    "fragile",
}

KEY_ABSENCE_COLUMNS = [
    "key_absent_player",
    "key_absence_reason",
    "key_absence_date",
    "key_absence_timing_category",
    "key_absence_matches_since",
    "key_absence_was_regular_starter",
    "key_absence_role_importance_0_5",
    "key_absence_already_reflected_score_0_5",
    "key_absence_raw_penalty",
    "key_absence_residual_penalty",
    "key_absence_rationale",
    "key_absence_sources",
    "key_absence_needs_review",
]

VALID_KEY_ABSENCE_TIMING_CATEGORIES = {
    "no_absence",
    "long_absence_already_adapted",
    "medium_absence_partly_adapted",
    "late_absence",
    "selection_omission",
    "unknown",
}

DEFAULT_KEY_ABSENCE = {
    "key_absent_player": "",
    "key_absence_reason": "",
    "key_absence_date": "",
    "key_absence_timing_category": "no_absence",
    "key_absence_matches_since": "",
    "key_absence_was_regular_starter": "false",
    "key_absence_role_importance_0_5": 0,
    "key_absence_already_reflected_score_0_5": 0,
    "key_absence_raw_penalty": 0.0,
    "key_absence_residual_penalty": 0.0,
    "key_absence_rationale": "",
    "key_absence_sources": "",
    "key_absence_needs_review": False,
}


def normalize_text(value: object) -> str:
    """Accent-stripped, lower-case, punctuation-collapsed name key."""

    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _collapse(value: object) -> str:
    return normalize_text(value).replace(" ", "")


def team_key(value: object) -> str:
    """Normalized team key so squad/seed/prediction spellings join robustly."""

    return normalize_text(value)


@dataclass
class MatchResult:
    candidate_input: str
    selected_candidate: str | None
    found: bool
    method: str
    player_name: str | None = None
    display_name: str | None = None
    position: str | None = None
    age: float | None = None
    club: str | None = None
    note: str = ""
    alternatives_considered: list[str] = field(default_factory=list)


def load_seed(path: Path = SEED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["team"] = df["team"].astype(str).str.strip()
    for col, default in DEFAULT_KEY_ABSENCE.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].where(pd.notna(df[col]), default)
    return df


def load_official_squads(path: Path = OFFICIAL_SQUADS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["team_key"] = df["team"].map(team_key)
    return df


def _player_variants(row: pd.Series) -> list[str]:
    parts = [
        row.get("player_name", ""),
        f"{row.get('first_names', '')} {row.get('last_names', '')}",
        row.get("name_on_shirt", ""),
        row.get("last_names", ""),
        row.get("first_names", ""),
    ]
    variants = [normalize_text(p) for p in parts]
    return [v for v in variants if v]


def _player_tokens(row: pd.Series) -> set[str]:
    tokens: set[str] = set()
    for field_name in ("player_name", "first_names", "last_names", "name_on_shirt"):
        tokens.update(normalize_text(row.get(field_name, "")).split())
    return tokens


def _token_matches(cand_token: str, player_tokens: set[str]) -> bool:
    """A candidate token matches if it equals, or is a >=3-char prefix of (or is
    prefixed by), some player token. Lets diminutives resolve without inviting
    one-letter false positives."""

    for pt in player_tokens:
        if cand_token == pt:
            return True
        if len(cand_token) >= 3 and pt.startswith(cand_token):
            return True
        if len(pt) >= 3 and cand_token.startswith(pt):
            return True
    return False


def _score_candidate_against_player(cand_norm: str, row: pd.Series) -> tuple[int, str]:
    """Return (score, method) for one candidate against one squad player."""

    cand_collapsed = cand_norm.replace(" ", "")
    cand_tokens = [t for t in cand_norm.split() if t not in _FILLER_TOKENS]
    if not cand_tokens or not cand_collapsed:
        return 0, "none"

    variants = _player_variants(row)
    collapsed_variants = {v.replace(" ", "") for v in variants}
    player_tokens = _player_tokens(row)

    if cand_collapsed in collapsed_variants:
        return 100, "exact_collapsed"
    if set(cand_tokens) <= player_tokens:
        # Reward multi-token full-name agreement above single-token overlaps.
        return 60 + len(cand_tokens), "token_subset"
    if all(_token_matches(c, player_tokens) for c in cand_tokens):
        # Handles diminutives / prefixes, e.g. "Gio" -> "Giovanni", "Eli" -> "Elijah".
        return 55 + len(cand_tokens), "token_prefix"
    if any(cand_collapsed in cv and len(cand_collapsed) >= 5 for cv in collapsed_variants):
        return 40, "collapsed_substring"

    best_ratio = max(
        (SequenceMatcher(None, cand_collapsed, cv).ratio() for cv in collapsed_variants),
        default=0.0,
    )
    if best_ratio >= _FUZZY_THRESHOLD and len(cand_collapsed) >= 5:
        return int(round(20 * best_ratio)), "fuzzy"
    return 0, "none"


def _match_single_candidate(squad: pd.DataFrame, candidate: str) -> MatchResult | None:
    cand_norm = normalize_text(candidate)
    cand_tokens = [t for t in cand_norm.split() if t not in _FILLER_TOKENS]
    if not cand_tokens:
        return None  # filler-only candidate (e.g. "young starter")

    scored: list[tuple[int, str, pd.Series]] = []
    for _, row in squad.iterrows():
        score, method = _score_candidate_against_player(cand_norm, row)
        if score > 0:
            scored.append((score, method, row))
    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_method, best_row = scored[0]
    # Ambiguous if a second player ties the top score without an exact hit.
    if best_score < 100 and sum(1 for s, _, _ in scored if s == best_score) > 1:
        return None

    return MatchResult(
        candidate_input=candidate,
        selected_candidate=candidate.strip(),
        found=True,
        method=best_method,
        player_name=str(best_row.get("player_name", "")),
        display_name=_display_name(best_row),
        position=str(best_row.get("position", "")),
        age=_safe_float(best_row.get("age_on_2026_06_11")),
        club=str(best_row.get("club", "")),
    )


def _display_name(row: pd.Series) -> str:
    first = str(row.get("first_names", "") or "").strip()
    last = str(row.get("last_names", "") or "").strip()
    if first and last:
        # Title-case while preserving the accented squad spelling.
        return f"{first} {last.title()}" if last.isupper() else f"{first} {last}"
    return str(row.get("player_name", "") or "").strip()


def _safe_float(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # filter NaN


def resolve_candidate(squad: pd.DataFrame, candidate_field: str, *, role: str) -> MatchResult:
    """Resolve a possibly ``/``-separated candidate string to a squad player.

    Candidates are tried in listed (strongest-first) order; the first valid,
    unambiguous match wins. Non-first selections are reported in ``note`` so the
    choice is never silent.
    """

    raw_candidates = [c.strip() for c in str(candidate_field).split("/") if c.strip()]
    if not raw_candidates:
        return MatchResult(candidate_field, None, False, "none", note="no candidate supplied")

    alternatives = []
    for idx, candidate in enumerate(raw_candidates):
        result = _match_single_candidate(squad, candidate)
        if result is not None:
            result.candidate_input = candidate_field
            result.alternatives_considered = raw_candidates
            if idx > 0:
                skipped = ", ".join(raw_candidates[:idx])
                result.note = (
                    f"selected '{candidate}' over earlier candidate(s) [{skipped}] "
                    f"that were not found / ambiguous in the official squad"
                )
            elif result.method != "exact_collapsed":
                result.note = f"matched via {result.method} (non-exact spelling)"
            return result
        alternatives.append(candidate)

    return MatchResult(
        candidate_input=candidate_field,
        selected_candidate=raw_candidates[0],
        found=False,
        method="none",
        alternatives_considered=raw_candidates,
        note=(
            f"no {role} candidate matched the official squad "
            f"(tried: {', '.join(raw_candidates)})"
        ),
    )


# --- documented overlay arithmetic ---------------------------------------------------------


def compute_net_human_upside(row: pd.Series) -> int:
    return int(
        row["star_upside_score_0_5"]
        + row["talent_breakout_score_0_5"]
        - max(row["star_risk_score_0_5"], row["talent_risk_score_0_5"])
    )


def compute_final_overlay(row: pd.Series) -> float:
    return round(
        float(row["net_human_upside_score"])
        + float(row["current_state_adjustment"])
        + float(row["chemistry_adjustment"])
        + float(row.get("key_absence_residual_penalty", 0.0) or 0.0),
        4,
    )


def round_to_quarter(value: float) -> float:
    return round(float(value) * 4) / 4


def compute_key_absence_residual_penalty(row: pd.Series) -> float:
    """Residual absence penalty after discounting already-reflected information."""

    raw = max(-1.25, min(0.0, float(row.get("key_absence_raw_penalty", 0.0) or 0.0)))
    reflected = max(
        0.0,
        min(5.0, float(row.get("key_absence_already_reflected_score_0_5", 0) or 0)),
    )
    residual = round_to_quarter(raw * (1 - reflected / 5))
    return max(-1.25, min(0.0, residual))


def _candidate_parts(candidate_field: object) -> list[str]:
    return [c.strip() for c in str(candidate_field or "").split("/") if c.strip()]


def any_absent_player_in_squad(squad: pd.DataFrame, absent_player_field: object) -> MatchResult | None:
    """Return the first named absence that is actually present in the official squad."""

    for candidate in _candidate_parts(absent_player_field):
        result = resolve_candidate(squad, candidate, role="absence")
        if result.found:
            return result
    return None


def official_squad_gate_key_absence(row: pd.Series, squad: pd.DataFrame) -> dict:
    """Apply the hard official-squad gate and residual-penalty arithmetic.

    The seed can contain an analyst residual override, but only when the
    rationale is explicit. If a named absence is actually found in the official
    squad, the residual penalty is forced to 0.0 and the row is review-flagged.
    """

    out = {col: row.get(col, DEFAULT_KEY_ABSENCE[col]) for col in KEY_ABSENCE_COLUMNS}
    timing = str(out["key_absence_timing_category"] or "no_absence")
    if timing not in VALID_KEY_ABSENCE_TIMING_CATEGORIES:
        timing = "unknown"
    out["key_absence_timing_category"] = timing

    raw = max(-1.25, min(0.0, float(out["key_absence_raw_penalty"] or 0.0)))
    out["key_absence_raw_penalty"] = raw
    out["key_absence_role_importance_0_5"] = int(
        max(0, min(5, float(out["key_absence_role_importance_0_5"] or 0)))
    )
    out["key_absence_already_reflected_score_0_5"] = int(
        max(0, min(5, float(out["key_absence_already_reflected_score_0_5"] or 0)))
    )
    out["key_absence_needs_review"] = str(out.get("key_absence_needs_review", "")).strip().casefold() in {
        "true",
        "1",
        "yes",
    }

    player = str(out["key_absent_player"] or "").strip()
    if not player or timing == "no_absence":
        out.update({
            "key_absent_player": "",
            "key_absence_reason": "",
            "key_absence_date": "",
            "key_absence_timing_category": "no_absence",
            "key_absence_matches_since": "",
            "key_absence_was_regular_starter": "false",
            "key_absence_role_importance_0_5": 0,
            "key_absence_already_reflected_score_0_5": 0,
            "key_absence_raw_penalty": 0.0,
            "key_absence_residual_penalty": 0.0,
            "key_absence_rationale": "",
            "key_absence_sources": "",
            "key_absence_needs_review": False,
        })
        return out

    present = any_absent_player_in_squad(squad, player)
    if present is not None:
        rationale = str(out.get("key_absence_rationale") or "").strip()
        gate_note = (
            f"Official squad gate found {present.selected_candidate} in the squad "
            f"({present.player_name}); absence penalty disabled."
        )
        out["key_absence_residual_penalty"] = 0.0
        out["key_absence_needs_review"] = True
        out["key_absence_rationale"] = f"{rationale} {gate_note}".strip()
        return out

    computed = compute_key_absence_residual_penalty(pd.Series(out))
    seed_residual = float(out.get("key_absence_residual_penalty") or 0.0)
    rationale = str(out.get("key_absence_rationale") or "").casefold()
    override_allowed = bool(rationale) and any(
        token in rationale
        for token in ("override", "cap", "do not stack", "not stack", "default")
    )
    if abs(seed_residual - computed) <= 1e-9 or override_allowed:
        residual = max(-1.25, min(0.0, round_to_quarter(seed_residual)))
    else:
        residual = computed
    out["key_absence_residual_penalty"] = residual
    return out


def clamp_chemistry(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def clamp_current_state(value: float) -> float:
    return max(-2.0, min(2.0, float(value)))


def team_to_group_map(scores_path: Path = V2_SCORES_PATH) -> dict[str, str]:
    scores = pd.read_csv(scores_path)
    mapping: dict[str, str] = {}
    for _, r in scores.iterrows():
        mapping[str(r["team_a"])] = str(r["group"])
        mapping[str(r["team_b"])] = str(r["group"])
    return mapping
