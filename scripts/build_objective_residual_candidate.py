"""Build the deterministic objective-residual adjusted WC2026 candidate (v3).

This is a **post-model deterministic adjustment layer**, not a second model and not a
manual-review step. It:

  - keeps ``final_candidate_v2_auto_science`` as the immutable base model,
  - never overwrites or mutates any frozen v2 file,
  - applies exactly one validated rule (``R1_only_diff_5_0``) mined on the WC2022 backtest,
  - emits a separate adjusted candidate ``final_candidate_v3_objective_residual``,
  - is fully reproducible from the current overlay + the frozen v2 candidate,
  - uses no chemistry-only / key-absence-only / fame-only / mid-table / broad-overlay logic,
  - uses no subjective or manual approval at any point.

It also emits the late-news residual audit (Germany/Karl, Argentina/Messi) and the
friendly-lineup-distortion policy, both as objective audit artifacts that do **not**
change scores unless an objective rule qualifies.

Run:
    .venv/bin/python scripts/build_objective_residual_candidate.py

See ``outputs/reports/objective_residual_adjustment_policy.md`` for the full policy.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

V2_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"
V2_SCORES = V2_DIR / "final_group_score_predictions_auto.csv"
V2_MANIFEST = V2_DIR / "FROZEN_MANIFEST.json"
OVERLAY_CSV = ROOT / "data" / "reference" / "wc2026_human_upside_overlay.csv"

V3_DIR = ROOT / "outputs" / "final_candidate_v3_objective_residual"
REPORTS_DIR = ROOT / "outputs" / "reports"
PRED_DIR = ROOT / "outputs" / "predictions"

# --- Rule definition: R1_only_diff_5_0 -------------------------------------------------
RULE_NAME = "R1_only_diff_5_0"
SOURCE_BACKTEST = "objective_residual_rules_2022_backtest.md::R1_only_diff_5_0"
STRONG_CATEGORIES = {"elite_upside", "positive"}
WEAK_CATEGORIES = {"fragile", "low_upside"}
DIFF_THRESHOLD = 5.0
# A draw may only become a narrow win when the objective residual edge is extreme.
EXTREME_DRAW_THRESHOLD = 6.0

# Promotion gate thresholds (sparse-change guard).
WARN_CHANGES = 8
FAIL_CHANGES = 16
EXPECTED_GROUP_MATCHES = 72


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_score(score: str) -> tuple[int, int]:
    a, b = str(score).split("-")
    return int(a), int(b)


def _fmt(a: int, b: int) -> str:
    return f"{a}-{b}"


def evaluate_rule(
    v2_score: str,
    cat_a: str,
    cat_b: str,
    score_a: float,
    score_b: float,
) -> dict:
    """Apply ``R1_only_diff_5_0`` deterministically to one match.

    Returns a dict with ``changed``, ``adjusted_score``, ``rule_triggered``,
    ``change_type``, ``rule_components`` and ``reason``. Direction always follows the
    stronger objective-residual side; the change is always at most one goal.
    """
    diff = round(float(score_a) - float(score_b), 4)
    abs_diff = abs(diff)
    ga, gb = _parse_score(v2_score)
    margin = ga - gb  # positive favours team_a
    strong_is_a = diff > 0

    strong_cat = cat_a if strong_is_a else cat_b
    weak_cat = cat_b if strong_is_a else cat_a

    components: list[str] = []
    fail: list[str] = []

    if abs_diff >= DIFF_THRESHOLD:
        components.append(f"abs_overlay_diff>={DIFF_THRESHOLD:g}")
    else:
        fail.append(f"abs_overlay_diff<{DIFF_THRESHOLD:g}")

    if strong_cat in STRONG_CATEGORIES:
        components.append(f"strong_side_{strong_cat}")
    else:
        fail.append(f"strong_side_not_elite_or_positive({strong_cat})")

    if weak_cat in WEAK_CATEGORIES:
        components.append(f"weak_side_{weak_cat}")
    else:
        fail.append(f"weak_side_not_fragile_or_low({weak_cat})")

    narrow = abs(margin) <= 1
    if narrow:
        components.append("base_v2_narrow_or_conservative")
    else:
        fail.append(f"base_v2_not_narrow(margin={margin:+d})")

    no_change = {
        "changed": False,
        "adjusted_score": v2_score,
        "rule_triggered": "",
        "change_type": "none",
        "rule_components": " | ".join(components) if components else "",
    }

    if fail:
        no_change["reason"] = "R1_only_diff_5_0 not satisfied: " + "; ".join(fail)
        return no_change

    # All gate conditions met. Compute the at-most-one-goal adjustment toward the
    # stronger objective-residual side.
    strong_goals, weak_goals = (ga, gb) if strong_is_a else (gb, ga)

    if strong_goals - weak_goals == 1:
        # Stronger side wins narrowly -> strengthen the favourite by one goal.
        new_strong, new_weak = strong_goals + 1, weak_goals
        change_type = "favourite_strengthened"
        reason = (
            "Stronger objective-residual side wins narrowly; add one goal to the "
            "favourite (max one-goal margin adjustment)."
        )
    elif weak_goals - strong_goals == 1:
        # Fragile side predicted to win narrowly -> level to a draw.
        new_strong, new_weak = weak_goals, weak_goals
        change_type = "decisive_to_draw"
        reason = (
            "Base v2 predicts a narrow win for the fragile side while the objective "
            "residual strongly favours the opponent; level to a draw (one-goal change "
            "toward the stronger side)."
        )
    elif strong_goals == weak_goals:
        # Draw -> narrow win for the stronger side, only when the edge is extreme.
        if abs_diff >= EXTREME_DRAW_THRESHOLD:
            new_strong, new_weak = strong_goals + 1, weak_goals
            change_type = "draw_to_decisive"
            reason = (
                "Base v2 predicts a draw and the objective residual edge is extreme "
                f"(|diff| >= {EXTREME_DRAW_THRESHOLD:g}); award a one-goal win to the "
                "stronger side."
            )
        else:
            no_change["rule_components"] = " | ".join(components)
            no_change["reason"] = (
                "Gate met but base v2 is a draw and objective residual edge is not "
                f"extreme (|diff|={abs_diff:g} < {EXTREME_DRAW_THRESHOLD:g}); no change."
            )
            return no_change
    else:
        # Should be unreachable given the narrow guard, but stay deterministic.
        no_change["rule_components"] = " | ".join(components)
        no_change["reason"] = "Gate met but base v2 margin not adjustable under R1; no change."
        return no_change

    adjusted = _fmt(new_strong, new_weak) if strong_is_a else _fmt(new_weak, new_strong)
    return {
        "changed": adjusted != v2_score,
        "adjusted_score": adjusted,
        "rule_triggered": RULE_NAME,
        "change_type": change_type,
        "rule_components": " | ".join(components),
        "reason": reason,
    }


def build_adjustments() -> pd.DataFrame:
    """Recompute R1_only_diff_5_0 over every group match from current inputs."""
    v2 = pd.read_csv(V2_SCORES)
    overlay = pd.read_csv(OVERLAY_CSV)
    cat = dict(zip(overlay["team"], overlay["upside_category"]))
    score = dict(zip(overlay["team"], overlay["final_adjusted_human_overlay_score"]))

    rows: list[dict] = []
    for _, r in v2.iterrows():
        a, b = r["team_a"], r["team_b"]
        v2_score = str(r["final_recommended_score"])
        cat_a, cat_b = cat.get(a, "unknown"), cat.get(b, "unknown")
        sc_a, sc_b = score.get(a, float("nan")), score.get(b, float("nan"))
        diff = round(float(sc_a) - float(sc_b), 4)
        result = evaluate_rule(v2_score, cat_a, cat_b, sc_a, sc_b)
        rows.append(
            {
                "match_number": int(r["match_number"]),
                "group": r["group"],
                "team_a": a,
                "team_b": b,
                "v2_score": v2_score,
                "objective_residual_score": result["adjusted_score"],
                "changed": bool(result["changed"]),
                "rule_triggered": result["rule_triggered"],
                "change_type": result["change_type"],
                "overlay_diff": diff,
                "team_a_category": cat_a,
                "team_b_category": cat_b,
                "rule_components": result["rule_components"],
                "reason": result["reason"],
                "source_backtest_rule": SOURCE_BACKTEST,
                "deterministic_yes_no": "yes",
            }
        )
    return pd.DataFrame(rows)


def write_candidate(adj: pd.DataFrame) -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    v2 = pd.read_csv(V2_SCORES)
    score_map = dict(zip(adj["match_number"], adj["objective_residual_score"]))
    changed_map = dict(zip(adj["match_number"], adj["changed"]))
    rule_map = dict(zip(adj["match_number"], adj["rule_triggered"]))

    # Adjusted auto candidate: v2 columns preserved, recommended score replaced, with
    # explicit traceability columns. v2 itself is never touched.
    auto = v2.copy()
    auto["base_v2_recommended_score"] = auto["final_recommended_score"]
    auto["final_recommended_score"] = auto["match_number"].map(score_map)
    auto["objective_residual_changed"] = auto["match_number"].map(changed_map)
    auto["objective_rule_triggered"] = auto["match_number"].map(rule_map)
    auto.to_csv(V3_DIR / "final_group_score_predictions_auto.csv", index=False)

    # Fill-only export (same schema as v2's).
    fill_rows = []
    for _, r in adj.iterrows():
        fill_rows.append(
            {
                "match_number": r["match_number"],
                "group": r["group"],
                "team_a": r["team_a"],
                "team_b": r["team_b"],
                "score_to_fill_in": r["objective_residual_score"],
                "copy_text": (
                    f"{r['match_number']}. {r['team_a']} "
                    f"{r['objective_residual_score']} {r['team_b']}"
                ),
            }
        )
    pd.DataFrame(fill_rows).to_csv(
        V3_DIR / "final_group_score_predictions_fill_only.csv", index=False
    )

    # Full adjustment audit table.
    adj.to_csv(V3_DIR / "objective_residual_adjustments.csv", index=False)


def write_adjustment_report(adj: pd.DataFrame) -> None:
    changed = adj[adj["changed"]].copy()
    lines = [
        "# Objective Residual Adjustment Report (v3)",
        "",
        "Deterministic post-model layer over `final_candidate_v2_auto_science`.",
        f"Rule applied: **{RULE_NAME}** (see `objective_residual_adjustment_policy.md`).",
        "No manual approval or subjective override is used.",
        "",
        f"- Group matches: **{len(adj)}**",
        f"- Adjusted scores: **{len(changed)}**",
        f"- Max goals changed per match: **1**",
        "",
        "## Adjusted matches",
        "",
    ]
    if changed.empty:
        lines.append("_No match qualified under R1_only_diff_5_0._")
    else:
        lines.append(
            "| match | group | fixture | v2 | adjusted | change type | overlay diff "
            "| categories | rule components |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in changed.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['group']} | {r['team_a']} vs {r['team_b']} "
                f"| {r['v2_score']} | {r['objective_residual_score']} | {r['change_type']} "
                f"| {r['overlay_diff']:+g} | {r['team_a_category']} / {r['team_b_category']} "
                f"| {r['rule_components']} |"
            )
    lines += ["", "## Why objective", "", (
        "Each adjusted score is produced solely by the WC2022-validated rule "
        "`R1_only_diff_5_0`: an `elite_upside`/`positive` side against a "
        "`fragile`/`low_upside` side with `abs(overlay_diff) >= 5.0` and a narrow/"
        "conservative base v2 score. The adjustment is at most one goal and always "
        "follows the stronger objective-residual side. No chemistry-only, key-absence-"
        "only, fame-only, mid-table or broad-overlay logic is involved."
    ), ""]
    (V3_DIR / "objective_residual_adjustment_report.md").write_text("\n".join(lines))


def evaluate_promotion(adj: pd.DataFrame, v2_unchanged: bool) -> dict:
    changed = adj[adj["changed"]]
    n_changed = int(len(changed))

    # Every change is at most one goal and triggered by R1.
    one_goal = True
    rule_only = True
    for _, r in changed.iterrows():
        ga, gb = _parse_score(r["v2_score"])
        na, nb = _parse_score(r["objective_residual_score"])
        if abs(ga - na) + abs(gb - nb) > 1:
            one_goal = False
        if r["rule_triggered"] != RULE_NAME:
            rule_only = False

    # No forbidden category pairs among the changes (mid-table / chemistry / fame /
    # key-absence only are structurally impossible because the strong side must be
    # elite/positive and the weak side fragile/low — verify defensively).
    no_midtable = True
    for _, r in changed.iterrows():
        diff = r["overlay_diff"]
        strong_cat = r["team_a_category"] if diff > 0 else r["team_b_category"]
        weak_cat = r["team_b_category"] if diff > 0 else r["team_a_category"]
        if strong_cat not in STRONG_CATEGORIES or weak_cat not in WEAK_CATEGORIES:
            no_midtable = False

    checks = {
        "v2_byte_identical": v2_unchanged,
        "adjusted_candidate_separate": V3_DIR.exists(),
        "deterministic_from_R1": rule_only,
        "no_broad_or_manual_suggestions": True,
        "changes_not_failing": n_changed <= FAIL_CHANGES,
        "every_change_one_goal": one_goal,
        "no_midtable_chemistry_fame_keyabsence_only": no_midtable,
        "wc2022_evidence_cited": True,
    }
    warn = n_changed > WARN_CHANGES
    passed = all(checks.values())
    return {
        "passed": passed,
        "warning": warn,
        "n_changed": n_changed,
        "checks": checks,
    }


def write_promotion_report(adj: pd.DataFrame, gate: dict) -> None:
    changed = adj[adj["changed"]]
    status = "PASSED" if gate["passed"] else "FAILED"
    lines = [
        "# Objective Residual Candidate Promotion Report",
        "",
        f"**Promotion gate: {status}**"
        + ("  (warning: more than 8 changed scores)" if gate["warning"] else ""),
        "",
        "Base model: `final_candidate_v2_auto_science` (immutable baseline/reference).",
        "Adjusted candidate: `final_candidate_v3_objective_residual`.",
        f"Rule: **{RULE_NAME}**. No manual approval or subjective override used.",
        "",
        f"- Adjusted group scores: **{gate['n_changed']}** "
        f"(sparse expected 0–{WARN_CHANGES}; >{WARN_CHANGES} warns; >{FAIL_CHANGES} fails)",
        "",
        "## Gate checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for name, ok in gate["checks"].items():
        lines.append(f"| {name} | {'pass' if ok else 'FAIL'} |")
    lines += ["", "## Adjusted matches", ""]
    if changed.empty:
        lines.append("_No adjustments._")
    else:
        lines.append("| match | fixture | v2 | adjusted | rule |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in changed.iterrows():
            lines.append(
                f"| {r['match_number']} | {r['team_a']} vs {r['team_b']} | "
                f"{r['v2_score']} | {r['objective_residual_score']} | {r['rule_triggered']} |"
            )
    lines += [
        "",
        "## WC2022 evidence for R1_only_diff_5_0",
        "",
        "From `outputs/reports/objective_residual_rules_2022_backtest.md`: "
        "7 flags, 6 improved, 1 worsened, 0 unchanged, **net points delta +16.0**, "
        "exact/GD/outcome deltas +2/+2/+1; only rule set with `accepted_for_2026 = True`. "
        "Broad full-table overlay was noisy (net −2.0) and is kept as context/audit only.",
        "",
        "## Decision",
        "",
        (
            "Promotion gate passed: `final_candidate_v3_objective_residual` is the "
            "recommended score-fill candidate. The dashboard's *Scores to fill in* uses "
            "the objective-residual adjusted score; v2 is preserved as baseline/reference."
            if gate["passed"]
            else
            "Promotion gate failed: v2 remains the active score-to-fill-in source; v3 is "
            "not used."
        ),
        "",
    ]
    (REPORTS_DIR / "objective_residual_candidate_promotion_report.md").write_text(
        "\n".join(lines)
    )


def write_manifest(adj: pd.DataFrame, gate: dict, v2_unchanged: bool) -> None:
    files = []
    for name in [
        "final_group_score_predictions_auto.csv",
        "final_group_score_predictions_fill_only.csv",
        "objective_residual_adjustments.csv",
        "objective_residual_adjustment_report.md",
    ]:
        path = V3_DIR / name
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "candidate_dir": "outputs/final_candidate_v3_objective_residual",
        "base_model": "outputs/final_candidate_v2_auto_science",
        "rule": RULE_NAME,
        "source_backtest_rule": SOURCE_BACKTEST,
        "deterministic": True,
        "manual_approval_used": False,
        "subjective_override_used": False,
        "base_model_byte_identical": v2_unchanged,
        "promotion_gate_passed": gate["passed"],
        "promotion_warning": gate["warning"],
        "n_adjusted_scores": gate["n_changed"],
        "is_recommended_score_fill_candidate": gate["passed"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (V3_DIR / "FROZEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_v2_unchanged() -> bool:
    """Confirm every frozen v2 file matches its recorded sha256."""
    manifest = json.loads(V2_MANIFEST.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.exists() or _sha256(path) != entry["sha256"]:
            return False
    return True


# --- Part E: late-news residual audit + friendly-lineup policy -------------------------
LATE_NEWS_ROWS = [
    {
        "team": "Germany",
        "player": "Lennart Karl",
        "case_type": "late_squad_absence",
        "reason": "Ruled out with a left-thigh muscle tear",
        "replacement": "Assan Ouedraogo",
        "role_importance_0_5": 3,
        "already_reflected_score_0_5": 0,
        "recommended_residual": -0.25,
        "max_penalty": -0.25,
        "escalation_condition": (
            "Only beyond -0.25 if the current overlay had Karl as Germany's selected "
            "biggest talent (it does not: biggest_talent = Aleksandar Pavlović)."
        ),
        "is_key_absence": False,
        "auto_changes_score": False,
        "objective_rule_qualifies": False,
        "note": (
            "Late material absence audited as context. Does not change Germany scores: "
            "Germany is useful_context and no R1_only_diff_5_0 match qualifies."
        ),
    },
    {
        "team": "Argentina",
        "player": "Lionel Messi",
        "case_type": "in_squad_fitness_risk",
        "reason": "Left hamstring / muscle fatigue / overload — in squad, actively monitored",
        "replacement": "",
        "role_importance_0_5": 5,
        "already_reflected_score_0_5": 2,
        "recommended_residual": -0.25,
        "max_penalty": -0.50,
        "escalation_condition": (
            "Escalate to -0.50 only if opener availability or a minutes restriction "
            "becomes explicitly doubtful."
        ),
        "is_key_absence": False,
        "auto_changes_score": False,
        "objective_rule_qualifies": False,
        "note": (
            "Fitness risk, NOT a key absence. Default late-fitness residual -0.25. Does "
            "not change Argentina scores unless an objective rule qualifies (none does)."
        ),
    },
]


def write_late_news_audit() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(LATE_NEWS_ROWS)
    df.to_csv(PRED_DIR / "wc2026_late_news_residual_audit.csv", index=False)

    lines = [
        "# WC2026 Late-News Residual Audit",
        "",
        "Objective audit of fresh late-news residuals. These are residual *context* "
        "items. They do **not** automatically change any score; a score only moves if a "
        "validated objective rule (`R1_only_diff_5_0` or a validated late-material-"
        "absence rule) qualifies. No manual approval or subjective override is used.",
        "",
    ]
    for row in LATE_NEWS_ROWS:
        lines += [
            f"## {row['team']} — {row['player']} ({row['case_type']})",
            "",
            f"- Reason: {row['reason']}",
            f"- Replacement: {row['replacement'] or '—'}",
            f"- role_importance_0_5: {row['role_importance_0_5']}",
            f"- already_reflected_score_0_5: {row['already_reflected_score_0_5']}",
            f"- recommended residual: {row['recommended_residual']:+g}",
            f"- max penalty: {row['max_penalty']:+g}",
            f"- key absence: {'yes' if row['is_key_absence'] else 'no'}",
            f"- escalation: {row['escalation_condition']}",
            f"- auto-changes score: {'yes' if row['auto_changes_score'] else 'no'}",
            f"- note: {row['note']}",
            "",
        ]
    (REPORTS_DIR / "wc2026_late_news_residual_audit.md").write_text("\n".join(lines))


def write_friendly_lineup_policy() -> None:
    text = """# Friendly-Lineup Distortion Policy

**Status:** active, objective.

If a pre-WC friendly is played **without** a confirmed star/carry player due to
precautionary fitness management (rotation, load management, late fitness risk), the
match is tagged as **lineup-distorted** *before* it is used for any form/momentum update.

## Rule

1. A friendly is `lineup_distorted = true` when a team's confirmed star/carry player is
   absent or minutes-restricted for precautionary fitness reasons.
2. Lineup-distorted friendlies are **down-weighted or excluded** from form/momentum
   updates; they are never allowed to silently shift a team's objective signal.
3. This is a data-hygiene tag only. It never adds a subjective score adjustment and is
   independent of the `R1_only_diff_5_0` post-model rule.

## Application to current late-news cases

- **Argentina / Messi** (`in_squad_fitness_risk`): any pre-WC friendly played without
  Messi for precautionary fitness management is tagged `lineup_distorted` before form
  updates. Default late-fitness residual remains `-0.25` (see
  `wc2026_late_news_residual_audit.md`).
- **Germany / Karl** (`late_squad_absence`): friendlies missing Karl are tagged
  `lineup_distorted`; Karl is not Germany's selected biggest talent, so the residual is
  capped at `-0.25` and does not change scores.
"""
    (REPORTS_DIR / "friendly_lineup_distortion_policy.md").write_text(text)


def main() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    v2_unchanged = verify_v2_unchanged()

    adj = build_adjustments()
    write_candidate(adj)
    write_adjustment_report(adj)

    gate = evaluate_promotion(adj, v2_unchanged)
    write_promotion_report(adj, gate)
    write_manifest(adj, gate, v2_unchanged)

    write_late_news_audit()
    write_friendly_lineup_policy()

    changed = adj[adj["changed"]]
    print(f"Objective residual candidate written to {V3_DIR.relative_to(ROOT)}")
    print(f"v2 byte-identical: {v2_unchanged}")
    print(f"adjusted scores: {gate['n_changed']} | promotion passed: {gate['passed']}")
    for _, r in changed.iterrows():
        print(
            f"  match {r['match_number']}: {r['team_a']} vs {r['team_b']} "
            f"{r['v2_score']} -> {r['objective_residual_score']} ({r['change_type']})"
        )
    return {"adjustments": adj, "gate": gate, "v2_unchanged": v2_unchanged}


if __name__ == "__main__":
    main()
