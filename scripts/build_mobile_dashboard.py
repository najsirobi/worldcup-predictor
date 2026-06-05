#!/usr/bin/env python3
"""Build the static mobile Travel Mode dashboard (Travel Mode, Task H).

Consolidates live outputs and active-candidate picks into a JSON payload,
rendered as a self-contained, mobile-friendly HTML file with robust error
handling and safe defaults for all properties.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from html import escape as _escape
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate
from src.live.scores_override import utc_now_iso
from src.live.submission_guard import guard_frozen_submission

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"
DOCS_DIR = ROOT / "docs"
REPO_SLUG = os.environ.get("TRAVEL_MODE_REPO", "")
V1_DIR = ROOT / "outputs" / "final_candidate_v1"


def _json_safe(value):
    """Recursively convert values into strict-JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=str)]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _payload_json(payload: dict, *, indent: int | None = None) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=indent)


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _review_rows(scores: pd.DataFrame) -> list[dict]:
    """Manual-review flagged fixtures, tolerant of v1/v2 column names."""
    flag_col = next(
        (c for c in ("manual_review_flag", "manual_review_flag_original") if c in scores.columns),
        None,
    )
    if flag_col is None:
        return []
    review = scores[scores[flag_col].astype(str).str.lower() == "true"]
    cols = ["match_number", "group", "team_a", "team_b", "final_recommended_score"]
    if "reason" in review.columns:
        cols.append("reason")
    return review[cols].to_dict(orient="records")


def _score_parts(score: object) -> tuple[object, object]:
    if not isinstance(score, str) or "-" not in score:
        return None, None
    left, right = score.split("-", 1)
    try:
        return int(left), int(right)
    except Exception:
        return None, None


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _match_date_map(*record_sets: list[dict]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for records in record_sets:
        for row in records:
            match_number = row.get("match_number")
            if match_number is None:
                continue
            date = row.get("date")
            if date:
                mapping[int(match_number)] = str(date)
    return mapping


def _submission_score_rows(
    scores: pd.DataFrame,
    date_map: dict[int, str],
    fill_map: dict[int, dict],
    prediction_vs_actual: list[dict],
) -> tuple[list[dict], str]:
    """Build per-match frozen submission rows with live actual overlays.

    Submitted scores and copy-friendly lines come from the clean fill-only export
    when present. Actual scores and points are overlaid from live
    prediction-vs-actual scoring without changing the submitted score.
    """
    actual_by_match = {
        int(row["match_number"]): row
        for row in prediction_vs_actual
        if row.get("match_number") is not None
    }
    rows: list[dict] = []
    copy_lines: list[str] = []
    for row in scores.where(pd.notna(scores), None).to_dict(orient="records"):
        match_number = int(row.get("match_number")) if row.get("match_number") is not None else None
        fill = fill_map.get(match_number, {})
        group = fill.get("group") or row.get("group") or ""
        team_a = fill.get("team_a") or row.get("team_a") or ""
        team_b = fill.get("team_b") or row.get("team_b") or ""
        # Authoritative score + copy line come from the fill-only export.
        score = fill.get("score_to_fill_in") or row.get("final_recommended_score") or ""
        copy_text = fill.get("copy_text") or f"{match_number}. {team_a} {score} {team_b}"
        team_a_goals, team_b_goals = _score_parts(score)
        date = date_map.get(match_number) if match_number is not None else None
        actual = actual_by_match.get(match_number or -1, {})
        points_earned = actual.get("points_earned", actual.get("total_points"))
        rows.append(
            {
                "match_number": match_number,
                "group": group,
                "date": date,
                "team_a": team_a,
                "team_b": team_b,
                "status": "locked/submitted",
                "submitted_score": score,
                "actual_score": actual.get("actual_score"),
                "points_earned": points_earned,
                "final_recommended_score": score,
                "score_to_fill_in": score,
                "copy_text": copy_text,
                "predicted_team_a_goals": team_a_goals,
                "predicted_team_b_goals": team_b_goals,
                "safe_score": row.get("safe_score"),
                "ev_score": row.get("ev_score"),
                "auto_consensus_score": row.get("auto_consensus_score"),
                "auto_policy_decision": row.get("auto_policy_decision"),
                "reason": row.get("reason"),
                "manual_review_flag_original": bool(row.get("manual_review_flag_original")),
                "manual_review_resolved_auto": bool(row.get("manual_review_resolved_auto")),
            }
        )
        copy_lines.append(copy_text)
    return rows, "\n".join(copy_lines)


def _compare_strings(v1: list[dict], v2: list[dict], keys: list[str]) -> int:
    changes = 0
    for left, right in zip(v1, v2):
        if any(left.get(key) != right.get(key) for key in keys):
            changes += 1
    return changes


FILL_ONLY_NAME = "final_group_score_predictions_fill_only.csv"


def _load_fill_map(candidate_dir: Path) -> dict[int, dict]:
    """Read the clean fill-only export keyed by match number."""
    path = candidate_dir / FILL_ONLY_NAME
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    fill_map: dict[int, dict] = {}
    for row in frame.where(pd.notna(frame), None).to_dict(orient="records"):
        number = row.get("match_number")
        if number is None:
            continue
        fill_map[int(number)] = {
            "group": row.get("group"),
            "team_a": row.get("team_a"),
            "team_b": row.get("team_b"),
            "score_to_fill_in": row.get("score_to_fill_in"),
            "copy_text": row.get("copy_text"),
        }
    return fill_map


def _load_knockout_predictions() -> dict:
    """Build fresh knockout predictions; fall back to any cached JSON, then empty."""
    try:
        from scripts.build_knockout_predictions import build as build_knockout

        return build_knockout()
    except Exception:
        return _read_json(LIVE_DIR / "knockout_predictions.json")


def build_payload() -> dict:
    candidate_obj = load_active_candidate()
    candidate = candidate_obj.as_dict()

    live_tables = _read_json(LIVE_DIR / "live_group_tables.json")
    sim = _read_json(LIVE_DIR / "live_group_stage_simulation_summary.json")
    pva = _read_json(LIVE_DIR / "prediction_vs_actual.json")
    scoring = _read_json(LIVE_DIR / "scoring_summary.json")
    knockout = _load_knockout_predictions()

    played = _read_csv_records(LIVE_DIR / "played_matches.csv")
    remaining = _read_csv_records(LIVE_DIR / "remaining_matches.csv")
    date_map = _match_date_map(played, remaining)

    scores = candidate_obj.load_score_predictions()
    standings_df = candidate_obj.load_standing_predictions()
    last8_df = candidate_obj.load_last8_predictions()
    standings = standings_df.to_dict(orient="records")
    last8 = last8_df.to_dict(orient="records")
    fill_map = _load_fill_map(candidate_obj.candidate_dir)
    prediction_vs_actual_matches = pva.get("matches", [])
    submission_scores, submission_copy_text = _submission_score_rows(
        scores,
        date_map,
        fill_map,
        prediction_vs_actual_matches,
    )
    review = _review_rows(scores)

    manual_auto_resolved = int(scores["manual_review_flag_original"].fillna(False).astype(bool).sum()) if "manual_review_flag_original" in scores.columns else 0
    ev_accepted = int((scores["auto_policy_decision"] == "ev_override_accepted").sum()) if "auto_policy_decision" in scores.columns else 0
    ev_rejected = int(
        (
            scores["safe_score"].astype(str).ne(scores["ev_score"].astype(str))
            & scores["auto_policy_decision"].astype(str).ne("ev_override_accepted")
        ).sum()
    ) if {"safe_score", "ev_score", "auto_policy_decision"}.issubset(scores.columns) else 0
    safe_kept = int((scores["final_recommended_score"].astype(str) == scores["safe_score"].astype(str)).sum()) if {"final_recommended_score", "safe_score"}.issubset(scores.columns) else 0

    v1_scores_path = V1_DIR / "final_group_score_predictions.csv"
    v1_standings_path = V1_DIR / "final_group_standing_predictions.csv"
    v1_last8_path = V1_DIR / "final_last8_predictions.csv"
    comparison_summary = {
        "score_changes_vs_v1": None,
        "group_standings_changed_vs_v1": None,
        "last8_changed_vs_v1": None,
    }
    if v1_scores_path.exists() and v1_standings_path.exists() and v1_last8_path.exists():
        v1_scores_df = pd.read_csv(v1_scores_path)
        v1_scores = v1_scores_df.where(pd.notna(v1_scores_df), None).to_dict(orient="records")
        current_scores = scores.where(pd.notna(scores), None).to_dict(orient="records")
        comparison_summary["score_changes_vs_v1"] = _compare_strings(v1_scores, current_scores, ["final_recommended_score"])
        v1_standings_df = pd.read_csv(v1_standings_path)
        v1_standings = v1_standings_df.where(pd.notna(v1_standings_df), None).to_dict(orient="records")
        comparison_summary["group_standings_changed_vs_v1"] = _compare_strings(v1_standings, standings, ["rank_1", "rank_2", "rank_3", "rank_4"])
        v1_last8_df = pd.read_csv(v1_last8_path)
        v1_last8 = v1_last8_df.where(pd.notna(v1_last8_df), None).to_dict(orient="records")
        comparison_summary["last8_changed_vs_v1"] = _compare_strings(v1_last8, last8, ["stage", "rank", "team", "probability", "stage_points", "expected_points", "selection_type"])

    scoring_summary = {
        "played_matches": scoring.get("played_matches", 0),
        "total_points": scoring.get("total_points", 0),
        "possible_points_for_played_matches": scoring.get(
            "possible_points_for_played_matches", 0
        ),
        "points_missed": scoring.get("points_missed", 0),
        "average_points_per_played_match": scoring.get("average_points_per_played_match", 0),
        "outcomes_correct": scoring.get("outcomes_correct", 0),
        "goal_differences_correct": scoring.get("goal_differences_correct", 0),
        "exact_scores_correct": scoring.get("exact_scores_correct", 0),
        "total_by_group": scoring.get("total_by_group", {}),
    }

    submission_summary = {
        "total_matches": len(submission_scores),
        "manual_review_rows_auto_resolved": manual_auto_resolved,
        "ev_overrides_accepted": ev_accepted,
        "ev_overrides_rejected": ev_rejected,
        "safe_scores_kept": safe_kept,
        **comparison_summary,
    }

    return {
        "generated_at": utc_now_iso(),
        "repo_slug": REPO_SLUG,
        "active_candidate": candidate,
        "summary": {
            "matches_played": live_tables.get("matches_played", len(played)),
            "matches_remaining": live_tables.get("matches_remaining", len(remaining)),
            "n_sims": sim.get("n_sims"),
        },
        "tie_break_note": live_tables.get("tie_break_note", ""),
        "sim_tie_break_note": sim.get("tie_break_note", ""),
        "semantics": live_tables.get("semantics", sim.get("semantics", {})),
        "actual_bracket_state": live_tables.get("actual_bracket_state", {}),
        "incentive_diagnostics": live_tables.get("incentive_diagnostics", {"teams": [], "matches": []}),
        "played_matches": played,
        "remaining_matches": remaining[:24],
        "remaining_matches_total": len(remaining),
        "groups": live_tables.get("groups", {}),
        "advancement": sim.get("teams", []),
        "prediction_vs_actual": prediction_vs_actual_matches,
        "scoring_summary": scoring_summary,
        "submission_summary": submission_summary,
        "submission_score_predictions": submission_scores,
        "submission_score_copy_text": submission_copy_text,
        "submission_group_standings": standings,
        "submission_last8_picks": last8,
        "final_group_standings": standings,
        "last8_picks": last8,
        "knockout_predictions": knockout,
        "manual_review": review,
    }


SECTION_DEFAULT_OPEN = {
    "overview": True,
    "submitted-scores": True,
    "group-standings": False,
    "last8": False,
    "knockout-predictions": True,
    "live-results": False,
    "prediction-vs-actual": False,
    "live-group-tables": False,
    "incentives": False,
    "advancement": False,
}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC2026 Travel Mode</title>
<style>
  :root {
    color-scheme: light;
    --background: #f7f4ee;
    --card-background: #ffffff;
    --card-bg: #ffffff;
    --subcard-bg: #f0ebe3;
    --text: #17202f;
    --muted-text: #667085;
    --border: #d8d0c5;
    --accent: #0f7796;
    --success: #178a49;
    --warning: #b66a05;
    --danger: #c33f32;
    --shadow: 0 12px 35px rgba(31, 41, 55, .08);
  }
  html[data-theme="dark"] {
    color-scheme: dark;
    --background: #0f1116;
    --card-background: #161a23;
    --card-bg: #161a23;
    --subcard-bg: #1b212c;
    --text: #e7e9ee;
    --muted-text: #9aa3b2;
    --border: #2a2f3a;
    --accent: #7dd3fc;
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
    --shadow: none;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0 0 3rem; background: var(--background); color: var(--text); line-height: 1.4; font-size: 16px; }
  header { position: sticky; top: 0; background: var(--card-bg); padding: .8rem 1rem; border-bottom: 1px solid var(--border); z-index: 5; box-shadow: var(--shadow); }
  .header-top { display:flex; justify-content:space-between; align-items:flex-start; gap:.8rem; }
  header h1 { margin: 0; font-size: 1.15rem; color: var(--text); }
  header .meta { font-size: .78rem; color: var(--muted-text); margin-top: .15rem; }
  header .cand { font-size: .78rem; color: var(--accent); margin-top: .15rem; }
  .theme-toggle { display:flex; align-items:center; gap:.25rem; border:1px solid var(--border); background:var(--subcard-bg); padding:.18rem; border-radius:999px; white-space:nowrap; }
  .theme-toggle span { color:var(--muted-text); font-size:.72rem; padding:0 .25rem; }
  .theme-toggle button { border:0; border-radius:999px; padding:.28rem .52rem; background:transparent; color:var(--muted-text); font-weight:700; font-size:.74rem; cursor:pointer; }
  html[data-theme="light"] .theme-toggle [data-theme-choice="light"],
  html[data-theme="dark"] .theme-toggle [data-theme-choice="dark"] { background:var(--accent); color:var(--card-bg); }
  nav.tabs { position: sticky; top: 4.6rem; z-index: 4; display:flex; gap:.4rem; overflow-x:auto; padding:.55rem .8rem; background:var(--background); border-bottom:1px solid var(--border); }
  nav.tabs a { white-space:nowrap; text-decoration:none; padding:.35rem .65rem; border-radius:999px; background:var(--card-bg); color:var(--text); font-size:.78rem; border:1px solid var(--border); }
  nav.tabs a:hover { border-color:var(--accent); color:var(--accent); }
  main { padding: 0 .8rem; max-width: 720px; margin: 0 auto; }
  .dashboard-section { margin-top: 1.1rem; background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); overflow:hidden; }
  .dashboard-section > summary { list-style:none; cursor:pointer; color:var(--text); padding:.85rem; display:flex; justify-content:space-between; align-items:center; gap:.65rem; }
  .dashboard-section > summary::-webkit-details-marker { display:none; }
  .dashboard-section > summary::after { content:"+"; display:grid; place-items:center; min-width:1.55rem; height:1.55rem; border-radius:999px; border:1px solid var(--border); color:var(--accent); font-weight:900; }
  .dashboard-section[open] > summary::after { content:"−"; }
  .section-title { margin:0; font-size:1rem; color:var(--text); display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; }
  .section-body { padding:0 .85rem .85rem; overflow-x:auto; }
  .subcard { background:var(--subcard-bg); border:1px solid var(--border); border-radius:10px; padding:.75rem; margin:.6rem 0; overflow-x:auto; }
  .subcard h3 { margin:.1rem 0 .45rem; font-size:.92rem; color:var(--text); display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
  .pill { display:inline-block; font-size:.7rem; padding:.1rem .45rem; border-radius:999px; background:var(--subcard-bg); color:var(--muted-text); border:1px solid var(--border); }
  table { width: 100%; min-width: 520px; border-collapse: collapse; font-size: .82rem; }
  th, td { text-align: right; padding: .34rem .35rem; border-bottom: 1px solid var(--border); vertical-align:top; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted-text); font-weight: 700; }
  /* Aligned tables: fixed layout so headers line up with cells on mobile. */
  table.aligned { width: 100%; table-layout: fixed; border-collapse: collapse; }
  table.aligned th, table.aligned td { padding: .34rem .28rem; border-bottom: 1px solid var(--border); overflow: hidden; }
  table.aligned th { color: var(--muted-text); font-weight: 700; }
  table.aligned .num { text-align: right; font-variant-numeric: tabular-nums; }
  table.aligned .ctr { text-align: center; font-variant-numeric: tabular-nums; }
  table.aligned .team { text-align: left; white-space: normal; overflow-wrap:anywhere; word-break: normal; }
  table.aligned .rank { display: inline-block; min-width: 1.2rem; color: var(--muted-text); font-variant-numeric: tabular-nums; }
  table.aligned col.numcol { width: 1.9rem; }
  table.aligned.standings-table th, table.aligned.standings-table td { text-align: left; white-space: normal; overflow-wrap:anywhere; word-break: normal; }
  table.aligned.standings-table col.gcol { width: 1.8rem; }
  .grp { font-weight: 700; color: var(--text); margin: .6rem 0 .2rem; font-size: .9rem; }
  .adv { color: var(--success); }
  .out { color: var(--danger); }
  .bar { height: 6px; border-radius: 4px; background:var(--subcard-bg); overflow:hidden; margin-top:2px; border:1px solid var(--border); }
  .bar > i { display:block; height:100%; background:var(--success); }
  .match { display:flex; justify-content:space-between; padding:.3rem 0; border-bottom:1px solid var(--border); font-size:.85rem; gap:.75rem; }
  .match .sc { font-weight:700; }
  .error { background:color-mix(in srgb, var(--danger) 16%, var(--card-bg)); border: 1px solid var(--danger); color:var(--text); padding:1rem; border-radius:8px; margin:1rem 0; }
  .error h3 { color:var(--danger); margin-top:0; }
  .error pre { background:var(--subcard-bg); color:var(--text); max-height:300px; overflow-y:auto; font-size:.7rem; }
  .warn { background:color-mix(in srgb, var(--warning) 16%, var(--card-bg)); border-color:var(--warning); }
  .warn h2 { color:var(--warning); }
  .muted { color:var(--muted-text); font-size:.8rem; }
  code { background:var(--subcard-bg); padding:.05rem .3rem; border-radius:4px; font-size:.8rem; }
  pre { background:var(--subcard-bg); padding:.6rem; border-radius:8px; overflow:auto; font-size:.75rem; }
  a { color:var(--accent); }
  details { margin-top:.4rem; }
  summary { cursor:pointer; color:var(--text); }
  .filelinks a { display:inline-block; margin:.15rem .4rem .15rem 0; font-size:.8rem; }
  ol { padding-left:1.2rem; } ol li { margin:.25rem 0; }
  .bignum { font-size:1.8rem; font-weight:800; color:var(--success); }
  .stats { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:.4rem; }
  .stat { flex:1 1 30%; background:var(--subcard-bg); border-radius:8px; padding:.5rem; text-align:center; }
  .stat b { display:block; font-size:1.2rem; color:var(--text); }
  .stat span { font-size:.72rem; color:var(--muted-text); }
  .summary-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:.55rem; }
  .summary-card { background:var(--subcard-bg); border:1px solid var(--border); border-radius:10px; padding:.7rem; }
  .summary-card b { display:block; font-size:1rem; margin-bottom:.15rem; }
  .summary-card span { color:var(--muted-text); font-size:.74rem; }
  .policy-list { margin:.6rem 0 0; padding-left:1.2rem; }
  .policy-list li { margin:.25rem 0; }
  .copybar { display:flex; align-items:center; justify-content:space-between; gap:.5rem; margin:.5rem 0 .65rem; flex-wrap:wrap; }
  .copybar button { background:var(--accent); color:var(--card-bg); border:0; border-radius:999px; padding:.4rem .75rem; font-weight:700; cursor:pointer; }
  .copyblock { white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal; background:var(--subcard-bg); border:1px solid var(--border); border-radius:10px; padding:.7rem; font-size:.78rem; }
  .group-summary { cursor:pointer; }
  .row-card { background:var(--subcard-bg); border:1px solid var(--border); border-radius:10px; padding:.7rem; margin:.55rem 0; }
  .row-top { display:flex; justify-content:space-between; gap:.6rem; align-items:flex-start; }
  .row-title { font-weight:800; font-size:.95rem; overflow-wrap:anywhere; }
  .score-big { font-size:1.8rem; font-weight:900; color:var(--success); letter-spacing:-0.02em; }
  .row-meta { display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.5rem; }
  .row-meta .pill { background:var(--card-bg); }
  .row-copy { margin-top:.55rem; font-size:.78rem; color:var(--muted-text); overflow-wrap:anywhere; }
  .label { color:var(--muted-text); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
  .row-top > div:last-child { text-align:right; }
  .audit-details { margin-top:.5rem; border-top:1px solid var(--border); padding-top:.4rem; }
  .audit-details > summary { color:var(--muted-text); font-size:.78rem; }
  .copybar b { font-size:.85rem; color:var(--text); }
  @media (max-width: 560px) {
    header { padding:.7rem .8rem; }
    .header-top { flex-direction:column; align-items:stretch; gap:.55rem; }
    .theme-toggle { align-self:flex-start; }
    nav.tabs { top:6.8rem; }
    main { padding:0 .6rem; }
    .summary-grid { grid-template-columns:1fr; }
    .row-top { flex-direction:column; }
    .row-top > div:last-child { text-align:left; }
  }
</style>
</head>
<body>
<header>
  <div class="header-top">
    <div>
      <h1>🏆 WC2026 Travel Mode</h1>
      <div class="meta" id="meta"><!--META_PLACEHOLDER--></div>
      <div class="cand" id="cand"><!--CAND_PLACEHOLDER--></div>
    </div>
    <div class="theme-toggle" aria-label="Light / Dark theme">
      <span>Light / Dark</span>
      <button type="button" data-theme-choice="light" aria-pressed="true">Light</button>
      <button type="button" data-theme-choice="dark" aria-pressed="false">Dark</button>
    </div>
  </div>
</header>
<nav class="tabs" id="tabs">
  <a href="#overview">Overview</a>
  <a href="#submitted-scores">Submitted Scores</a>
  <a href="#group-standings">Group Standings</a>
  <a href="#last8">Last-8</a>
  <a href="#knockout-predictions">Knockout</a>
  <a href="#live-results">Live Results</a>
  <a href="#prediction-vs-actual">Prediction vs Actual</a>
  <a href="#live-group-tables">Live group tables</a>
  <a href="#incentives">Incentives</a>
  <a href="#advancement">Advancement</a>
</nav>
<main id="app"><!--APP_PLACEHOLDER--></main>
<script id="payload" type="application/json"><!--PAYLOAD_JSON_PLACEHOLDER--></script>
<script>
(function() {
  const app = document.getElementById('app');
  const metaEl = document.getElementById('meta');
  const candEl = document.getElementById('cand');
  const payloadEl = document.getElementById('payload');

  function safeObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function(ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function fmtNum(value) {
    const n = Number(value);
    return Number.isFinite(n) ? String(Math.round(n * 100) / 100) : '—';
  }

  function fmtInt(value) {
    const n = Number(value);
    return Number.isFinite(n) ? String(Math.round(n)) : '—';
  }

  function fmtPct(value) {
    const n = Number(value);
    return Number.isFinite(n) ? String(Math.round(n * 100)) + '%' : '—';
  }

  function pctWidth(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(0, Math.min(100, Math.round(n * 100))) : 0;
  }

  const DEFAULT_OPEN_SECTIONS = {
    overview: true,
    'submitted-scores': true,
    'group-standings': false,
    last8: false,
    'knockout-predictions': true,
    'live-results': false,
    'prediction-vs-actual': false,
    'live-group-tables': false,
    incentives: false,
    advancement: false
  };

  function buildSection(id, title, pill, body) {
    const open = DEFAULT_OPEN_SECTIONS[id] ? ' open' : '';
    return '<details class="dashboard-section" id="' + id + '"' + open + '><summary><h2 class="section-title">' + title + (pill ? '<span class="pill">' + pill + '</span>' : '') + '</h2></summary><div class="section-body">' + body + '</div></details>';
  }

  function buildSubcard(title, body, rightMeta) {
    return '<div class="subcard"><h3><span>' + title + '</span>' + (rightMeta ? '<span class="pill">' + rightMeta + '</span>' : '') + '</h3>' + body + '</div>';
  }

  function showError(title, err, data) {
    // Non-destructive: the server-rendered sections stay; we only prepend a
    // banner so the dashboard is still usable if the JS enhancement fails.
    const keys = data && typeof data === 'object' && !Array.isArray(data) ? Object.keys(data) : [];
    const message = err && err.message ? err.message : String(err || 'Unknown error');
    const stack = err && err.stack ? '<pre>' + esc(err.stack) + '</pre>' : '';
    const banner = '<section class="error" id="js-error-banner"><h3>⚠️ ' + esc(title) + ' (showing pre-rendered content)</h3><p>' + esc(message) + '</p><p><b>Loaded keys:</b> ' + esc(keys.length ? keys.join(', ') : 'none') + '</p>' + stack + '</section>';
    if (app && !document.getElementById('js-error-banner')) {
      app.insertAdjacentHTML('afterbegin', banner);
    }
  }

  function parseInlinePayload() {
    const raw = payloadEl && payloadEl.textContent ? payloadEl.textContent.trim() : '';
    if (!raw) return null;
    return JSON.parse(raw);
  }

  function fetchPayload() {
    return fetch('./mobile_dashboard_data.json', { cache: 'no-store' })
      .then(function(response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        return response.text();
      })
      .then(function(text) {
        const raw = text ? text.trim() : '';
        if (!raw) {
          throw new Error('Fetched dashboard payload was empty');
        }
        return JSON.parse(raw);
      });
  }

  function validatePayload(raw) {
    const data = safeObject(raw);
    const requiredObjects = ['active_candidate', 'summary', 'groups', 'scoring_summary', 'submission_summary'];
    const requiredArrays = ['played_matches', 'remaining_matches', 'advancement', 'prediction_vs_actual', 'submission_score_predictions', 'submission_group_standings', 'submission_last8_picks', 'final_group_standings', 'last8_picks', 'manual_review'];
    const errors = [];

    if (!data || typeof data !== 'object') {
      errors.push('payload must be an object');
    }

    for (let i = 0; i < requiredObjects.length; i += 1) {
      const key = requiredObjects[i];
      if (!(key in data)) {
        errors.push(key + ' missing');
      } else if (data[key] != null && (typeof data[key] !== 'object' || Array.isArray(data[key]))) {
        errors.push(key + ' must be an object');
      }
    }

    for (let i = 0; i < requiredArrays.length; i += 1) {
      const key = requiredArrays[i];
      if (!(key in data)) {
        errors.push(key + ' missing');
      } else if (data[key] != null && !Array.isArray(data[key])) {
        errors.push(key + ' must be an array');
      }
    }

    if (errors.length) {
      const err = new Error('Dashboard schema validation failed: ' + errors.join('; '));
      err.loadedKeys = Object.keys(data);
      throw err;
    }

    const remainingMatches = safeArray(data.remaining_matches);

    return {
      generated_at: typeof data.generated_at === 'string' ? data.generated_at : '',
      repo_slug: typeof data.repo_slug === 'string' ? data.repo_slug : '',
      active_candidate: safeObject(data.active_candidate),
      summary: safeObject(data.summary),
      tie_break_note: typeof data.tie_break_note === 'string' ? data.tie_break_note : '',
      sim_tie_break_note: typeof data.sim_tie_break_note === 'string' ? data.sim_tie_break_note : '',
      semantics: safeObject(data.semantics),
      actual_bracket_state: safeObject(data.actual_bracket_state),
      incentive_diagnostics: safeObject(data.incentive_diagnostics),
      played_matches: safeArray(data.played_matches),
      remaining_matches: remainingMatches,
      remaining_matches_total: Number.isFinite(Number(data.remaining_matches_total)) ? Number(data.remaining_matches_total) : remainingMatches.length,
      groups: safeObject(data.groups),
      advancement: safeArray(data.advancement),
      prediction_vs_actual: safeArray(data.prediction_vs_actual),
      scoring_summary: safeObject(data.scoring_summary),
      submission_summary: safeObject(data.submission_summary),
      submission_score_predictions: safeArray(data.submission_score_predictions),
      submission_score_copy_text: typeof data.submission_score_copy_text === 'string' ? data.submission_score_copy_text : '',
      submission_group_standings: safeArray(data.submission_group_standings),
      submission_last8_picks: safeArray(data.submission_last8_picks),
      final_group_standings: safeArray(data.final_group_standings),
      last8_picks: safeArray(data.last8_picks),
      knockout_predictions: safeObject(data.knockout_predictions),
      manual_review: safeArray(data.manual_review)
    };
  }

  function renderPlayedMatches(rows) {
    if (!rows.length) {
      return '<p class="muted">No matches played yet.</p>';
    }
    let html = '';
    for (let i = 0; i < rows.length; i += 1) {
      const m = safeObject(rows[i]);
      html += '<div class="match"><span>#' + fmtInt(m.match_number) + ' ' + esc(m.team_a || '—') + ' v ' + esc(m.team_b || '—') + '</span><span class="sc">' + fmtInt(m.team_a_goals) + '–' + fmtInt(m.team_b_goals) + '</span></div>';
    }
    return html;
  }

  function renderPredictionVsActual(rows) {
    if (!rows.length) {
      return '<p class="muted">No prediction-vs-actual data yet.</p>';
    }
    let table = '<table class="aligned">'
      + '<colgroup><col class="numcol"><col><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"></colgroup>'
      + '<tr><th class="num">#</th><th class="team">Match</th><th class="num">Submitted</th><th class="num">Actual</th><th class="num">Exact</th><th class="num">Points</th></tr>';
    let details = '';
    for (let i = 0; i < rows.length; i += 1) {
      const m = safeObject(rows[i]);
      const points = m.points_earned != null ? m.points_earned : m.total_points;
      table += '<tr><td class="num">' + fmtInt(m.match_number) + '</td><td class="team">' + esc(m.team_a || '—') + ' v ' + esc(m.team_b || '—') + '</td><td class="num">' + esc(m.submitted_score || m.predicted_score || '—') + '</td><td class="num">' + esc(m.actual_score || '—') + '</td><td class="num">' + (m.exact_score_correct ? 'yes' : 'no') + '</td><td class="num">' + fmtNum(points) + '</td></tr>';
      details += '<details><summary>#' + fmtInt(m.match_number) + ' ' + esc(m.team_a || '—') + ' v ' + esc(m.team_b || '—') + ' - ' + fmtNum(points) + ' points earned</summary><p class="muted">' + esc(m.scoring_explanation || 'No explanation available.') + '</p></details>';
    }
    table += '</table>';
    return table + details;
  }

  function renderGroups(groups) {
    const keys = Object.keys(groups).sort();
    let html = '';
    for (let i = 0; i < keys.length; i += 1) {
      const groupName = keys[i];
      const rows = safeArray(groups[groupName]);
      if (!rows.length) {
        continue;
      }
      html += '<div class="grp">Group ' + esc(groupName) + '</div>'
        + '<table class="aligned">'
        + '<colgroup><col><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"></colgroup>'
        + '<tr><th class="team">Team</th><th class="num">P</th><th class="num">W</th><th class="num">D</th><th class="num">L</th><th class="num">GF</th><th class="num">GA</th><th class="num">GD</th><th class="num">Pts</th></tr>';
      for (let j = 0; j < rows.length; j += 1) {
        const row = safeObject(rows[j]);
        const gd = Number(row.goal_difference);
        const gdText = Number.isFinite(gd) ? (gd >= 0 ? '+' : '') + fmtInt(gd) : '—';
        html += '<tr><td class="team"><span class="rank">' + fmtInt(row.rank) + '</span>' + esc(row.team || '—') + '</td><td class="num">' + fmtInt(row.played) + '</td><td class="num">' + fmtInt(row.won) + '</td><td class="num">' + fmtInt(row.drawn) + '</td><td class="num">' + fmtInt(row.lost) + '</td><td class="num">' + fmtInt(row.goals_for) + '</td><td class="num">' + fmtInt(row.goals_against) + '</td><td class="num">' + gdText + '</td><td class="num">' + fmtInt(row.points) + '</td></tr>';
      }
      html += '</table>';
    }
    return html || '<p class="muted">No live group table data yet.</p>';
  }

  function renderAdvancement(rows) {
    if (!rows.length) {
      return '<p class="muted">No advancement probabilities yet.</p>';
    }
    const byGroup = {};
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      const groupName = row.group || '?';
      if (!byGroup[groupName]) {
        byGroup[groupName] = [];
      }
      byGroup[groupName].push(row);
    }
    const keys = Object.keys(byGroup).sort();
    let html = '';
    for (let i = 0; i < keys.length; i += 1) {
      const groupName = keys[i];
      const groupRows = byGroup[groupName];
      html += '<div class="grp">Group ' + esc(groupName) + '</div>';
      for (let j = 0; j < groupRows.length; j += 1) {
        const row = groupRows[j];
        const className = Number(row.p_advance) >= 0.5 ? 'adv' : (Number(row.p_advance) <= 0.05 ? 'out' : '');
        html += '<div style="margin:.35rem 0"><div style="display:flex;justify-content:space-between;gap:.5rem"><span class="' + className + '">' + esc(row.team || '—') + '</span><span class="muted">adv ' + fmtPct(row.p_advance) + ' · win ' + fmtPct(row.p_rank1) + '</span></div><div class="bar"><i style="width:' + pctWidth(row.p_advance) + '%"></i></div></div>';
      }
    }
    return html;
  }

  function renderRemainingMatches(rows, total) {
    if (!rows.length) {
      return '<p class="muted">No remaining matches.</p>';
    }
    let html = '';
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      html += '<div class="match"><span>#' + fmtInt(row.match_number) + ' ' + esc(row.team_a || '—') + ' v ' + esc(row.team_b || '—') + '</span><span class="muted">' + esc(row.date || '—') + '</span></div>';
    }
    if (Number.isFinite(Number(total)) && total > rows.length) {
      html += '<p class="muted">... ' + fmtInt(Number(total) - rows.length) + ' more</p>';
    }
    return html;
  }

  function renderStandings(rows) {
    if (!rows.length) {
      return '<p class="muted">No group picks available yet.</p>';
    }
    let html = '<table><tr><th>G</th><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr>';
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      html += '<tr><td>' + esc(row.group || '—') + '</td><td>' + esc(row.rank_1 || '—') + '</td><td>' + esc(row.rank_2 || '—') + '</td><td>' + esc(row.rank_3 || '—') + '</td><td>' + esc(row.rank_4 || '—') + '</td></tr>';
    }
    return html + '</table>';
  }

  function renderLast8(rows) {
    if (!rows.length) {
      return '<p class="muted">No last-8 picks yet.</p>';
    }
    const stageOrder = ['quarter_finalist', 'semi_finalist', 'finalist', 'winner'];
    const grouped = {};
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      const stageName = row.stage || 'unknown';
      if (!grouped[stageName]) {
        grouped[stageName] = [];
      }
      grouped[stageName].push(row);
    }
    const keys = Object.keys(grouped).sort(function(a, b) {
      const ai = stageOrder.indexOf(a);
      const bi = stageOrder.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
    let html = '';
    for (let i = 0; i < keys.length; i += 1) {
      const stageName = keys[i];
      html += '<div class="grp">' + esc(stageName.replace(/_/g, ' ')) + '</div>';
      const stageRows = grouped[stageName];
      for (let j = 0; j < stageRows.length; j += 1) {
        const row = stageRows[j];
        html += '<div class="match"><span>' + fmtInt(row.rank) + '. ' + esc(row.team || '—') + '</span><span class="muted">' + fmtPct(row.probability) + '</span></div>';
      }
    }
    return html;
  }

  function renderProjectedFutureBracket(bracketState, advancement) {
    const state = safeObject(bracketState);
    const r32 = safeArray(state.round_of_32_matches);
    let html = '<p class="muted"><b>Projected future bracket.</b> Submitted group-stage predictions stay locked; actual results pin only played matches and future projections use only unplayed fixtures.</p>';
    html += '<p class="muted"><b>Future recommendation:</b> recommendations are future-only and apply only to matches that have not been played.</p>';
    html += '<p class="muted">Bracket state: ' + esc(state.status || 'pending_group_stage') + '. ' + esc(state.note || '') + '</p>';
    if (r32.length) {
      html += '<table><tr><th>#</th><th>Team A</th><th>Team B</th></tr>';
      for (let i = 0; i < r32.length; i += 1) {
        const row = safeObject(r32[i]);
        html += '<tr><td>' + fmtInt(row.match_number) + '</td><td>' + esc(row.team_a || '—') + '</td><td>' + esc(row.team_b || '—') + '</td></tr>';
      }
      html += '</table>';
    } else if (advancement.length) {
      html += '<p class="muted">Current future signal: advancement probabilities combine actual played results with frozen simulations for unplayed group matches.</p>';
    }
    return html;
  }

  function renderValidationNotes(data) {
    const notes = [];
    if (data.tie_break_note) {
      notes.push('<li>' + esc(data.tie_break_note) + '</li>');
    }
    if (data.sim_tie_break_note) {
      notes.push('<li>' + esc(data.sim_tie_break_note) + '</li>');
    }
    const manualReview = safeArray(data.manual_review);
    if (manualReview.length) {
      notes.push('<li>Original manual-review flags are shown below for audit only.</li>');
    } else {
      notes.push('<li>No original manual-review flags were present.</li>');
    }
    let html = '<ul>';
    for (let i = 0; i < notes.length; i += 1) {
      html += notes[i];
    }
    html += '</ul>';
    if (manualReview.length) {
      html += '<div class="grp">Science-only audit</div>';
      for (let i = 0; i < manualReview.length; i += 1) {
        const row = safeObject(manualReview[i]);
        html += '<div class="match"><span>#' + fmtInt(row.match_number) + ' ' + esc(row.team_a || '—') + ' v ' + esc(row.team_b || '—') + '</span><span class="muted">auto-resolved by science-only policy</span></div>';
      }
    }
    return html;
  }

  function renderOverview(data) {
    const summary = safeObject(data.submission_summary);
    const liveSummary = safeObject(data.summary);
    const scoring = safeObject(data.scoring_summary);
    const activeCandidate = safeObject(data.active_candidate);
    // Practical, at-a-glance cards for frozen submissions and live scoring.
    const cards = [
      ['Submitted predictions', fmtInt(summary.total_matches) + ' / 72', 'locked group-stage scores'],
      ['Submitted group standings', fmtInt(safeArray(data.submission_group_standings).length) + ' / 12', 'rank 1-4 per group'],
      ['Submitted Last-8 picks', fmtInt(safeArray(data.submission_last8_picks).length), 'quarters to winner'],
      ['Matches played', fmtInt(liveSummary.matches_played) + ' / 72', fmtInt(liveSummary.matches_remaining) + ' remaining'],
    ];
    if (Number(liveSummary.matches_played) > 0) {
      cards.push(['Points earned', fmtNum(scoring.total_points), fmtNum(scoring.average_points_per_played_match) + ' avg/match']);
    }
    let html = '<div class="summary-grid">';
    for (let i = 0; i < cards.length; i += 1) {
      const card = cards[i];
      html += '<div class="summary-card"><b>' + esc(card[0]) + '</b><span>' + esc(card[1]) + '</span><div class="muted" style="margin-top:.35rem">' + esc(card[2]) + '</div></div>';
    }
    html += '</div>';
    html += '<p class="muted" style="margin-top:.75rem">Submitted prediction files are frozen. Actual result updates are isolated to prediction-vs-actual scoring, the live group table, actual bracket state, and future-only projections.</p>';
    // Audit/policy detail is secondary — kept collapsed, not shown by default.
    html += '<details class="audit-details"><summary>Audit details (how the picks were chosen)</summary><div class="row-meta" style="margin-top:.5rem">'
      + '<span class="pill">Auto-resolved rows: ' + fmtInt(summary.manual_review_rows_auto_resolved) + '</span>'
      + '<span class="pill">EV overrides accepted: ' + fmtInt(summary.ev_overrides_accepted) + '</span>'
      + '<span class="pill">EV overrides rejected: ' + fmtInt(summary.ev_overrides_rejected) + '</span>'
      + '<span class="pill">Safe scores kept: ' + fmtInt(summary.safe_scores_kept) + '</span>'
      + '</div><p class="muted" style="margin:.45rem 0 0">Active candidate: ' + esc(activeCandidate.name || '—') + '. See <code>outputs/reports/score_selection_policy_dashboard_note.md</code> for the full selection policy.</p></details>';
    return html;
  }

  function copyLine(row) {
    if (row.copy_text) {
      return String(row.copy_text);
    }
    const score = row.submitted_score || row.score_to_fill_in || row.final_recommended_score || '—';
    return fmtInt(row.match_number) + '. ' + (row.team_a || '—') + ' ' + score + ' ' + (row.team_b || '—');
  }

  function copyButton(targetId, label) {
    return '<button type="button" onclick="navigator.clipboard.writeText(document.getElementById(&quot;' + targetId + '&quot;).innerText)">' + esc(label) + '</button>';
  }

  function renderSubmissionScores(rows, copyText) {
    if (!rows.length) {
      return '<p class="muted">No submitted score predictions available.</p>';
    }
    const byGroup = {};
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      const group = row.group || '?';
      if (!byGroup[group]) {
        byGroup[group] = [];
      }
      byGroup[group].push(row);
    }
    const groupKeys = Object.keys(byGroup).sort();
    const total = rows.length;

    let html = '<p class="muted"><b>Submitted prediction.</b> These group-stage score predictions are locked/submitted. Actual results can earn points and update live tables while submitted picks remain unchanged.</p>';

    // "All submitted predictions" copy block.
    html += '<div class="copybar"><b>All submitted predictions</b>' + copyButton('scores-copy', 'Copy all scores') + '<span class="muted">' + fmtInt(total) + ' lines</span></div>';
    html += '<pre class="copyblock" id="scores-copy">' + esc(copyText) + '</pre>';

    // Per-group copy blocks (collapsed by default).
    let perGroupCopy = '';
    for (let i = 0; i < groupKeys.length; i += 1) {
      const groupName = groupKeys[i];
      const groupRows = byGroup[groupName];
      const blockId = 'scores-copy-group-' + esc(groupName);
      let lines = '';
      for (let j = 0; j < groupRows.length; j += 1) {
        lines += (j ? '\\n' : '') + copyLine(safeObject(groupRows[j]));
      }
      perGroupCopy += '<div class="copybar"><b>Group ' + esc(groupName) + '</b>' + copyButton(blockId, 'Copy group ' + esc(groupName)) + '</div>';
      perGroupCopy += '<pre class="copyblock" id="' + blockId + '">' + esc(lines) + '</pre>';
    }
    html += '<details class="group-summary"><summary class="group-summary">Per-group copy blocks</summary>' + perGroupCopy + '</details>';

    // Per-match cards: one prominent score, alternatives behind "Why?".
    for (let i = 0; i < groupKeys.length; i += 1) {
      const groupName = groupKeys[i];
      const groupRows = byGroup[groupName];
      let groupHtml = '';
      for (let j = 0; j < groupRows.length; j += 1) {
        const row = safeObject(groupRows[j]);
        const score = row.submitted_score || row.score_to_fill_in || row.final_recommended_score || '—';
        const actualScore = row.actual_score || 'Pending';
        const pointsEarned = row.points_earned == null ? 'Pending' : fmtNum(row.points_earned);
        const policy = String(row.auto_policy_decision || '');
        let policyExplain = 'Selected by the science-only policy.';
        if (policy === 'ev_override_accepted') {
          policyExplain = 'EV alternative accepted: its expected-points uplift cleared the strict override threshold.';
        } else if (String(row.safe_score || '') === String(row.ev_score || '')) {
          policyExplain = 'All scientific sources agreed on this score.';
        } else {
          policyExplain = 'Safe alternative kept: the EV alternative did not clear the strict override threshold.';
        }
        // Primary line: submitted score, actual score, points earned, locked status.
        groupHtml += '<article class="row-card"><div class="row-top"><div><div class="label">Match ' + fmtInt(row.match_number) + (row.date ? ' · ' + esc(row.date) : '') + '</div><div class="row-title">' + esc(row.team_a || '—') + ' vs ' + esc(row.team_b || '—') + '</div><div class="row-meta"><span class="pill">status: ' + esc(row.status || 'locked/submitted') + '</span></div></div><div><div class="label">Submitted prediction</div><div class="score-big">' + esc(score) + '</div><div class="muted" style="margin-top:.25rem">Actual result: ' + esc(actualScore) + '</div><div class="muted">Points earned: ' + esc(pointsEarned) + '</div></div></div>';
        groupHtml += '<div class="row-copy">' + esc(copyLine(row)) + '</div>';
        // Secondary, collapsed audit details.
        groupHtml += '<details class="audit-details"><summary>Why? / Audit details</summary><div class="row-meta">'
          + '<span class="pill">Safe alternative: ' + esc(row.safe_score || '—') + '</span>'
          + '<span class="pill">EV alternative: ' + esc(row.ev_score || '—') + '</span>'
          + '<span class="pill">Consensus/modal score: ' + esc(row.auto_consensus_score || '—') + '</span>'
          + '<span class="pill">Policy decision: ' + esc(row.auto_policy_decision || '—') + '</span>'
          + '</div>'
          + '<p class="muted" style="margin:.45rem 0 0">' + esc(policyExplain) + '</p>'
          + '<p class="muted" style="margin:.25rem 0 0">Reason: ' + esc(row.reason || '—') + '</p>'
          + '</details></article>';
      }
      html += '<details class="group-summary"><summary class="group-summary">Group ' + esc(groupName) + ' (' + fmtInt(groupRows.length) + ' matches)</summary>' + groupHtml + '</details>';
    }
    return html;
  }

  function nextRoundStatusLabel(row) {
    if (row.status === 'played') return 'played';
    if (row.status === 'teams_set') return 'teams_set';
    return 'projected — pending previous results';
  }

  function renderNextRound(data) {
    const matches = safeArray(data.next_round_matches);
    const label = data.next_round_label;
    if (!matches.length || !label) {
      const all = safeArray(data.matches);
      const allPlayed = all.length > 0 && all.every(function(m) { return safeObject(m).status === 'played'; });
      const msg = allPlayed
        ? 'All knockout rounds are complete.'
        : 'No knockout round ready yet — waiting for group stage results.';
      return '<div class="subcard"><h3>Next round to predict</h3><p class="muted">' + esc(msg) + '</p></div>';
    }
    let html = '<div class="subcard"><h3><span>Next round to predict</span><span class="pill">' + esc(label) + '</span></h3>';
    html += '<p class="muted">The full <b>' + esc(label) + '</b> (' + fmtInt(matches.length) + ' matches). Played matches show the <b>Actual result</b>; matches without both real participants yet show the best <b>Projected matchup</b>, labelled <i>projected — pending previous results</i>.</p>';
    html += '<div class="copybar"><b>Copy ' + esc(label) + '</b>' + copyButton('next-round-copy', 'Copy round') + '<span class="muted">' + fmtInt(matches.length) + ' lines</span></div>';
    html += '<pre class="copyblock" id="next-round-copy">' + esc(data.next_round_copy_text || '') + '</pre>';
    for (let j = 0; j < matches.length; j += 1) {
      const row = safeObject(matches[j]);
      const teamA = row.current_team_a || row.projected_team_a || 'TBD';
      const teamB = row.current_team_b || row.projected_team_b || 'TBD';
      const score = row.current_score || row.projected_score || '—';
      const adv = row.current_advancing_team || row.projected_advancing_team || '—';
      const so = row.current_shootout ? '<span class="pill">shoot-out</span>' : '';
      const played = row.status === 'played';
      const bigLabel = played ? 'Actual result' : 'Current recommendation';
      const bigValue = played ? (row.actual_score || score) : score;
      html += '<article class="row-card"><div class="row-top"><div>'
        + '<div class="label">Match ' + fmtInt(row.match_number) + ' · ' + esc(row.round_label || label) + '</div>'
        + '<div class="row-title">' + esc(teamA) + ' vs ' + esc(teamB) + '</div>'
        + '<div class="row-meta"><span class="pill">status: ' + esc(nextRoundStatusLabel(row)) + '</span>' + so + '<span class="pill">advances: ' + esc(adv) + '</span></div>'
        + '</div><div><div class="label">' + bigLabel + '</div><div class="score-big">' + esc(bigValue) + '</div>'
        + (played && row.points_earned_estimate != null ? '<div class="muted" style="margin-top:.25rem">Points (est.): ' + fmtNum(row.points_earned_estimate) + '</div>' : '')
        + '</div></div>';
      html += '<div class="row-copy">' + esc(row.copy_text || '') + '</div>';
      html += '<details class="audit-details"><summary>Projected matchup vs current recommendation</summary><div class="row-meta">'
        + '<span class="pill">Projected matchup: ' + esc(row.projected_team_a || 'TBD') + ' ' + esc(row.projected_score || '—') + ' ' + esc(row.projected_team_b || 'TBD') + ' (adv ' + esc(row.projected_advancing_team || '—') + ')</span>'
        + '<span class="pill">Current recommendation: ' + esc(teamA) + ' ' + esc(score) + ' ' + esc(teamB) + ' (adv ' + esc(adv) + ')</span>'
        + (row.actual_score ? '<span class="pill">Actual result: ' + esc(row.actual_score) + (row.actual_advancing_team ? ' (adv ' + esc(row.actual_advancing_team) + ')' : '') + '</span>' : '')
        + '</div></details></article>';
    }
    html += '</div>';
    return html;
  }

  function renderKnockoutPredictions(knockout) {
    const data = safeObject(knockout);
    const matches = safeArray(data.matches);
    if (!matches.length) {
      return '<p class="muted">No knockout predictions available yet. They are generated from the projected bracket and refresh as actual results arrive.</p>'
        + renderNextRound(data);
    }
    const rounds = safeArray(data.rounds);
    const labels = safeObject(data.round_labels);
    const byRound = safeObject(data.matches_by_round);
    const groupComplete = !!data.group_stage_complete;

    let html = '<p class="muted"><b>Knockout match predictions</b> — predicted exact score, who advances, and the shoot-out call for every match (rules: correct team x odd, +2 exact score, +2 shoot-out). These are <b>future recommendations</b> entered round by round: enter the previous round\'s actual results and the next round\'s real participants resolve here, so you always see predictions for the teams that are actually through. Your submitted group-stage scores stay locked.</p>';
    html += '<p class="muted">Teams shown are ' + (groupComplete ? 'the <b>actual</b> qualifiers.' : 'the model\'s <b>projected</b> bracket until real results arrive.') + ' The <b>projected</b> line is your original up-front gamble; the <b>current</b> line is the refreshed recommendation, so you can compare them.</p>';

    // The full next unresolved round, shown prominently at the top.
    html += renderNextRound(data);

    // "All knockout predictions" copy block.
    html += '<div class="copybar"><b>All knockout predictions</b>' + copyButton('knockout-copy', 'Copy all') + '<span class="muted">' + fmtInt(matches.length) + ' matches</span></div>';
    html += '<pre class="copyblock" id="knockout-copy">' + esc(data.copy_text || '') + '</pre>';

    const roundKeys = rounds.length ? rounds : Object.keys(byRound);
    for (let i = 0; i < roundKeys.length; i += 1) {
      const roundName = roundKeys[i];
      const roundRows = safeArray(byRound[roundName]);
      if (!roundRows.length) {
        continue;
      }
      let roundHtml = '';
      let roundCopy = '';
      for (let j = 0; j < roundRows.length; j += 1) {
        const row = safeObject(roundRows[j]);
        if (row.copy_text) {
          roundCopy += (roundCopy ? '\\n' : '') + row.copy_text;
        }
        const teamA = row.current_team_a || row.projected_team_a || 'TBD';
        const teamB = row.current_team_b || row.projected_team_b || 'TBD';
        const score = row.current_score || row.projected_score || '—';
        const adv = row.current_advancing_team || row.projected_advancing_team || '—';
        const soTag = row.current_shootout ? '<span class="pill">shoot-out</span>' : '';
        const statusPill = '<span class="pill">status: ' + esc(row.status || 'projected') + '</span>';
        const teamsPill = '<span class="pill">teams: ' + esc(row.teams_source || 'projected') + '</span>';
        roundHtml += '<article class="row-card"><div class="row-top"><div><div class="label">Match ' + fmtInt(row.match_number) + ' · ' + esc(row.round_label || roundName) + '</div><div class="row-title">' + esc(teamA) + ' vs ' + esc(teamB) + '</div><div class="row-meta">' + statusPill + teamsPill + '<span class="pill">advances: ' + esc(adv) + '</span>' + soTag + '</div></div><div><div class="label">Predicted score</div><div class="score-big">' + esc(score) + '</div></div></div>';
        // Comparison: projected gamble vs current + actual result when played.
        roundHtml += '<details class="audit-details"><summary>Compare gambles' + (row.status === 'played' ? ' &amp; result' : '') + '</summary><div class="row-meta">'
          + '<span class="pill">Projected: ' + esc(row.projected_team_a || 'TBD') + ' ' + esc(row.projected_score || '—') + ' ' + esc(row.projected_team_b || 'TBD') + ' (adv ' + esc(row.projected_advancing_team || '—') + ')</span>'
          + '<span class="pill">Current: ' + esc(teamA) + ' ' + esc(score) + ' ' + esc(teamB) + ' (adv ' + esc(adv) + ')</span>'
          + (row.actual_score ? '<span class="pill">Actual result: ' + esc(row.actual_score) + (row.actual_advancing_team ? ' (adv ' + esc(row.actual_advancing_team) + ')' : '') + '</span>' : '<span class="pill">Actual result: pending</span>')
          + (row.points_earned_estimate == null ? '' : '<span class="pill">Points (est.): ' + fmtNum(row.points_earned_estimate) + '</span>')
          + '</div></details></article>';
      }
      const blockId = 'knockout-copy-' + esc(roundName);
      let inner = '<div class="copybar"><b>' + esc(labels[roundName] || roundName) + '</b>' + copyButton(blockId, 'Copy round') + '</div>';
      inner += '<pre class="copyblock" id="' + blockId + '">' + esc(roundCopy) + '</pre>' + roundHtml;
      html += '<details class="group-summary"' + (roundName === 'R32' ? ' open' : '') + '><summary class="group-summary">' + esc(labels[roundName] || roundName) + ' (' + fmtInt(roundRows.length) + ' matches)</summary>' + inner + '</details>';
    }
    return html;
  }

  function renderSubmissionStandings(rows) {
    if (!rows.length) {
      return '<p class="muted">No submitted group standings available.</p>';
    }
    let html = '<p class="muted"><b>Group standings to fill in.</b> These are your <b>Submitted prediction</b> of the final group order (rank 1-4), locked/submitted and separate from the live group table.</p>';
    html += '<table class="aligned standings-table">'
      + '<colgroup><col class="gcol"><col><col><col><col></colgroup>'
      + '<tr><th>G</th><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr>';
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      html += '<tr><td>' + esc(row.group || '—') + '</td><td>' + esc(row.rank_1 || '—') + '</td><td>' + esc(row.rank_2 || '—') + '</td><td>' + esc(row.rank_3 || '—') + '</td><td>' + esc(row.rank_4 || '—') + '</td></tr>';
    }
    return html + '</table>';
  }

  function renderSubmissionLast8(rows) {
    if (!rows.length) {
      return '<p class="muted">No submitted Last-8 picks available.</p>';
    }
    const stageOrder = ['quarter_finalist', 'semi_finalist', 'finalist', 'winner'];
    const grouped = {};
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      const stageName = row.stage || 'unknown';
      if (!grouped[stageName]) {
        grouped[stageName] = [];
      }
      grouped[stageName].push(row);
    }
    const keys = Object.keys(grouped).sort(function(a, b) {
      const ai = stageOrder.indexOf(a);
      const bi = stageOrder.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
    let html = '<p class="muted"><b>Submitted prediction.</b> These Last-8 / progression picks are locked/submitted. Future recommendation output must stay future-only.</p>';
    for (let i = 0; i < keys.length; i += 1) {
      const stageName = keys[i];
      const stageRows = grouped[stageName];
      html += '<details class="group-summary" open><summary class="group-summary">' + esc(stageName.replace(/_/g, ' ')) + ' (' + fmtInt(stageRows.length) + ' picks)</summary>';
      for (let j = 0; j < stageRows.length; j += 1) {
        const row = safeObject(stageRows[j]);
        html += '<div class="row-card"><div class="row-top"><div><div class="label">Rank ' + fmtInt(row.rank) + '</div><div class="row-title">' + esc(row.team || '—') + '</div></div><div class="score-big" style="font-size:1.2rem;color:#7dd3fc">' + fmtPct(row.probability) + '</div></div><div class="row-meta"><span class="pill">' + esc(stageName.replace(/_/g, ' ')) + '</span><span class="pill">' + esc(row.selection_type || '—') + '</span></div></div>';
      }
      html += '</details>';
    }
    return html;
  }

  // Progressive enhancement only. Every section is already server-rendered into
  // #app at build time, so the dashboard is fully usable with JavaScript off or
  // broken. Here we only hydrate the header lines and wire the tab anchors; on
  // any failure we keep the pre-rendered content and just prepend a banner.
  function setupTabs() {
    const nav = document.getElementById('tabs');
    if (!nav) return;
    const links = nav.querySelectorAll('a[href^="#"]');
    for (let i = 0; i < links.length; i += 1) {
      links[i].addEventListener('click', function() {
        // Sections stay in the document; the anchor simply scrolls to them.
        // (Kept as a hook for future show/hide filtering without hiding by default.)
      });
    }
  }

  function applyTheme(theme) {
    const normalized = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', normalized);
    const buttons = document.querySelectorAll('[data-theme-choice]');
    for (let i = 0; i < buttons.length; i += 1) {
      const active = buttons[i].getAttribute('data-theme-choice') === normalized;
      buttons[i].setAttribute('aria-pressed', active ? 'true' : 'false');
    }
    return normalized;
  }

  function setupThemeToggle() {
    let saved = 'light';
    try {
      saved = localStorage.getItem('travelModeTheme') || 'light';
    } catch (err) {
      saved = 'light';
    }
    applyTheme(saved);
    const buttons = document.querySelectorAll('[data-theme-choice]');
    for (let i = 0; i < buttons.length; i += 1) {
      buttons[i].addEventListener('click', function() {
        const next = applyTheme(this.getAttribute('data-theme-choice'));
        try {
          localStorage.setItem('travelModeTheme', next);
        } catch (err) {
          // Preference persistence is optional; the light CSS default still works.
        }
      });
    }
  }

  function hydrate(data) {
    try {
      const summary = safeObject(data.summary);
      const activeCandidate = safeObject(data.active_candidate);
      if (metaEl) {
        const sims = Number.isFinite(Number(summary.n_sims)) ? Number(summary.n_sims).toLocaleString() + ' sims' : 'no sims';
        metaEl.textContent = 'Updated ' + (data.generated_at || '—') + ' · ' + fmtInt(summary.matches_played) + '/72 played · ' + sims;
      }
      if (candEl) {
        candEl.textContent = 'Active candidate: ' + (activeCandidate.name || '—') + ' (' + (activeCandidate.active_candidate_dir || '—') + ')';
      }
      setupTabs();
      setupThemeToggle();
    } catch (err) {
      showError('Enhancement failed', err, data);
    }
  }

  let rawData = null;
  try {
    const rawInline = parseInlinePayload();
    if (rawInline === null) {
      // No inline payload: header sections are still server-rendered; try to
      // hydrate the header from the JSON file but never blank the content.
      fetchPayload()
        .then(function(raw) {
          hydrate(validatePayload(raw));
        })
        .catch(function(err) {
          showError('Could not load dashboard data (showing pre-rendered content)', err, null);
        });
      return;
    }

    rawData = rawInline;
    const data = validatePayload(rawData);
    hydrate(data);
  } catch (err) {
    showError('Initialization failed', err, rawData);
    return;
  }

  // Legacy full client renderer, retained as a non-default fallback. The static
  // server-rendered sections are authoritative; this is intentionally not called
  // on load so a render bug can never blank the page.
  function renderDashboard(data) {
    try {
      const summary = safeObject(data.summary);
      const activeCandidate = safeObject(data.active_candidate);
      const scoringSummary = safeObject(data.scoring_summary);
      const submissionSummary = safeObject(data.submission_summary);
      const groups = safeObject(data.groups);
      const advancement = safeArray(data.advancement);
      const playedMatches = safeArray(data.played_matches);
      const predictionVsActual = safeArray(data.prediction_vs_actual);
      const remainingMatches = safeArray(data.remaining_matches);
      const submissionScores = safeArray(data.submission_score_predictions);
      const submissionStandings = safeArray(data.submission_group_standings);
      const submissionLast8 = safeArray(data.submission_last8_picks);
      const manualReview = safeArray(data.manual_review);
      const bracketState = safeObject(data.actual_bracket_state);
      const byGroup = safeObject(scoringSummary.total_by_group);

      if (metaEl) {
        const sims = Number.isFinite(Number(summary.n_sims)) ? Number(summary.n_sims).toLocaleString() + ' sims' : 'no sims';
        metaEl.textContent = 'Updated ' + (data.generated_at || '—') + ' · ' + fmtInt(summary.matches_played) + '/72 played · ' + sims;
      }
      if (candEl) {
        candEl.textContent = 'Active candidate: ' + (activeCandidate.name || '—') + ' (' + (activeCandidate.active_candidate_dir || '—') + ')';
      }

      let activeHtml = '<table><tr><td>Candidate</td><td>' + esc(activeCandidate.name || '—') + '</td></tr><tr><td>Directory</td><td>' + esc(activeCandidate.active_candidate_dir || '—') + '</td></tr><tr><td>Scores</td><td>' + esc(activeCandidate.score_predictions_file || '—') + '</td></tr><tr><td>Standings</td><td>' + esc(activeCandidate.standing_predictions_file || '—') + '</td></tr><tr><td>Last-8</td><td>' + esc(activeCandidate.last8_predictions_file || '—') + '</td></tr><tr><td>Last updated</td><td>' + esc(data.generated_at || '—') + '</td></tr></table>';
      if (data.repo_slug) {
        activeHtml += '<p><a href="https://github.com/' + esc(data.repo_slug) + '/actions/workflows/travel_mode_update.yml">Open the single-match workflow</a></p>';
      } else {
        activeHtml += '<p class="muted">Set TRAVEL_MODE_REPO to enable the workflow link.</p>';
      }
      const overviewHtml = renderOverview(data);
      const submitHtml = renderSubmissionScores(submissionScores, data.submission_score_copy_text || '');
      const standingsHtml = renderSubmissionStandings(submissionStandings);
      const last8Html = renderSubmissionLast8(submissionLast8);
      const knockoutHtml = renderKnockoutPredictions(data.knockout_predictions);

      let instructionsHtml = '<p><b>Score input instructions:</b> Use the GitHub Actions workflow or an issue comment with a <code>/WK-SCORES</code> block.</p>';
      instructionsHtml += '<p class="muted">The dashboard uses the inline payload first and only fetches <code>./mobile_dashboard_data.json</code> when the inline payload is absent.</p>';

      const playedHtml = renderPlayedMatches(playedMatches);

      let scoringHtml = '';
      if (Number(summary.matches_played) > 0) {
        scoringHtml = '<div class="bignum">' + fmtNum(scoringSummary.total_points) + ' pts</div><p class="muted">' + fmtInt(scoringSummary.played_matches) + ' played · ' + fmtNum(scoringSummary.possible_points_for_played_matches) + ' possible · ' + fmtNum(scoringSummary.points_missed) + ' missed · ' + fmtNum(scoringSummary.average_points_per_played_match) + ' avg</p>';
        scoringHtml += '<div class="stats"><div class="stat"><b>' + fmtInt(scoringSummary.outcomes_correct) + '/' + fmtInt(scoringSummary.played_matches) + '</b><span>outcomes</span></div><div class="stat"><b>' + fmtInt(scoringSummary.goal_differences_correct) + '/' + fmtInt(scoringSummary.played_matches) + '</b><span>goal diffs</span></div><div class="stat"><b>' + fmtInt(scoringSummary.exact_scores_correct) + '/' + fmtInt(scoringSummary.played_matches) + '</b><span>exact scores</span></div></div>';
        const groupKeys = Object.keys(byGroup).sort();
        if (groupKeys.length) {
          scoringHtml += '<div class="grp">By group</div><table><tr><th>G</th><th>MP</th><th>Pts</th><th>Poss</th></tr>';
          for (let i = 0; i < groupKeys.length; i += 1) {
            const groupName = groupKeys[i];
            const row = safeObject(byGroup[groupName]);
            scoringHtml += '<tr><td>' + esc(groupName) + '</td><td>' + fmtInt(row.played_matches) + '</td><td>' + fmtNum(row.total_points) + '</td><td>' + fmtNum(row.possible_points) + '</td></tr>';
          }
          scoringHtml += '</table>';
        }
      } else {
        scoringHtml = '<p class="muted">No scoring summary yet.</p>';
      }

      const pvaHtml = renderPredictionVsActual(predictionVsActual);
      const groupsHtml = renderGroups(groups);
      const incentivesHtml = '<p class="muted">Final group-match incentive diagnostics are live context only. They do not change submitted predictions.</p>';
      const advancementHtml = renderAdvancement(advancement);
      const projectedFutureBracketHtml = renderProjectedFutureBracket(bracketState, advancement);
      const remainingHtml = renderRemainingMatches(remainingMatches, data.remaining_matches_total);
      const auditHtml = renderValidationNotes({ tie_break_note: data.tie_break_note, sim_tie_break_note: data.sim_tie_break_note, manual_review: manualReview });
      const sims = Number.isFinite(Number(summary.n_sims)) ? Number(summary.n_sims).toLocaleString() + ' sims' : null;
      // Live results: tracking only, shown after the submission sections.
      const liveHtml = [
        buildSubcard('Score input instructions', instructionsHtml),
        buildSubcard('Played matches', playedHtml, fmtInt(summary.matches_played) + '/72'),
        buildSubcard('Points earned / scoring summary', scoringHtml),
        buildSubcard('Remaining matches', remainingHtml, fmtInt(summary.matches_remaining) + ' left')
      ].join('');
      // Audit/validation kept secondary and collapsed (not an action item).
      const auditCollapsed = '<details class="audit-details"><summary>Audit &amp; validation notes</summary>' + auditHtml + '</details>';

      const filesHtml = '<div class="filelinks"><a href="prediction_vs_actual.csv">pva.csv</a> <a href="scoring_summary.csv">score.csv</a> <a href="live_group_tables.csv">tables.csv</a> <a href="mobile_dashboard_data.json">data.json</a></div>';

      const liveGroupTablesHtml = '<p class="muted">Live group table built from actual played results. Your submitted group standings above stay locked.</p>' + groupsHtml;
      const bracketHtml = '<p class="muted">Advancement combines actual played results with frozen simulations of unplayed matches.</p>' + advancementHtml + projectedFutureBracketHtml;

      app.innerHTML = [
        buildSection('overview', 'Overview', 'live', overviewHtml),
        buildSection('submitted-scores', 'Scores to fill in', fmtInt(submissionScores.length) + ' matches', submitHtml),
        buildSection('group-standings', 'Group standings to fill in', 'submitted', standingsHtml),
        buildSection('last8', 'Last-8 to fill in', 'submitted', last8Html),
        buildSection('knockout-predictions', 'Knockout predictions', safeArray(safeObject(data.knockout_predictions).matches).length + ' matches', knockoutHtml),
        buildSection('live-results', 'Live results', null, liveHtml + filesHtml + auditCollapsed),
        buildSection('prediction-vs-actual', 'Prediction vs Actual', fmtInt(predictionVsActual.length) + ' scored', pvaHtml),
        buildSection('live-group-tables', 'Live group tables', null, liveGroupTablesHtml),
        buildSection('incentives', 'Final group incentives', null, incentivesHtml),
        buildSection('advancement', 'Advancement / bracket projections', sims, bracketHtml)
      ].join('');
    } catch (err) {
      showError('Render failed', err, data);
    }
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server-side rendering (SSR).
#
# The dashboard is pre-rendered to static HTML here at build time so the page is
# fully useful even if the client JavaScript fails to run. The inline <script>
# only *enhances* (hydrates meta/candidate lines, re-renders identical sections,
# wires tab anchors); on any JS error the pre-rendered sections remain and a
# small banner is prepended. No section is created only by JavaScript.
# ---------------------------------------------------------------------------


def _esc(value) -> str:
    if value is None:
        return ""
    return _escape(str(value), quote=True)


def _fi(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return str(round(f)) if math.isfinite(f) else "—"


def _fnum(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    r = round(f * 100) / 100
    return str(int(r)) if r == int(r) else str(r)


def _fpct(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{round(f * 100)}%" if math.isfinite(f) else "—"


def _pctw(value) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f):
        return 0
    return max(0, min(100, round(f * 100)))


def _obj(value) -> dict:
    return value if isinstance(value, dict) else {}


def _arr(value) -> list:
    return value if isinstance(value, list) else []


def _section(section_id: str, title: str, pill, body: str) -> str:
    pill_html = f'<span class="pill">{_esc(pill)}</span>' if pill else ""
    open_attr = " open" if SECTION_DEFAULT_OPEN.get(section_id, False) else ""
    return (
        f'<details class="dashboard-section" id="{section_id}"{open_attr}>'
        f'<summary><h2 class="section-title">{_esc(title)}{pill_html}</h2></summary>'
        f'<div class="section-body">{body}</div></details>'
    )


def _subcard(title: str, body: str, right=None) -> str:
    right_html = f'<span class="pill">{_esc(right)}</span>' if right else ""
    return (
        f'<div class="subcard"><h3><span>{_esc(title)}</span>{right_html}</h3>'
        f'{body}</div>'
    )


def _copy_btn(target_id: str, label: str) -> str:
    return (
        '<button type="button" onclick="navigator.clipboard.writeText('
        f"document.getElementById('{target_id}').innerText)\">{_esc(label)}</button>"
    )


def _render_overview(data: dict) -> str:
    summary = _obj(data.get("submission_summary"))
    live = _obj(data.get("summary"))
    scoring = _obj(data.get("scoring_summary"))
    candidate = _obj(data.get("active_candidate"))
    cards = [
        ("Scores to fill in", f"{_fi(summary.get('total_matches'))} / 72", "one score per group-stage match"),
        ("Group standings", f"{_fi(len(_arr(data.get('submission_group_standings'))))} / 12", "rank 1-4 per group"),
        ("Last-8 picks", _fi(len(_arr(data.get("submission_last8_picks")))), "quarters → winner"),
        ("Matches played", f"{_fi(live.get('matches_played'))} / 72", f"{_fi(live.get('matches_remaining'))} remaining"),
    ]
    if float(live.get("matches_played") or 0) > 0:
        cards.append(("Points so far", _fnum(scoring.get("total_points")), f"{_fnum(scoring.get('average_points_per_played_match'))} avg/match"))
    html = '<div class="summary-grid">'
    for label, value, sub in cards:
        html += (
            f'<div class="summary-card"><b>{_esc(label)}</b><span>{_esc(value)}</span>'
            f'<div class="muted" style="margin-top:.35rem">{_esc(sub)}</div></div>'
        )
    html += "</div>"
    html += (
        '<p class="muted" style="margin-top:.75rem">Fill in the game from the sections '
        "below: <b>Scores to fill in</b> (72 match scores), <b>Group standings to fill in</b>, "
        "<b>Last-8 to fill in</b>, and <b>Knockout predictions</b>. Live tracking sits "
        "further down once matches are played. Submitted prediction files are frozen; "
        "actual results only update scoring, live tables, and future projections.</p>"
    )
    html += (
        '<details class="audit-details"><summary>Audit details (how the picks were chosen)</summary>'
        '<div class="row-meta" style="margin-top:.5rem">'
        f'<span class="pill">Auto-resolved rows: {_fi(summary.get("manual_review_rows_auto_resolved"))}</span>'
        f'<span class="pill">EV overrides accepted: {_fi(summary.get("ev_overrides_accepted"))}</span>'
        f'<span class="pill">EV overrides rejected: {_fi(summary.get("ev_overrides_rejected"))}</span>'
        f'<span class="pill">Safe scores kept: {_fi(summary.get("safe_scores_kept"))}</span>'
        "</div>"
        f'<p class="muted" style="margin:.45rem 0 0">Active candidate: {_esc(candidate.get("name") or "—")} '
        f'({_esc(candidate.get("active_candidate_dir") or "—")}).</p></details>'
    )
    return html


def _render_submission_scores(data: dict) -> str:
    rows = _arr(data.get("submission_score_predictions"))
    copy_text = data.get("submission_score_copy_text") or ""
    if not rows:
        return '<p class="muted">No submitted score predictions available.</p>'
    by_group: dict[str, list] = {}
    for row in rows:
        by_group.setdefault(_obj(row).get("group") or "?", []).append(_obj(row))
    group_keys = sorted(by_group)

    html = (
        '<p class="muted"><b>All 72 group-stage scores to fill in</b>, grouped A-L. '
        "One score per match — the big green number. These are your <b>Submitted prediction</b> "
        "(locked/submitted); copy the whole list or one group at a time below.</p>"
    )
    html += (
        f'<div class="copybar"><b>All submitted predictions</b>{_copy_btn("scores-copy", "Copy all scores")}'
        f'<span class="muted">{_fi(len(rows))} lines</span></div>'
    )
    html += f'<pre class="copyblock" id="scores-copy">{_esc(copy_text)}</pre>'

    per_group = ""
    for group in group_keys:
        block_id = f"scores-copy-group-{_esc(group)}"
        lines = "\n".join(
            (r.get("copy_text") or f"{_fi(r.get('match_number'))}. {r.get('team_a')} {r.get('score_to_fill_in') or r.get('submitted_score') or ''} {r.get('team_b')}")
            for r in by_group[group]
        )
        per_group += f'<div class="copybar"><b>Group {_esc(group)}</b>{_copy_btn(block_id, "Copy group " + group)}</div>'
        per_group += f'<pre class="copyblock" id="{block_id}">{_esc(lines)}</pre>'
    html += f'<details class="group-summary"><summary class="group-summary">Per-group copy blocks</summary>{per_group}</details>'

    for group in group_keys:
        group_html = ""
        for row in by_group[group]:
            score = row.get("submitted_score") or row.get("score_to_fill_in") or row.get("final_recommended_score") or "—"
            actual_score = row.get("actual_score") or "Pending"
            points = row.get("points_earned")
            points_txt = "Pending" if points is None else _fnum(points)
            policy = str(row.get("auto_policy_decision") or "")
            if policy == "ev_override_accepted":
                explain = "EV alternative accepted: its expected-points uplift cleared the strict override threshold."
            elif str(row.get("safe_score") or "") == str(row.get("ev_score") or ""):
                explain = "All scientific sources agreed on this score."
            else:
                explain = "Safe alternative kept: the EV alternative did not clear the strict override threshold."
            date = f' · {_esc(row.get("date"))}' if row.get("date") else ""
            copy_line = row.get("copy_text") or f"{_fi(row.get('match_number'))}. {row.get('team_a')} {score} {row.get('team_b')}"
            group_html += (
                '<article class="row-card"><div class="row-top"><div>'
                f'<div class="label">Match {_fi(row.get("match_number"))}{date}</div>'
                f'<div class="row-title">{_esc(row.get("team_a") or "—")} vs {_esc(row.get("team_b") or "—")}</div>'
                f'<div class="row-meta"><span class="pill">status: {_esc(row.get("status") or "locked/submitted")}</span></div>'
                '</div><div><div class="label">Submitted prediction</div>'
                f'<div class="score-big">{_esc(score)}</div>'
                f'<div class="muted" style="margin-top:.25rem">Actual result: {_esc(actual_score)}</div>'
                f'<div class="muted">Points earned: {_esc(points_txt)}</div></div></div>'
                f'<div class="row-copy">{_esc(copy_line)}</div>'
                '<details class="audit-details"><summary>Why? / Audit details</summary><div class="row-meta">'
                f'<span class="pill">Safe alternative: {_esc(row.get("safe_score") or "—")}</span>'
                f'<span class="pill">EV alternative: {_esc(row.get("ev_score") or "—")}</span>'
                f'<span class="pill">Consensus/modal score: {_esc(row.get("auto_consensus_score") or "—")}</span>'
                f'<span class="pill">Policy decision: {_esc(row.get("auto_policy_decision") or "—")}</span>'
                "</div>"
                f'<p class="muted" style="margin:.45rem 0 0">{_esc(explain)}</p>'
                f'<p class="muted" style="margin:.25rem 0 0">Reason: {_esc(row.get("reason") or "—")}</p>'
                "</details></article>"
            )
        html += (
            f'<details class="group-summary"><summary class="group-summary">Group {_esc(group)} '
            f'({_fi(len(by_group[group]))} matches)</summary>{group_html}</details>'
        )
    return html


def _render_submission_standings(data: dict) -> str:
    rows = _arr(data.get("submission_group_standings"))
    if not rows:
        return '<p class="muted">No submitted group standings available.</p>'
    html = (
        '<p class="muted"><b>Group standings to fill in.</b> Your <b>Submitted prediction</b> of '
        "the final group order (rank 1-4), locked/submitted and separate from the live group table.</p>"
    )
    html += (
        '<table class="aligned standings-table">'
        '<colgroup><col class="gcol"><col><col><col><col></colgroup>'
        "<tr><th>G</th><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr>"
    )
    for row in rows:
        row = _obj(row)
        html += (
            f'<tr><td>{_esc(row.get("group") or "—")}</td>'
            f'<td>{_esc(row.get("rank_1") or "—")}</td><td>{_esc(row.get("rank_2") or "—")}</td>'
            f'<td>{_esc(row.get("rank_3") or "—")}</td><td>{_esc(row.get("rank_4") or "—")}</td></tr>'
        )
    return html + "</table>"


_STAGE_ORDER = ["quarter_finalist", "semi_finalist", "finalist", "winner"]


def _render_submission_last8(data: dict) -> str:
    rows = _arr(data.get("submission_last8_picks"))
    if not rows:
        return '<p class="muted">No submitted Last-8 picks available.</p>'
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(_obj(row).get("stage") or "unknown", []).append(_obj(row))

    def order(name: str) -> tuple:
        return (_STAGE_ORDER.index(name) if name in _STAGE_ORDER else len(_STAGE_ORDER), name)

    html = (
        '<p class="muted"><b>Last-8 to fill in.</b> These Last-8 / progression picks are your '
        "<b>Submitted prediction</b> (locked/submitted). Future recommendation output stays future-only.</p>"
    )
    for stage in sorted(grouped, key=order):
        stage_rows = grouped[stage]
        label = stage.replace("_", " ")
        html += f'<details class="group-summary" open><summary class="group-summary">{_esc(label)} ({_fi(len(stage_rows))} picks)</summary>'
        for row in stage_rows:
            html += (
                '<div class="row-card"><div class="row-top"><div>'
                f'<div class="label">Rank {_fi(row.get("rank"))}</div>'
                f'<div class="row-title">{_esc(row.get("team") or "—")}</div></div>'
                f'<div class="score-big" style="font-size:1.2rem;color:#7dd3fc">{_fpct(row.get("probability"))}</div></div>'
                f'<div class="row-meta"><span class="pill">{_esc(label)}</span>'
                f'<span class="pill">{_esc(row.get("selection_type") or "—")}</span></div></div>'
            )
        html += "</details>"
    return html


def _knockout_status_label(row: dict) -> str:
    status = row.get("status")
    if status == "played":
        return "played"
    if status == "teams_set":
        return "teams_set"
    return "projected — pending previous results"


def _render_next_round(k: dict) -> str:
    matches = _arr(k.get("next_round_matches"))
    label = k.get("next_round_label")
    if not matches or not label:
        all_m = _arr(k.get("matches"))
        all_played = bool(all_m) and all(_obj(m).get("status") == "played" for m in all_m)
        msg = (
            "All knockout rounds are complete."
            if all_played
            else "No knockout round ready yet — waiting for group stage results."
        )
        return f'<div class="subcard"><h3>Next round to predict</h3><p class="muted">{_esc(msg)}</p></div>'
    html = (
        f'<div class="subcard"><h3><span>Next round to predict</span><span class="pill">{_esc(label)}</span></h3>'
        f'<p class="muted">The full <b>{_esc(label)}</b> ({_fi(len(matches))} matches). Played matches show '
        "the <b>Actual result</b>; matches without both real participants yet show the best "
        "<b>Projected matchup</b>, labelled <i>projected — pending previous results</i>.</p>"
    )
    html += (
        f'<div class="copybar"><b>Copy {_esc(label)}</b>{_copy_btn("next-round-copy", "Copy round")}'
        f'<span class="muted">{_fi(len(matches))} lines</span></div>'
    )
    html += f'<pre class="copyblock" id="next-round-copy">{_esc(k.get("next_round_copy_text") or "")}</pre>'
    for row in matches:
        row = _obj(row)
        team_a = row.get("current_team_a") or row.get("projected_team_a") or "TBD"
        team_b = row.get("current_team_b") or row.get("projected_team_b") or "TBD"
        score = row.get("current_score") or row.get("projected_score") or "—"
        adv = row.get("current_advancing_team") or row.get("projected_advancing_team") or "—"
        so = '<span class="pill">shoot-out</span>' if row.get("current_shootout") else ""
        played = row.get("status") == "played"
        big_label = "Actual result" if played else "Current recommendation"
        big_value = (row.get("actual_score") or score) if played else score
        pts = ""
        if played and row.get("points_earned_estimate") is not None:
            pts = f'<div class="muted" style="margin-top:.25rem">Points (est.): {_fnum(row.get("points_earned_estimate"))}</div>'
        actual_pill = (
            f'<span class="pill">Actual result: {_esc(row.get("actual_score"))}'
            + (f' (adv {_esc(row.get("actual_advancing_team"))})' if row.get("actual_advancing_team") else "")
            + "</span>"
            if row.get("actual_score")
            else ""
        )
        html += (
            '<article class="row-card"><div class="row-top"><div>'
            f'<div class="label">Match {_fi(row.get("match_number"))} · {_esc(row.get("round_label") or label)}</div>'
            f'<div class="row-title">{_esc(team_a)} vs {_esc(team_b)}</div>'
            f'<div class="row-meta"><span class="pill">status: {_esc(_knockout_status_label(row))}</span>{so}'
            f'<span class="pill">advances: {_esc(adv)}</span></div>'
            f'</div><div><div class="label">{big_label}</div><div class="score-big">{_esc(big_value)}</div>{pts}</div></div>'
            f'<div class="row-copy">{_esc(row.get("copy_text") or "")}</div>'
            '<details class="audit-details"><summary>Projected matchup vs current recommendation</summary><div class="row-meta">'
            f'<span class="pill">Projected matchup: {_esc(row.get("projected_team_a") or "TBD")} {_esc(row.get("projected_score") or "—")} {_esc(row.get("projected_team_b") or "TBD")} (adv {_esc(row.get("projected_advancing_team") or "—")})</span>'
            f'<span class="pill">Current recommendation: {_esc(team_a)} {_esc(score)} {_esc(team_b)} (adv {_esc(adv)})</span>'
            f"{actual_pill}</div></details></article>"
        )
    html += "</div>"
    return html


def _render_knockout(data: dict) -> str:
    k = _obj(data.get("knockout_predictions"))
    matches = _arr(k.get("matches"))
    if not matches:
        return (
            '<p class="muted">No knockout predictions available yet. They are generated from the '
            "projected bracket and refresh as actual results arrive.</p>" + _render_next_round(k)
        )
    group_complete = bool(k.get("group_stage_complete"))
    labels = _obj(k.get("round_labels"))
    by_round = _obj(k.get("matches_by_round"))
    rounds = _arr(k.get("rounds")) or list(by_round)

    html = (
        '<p class="muted"><b>Knockout match predictions</b> — predicted exact score, who advances, '
        "and the shoot-out call for every match. These are <b>future recommendations</b> entered "
        "round by round; your submitted group-stage scores stay locked.</p>"
    )
    html += (
        '<p class="muted">Teams shown are '
        + ("the <b>actual</b> qualifiers." if group_complete else "the model's <b>projected</b> bracket until real results arrive.")
        + " The <b>projected</b> line is your original up-front gamble; the <b>current</b> line is the refreshed recommendation.</p>"
    )
    html += _render_next_round(k)
    html += (
        f'<div class="copybar"><b>All knockout predictions</b>{_copy_btn("knockout-copy", "Copy all")}'
        f'<span class="muted">{_fi(len(matches))} matches</span></div>'
    )
    html += f'<pre class="copyblock" id="knockout-copy">{_esc(k.get("copy_text") or "")}</pre>'

    for round_name in rounds:
        round_rows = _arr(by_round.get(round_name))
        if not round_rows:
            continue
        round_label = labels.get(round_name) or round_name
        round_copy = "\n".join(_obj(r).get("copy_text") or "" for r in round_rows if _obj(r).get("copy_text"))
        block_id = f"knockout-copy-{_esc(round_name)}"
        inner = f'<div class="copybar"><b>{_esc(round_label)}</b>{_copy_btn(block_id, "Copy round")}</div>'
        inner += f'<pre class="copyblock" id="{block_id}">{_esc(round_copy)}</pre>'
        for row in round_rows:
            row = _obj(row)
            team_a = row.get("current_team_a") or row.get("projected_team_a") or "TBD"
            team_b = row.get("current_team_b") or row.get("projected_team_b") or "TBD"
            score = row.get("current_score") or row.get("projected_score") or "—"
            adv = row.get("current_advancing_team") or row.get("projected_advancing_team") or "—"
            so = '<span class="pill">shoot-out</span>' if row.get("current_shootout") else ""
            inner += (
                '<article class="row-card"><div class="row-top"><div>'
                f'<div class="label">Match {_fi(row.get("match_number"))} · {_esc(row.get("round_label") or round_label)}</div>'
                f'<div class="row-title">{_esc(team_a)} vs {_esc(team_b)}</div>'
                f'<div class="row-meta"><span class="pill">status: {_esc(row.get("status") or "projected")}</span>'
                f'<span class="pill">teams: {_esc(row.get("teams_source") or "projected")}</span>'
                f'<span class="pill">advances: {_esc(adv)}</span>{so}</div>'
                f'</div><div><div class="label">Predicted score</div><div class="score-big">{_esc(score)}</div></div></div>'
                f'<div class="row-copy">{_esc(row.get("copy_text") or "")}</div>'
                '<details class="audit-details"><summary>Compare gambles</summary><div class="row-meta">'
                f'<span class="pill">Projected: {_esc(row.get("projected_team_a") or "TBD")} {_esc(row.get("projected_score") or "—")} {_esc(row.get("projected_team_b") or "TBD")} (adv {_esc(row.get("projected_advancing_team") or "—")})</span>'
                f'<span class="pill">Current: {_esc(team_a)} {_esc(score)} {_esc(team_b)} (adv {_esc(adv)})</span>'
                + (f'<span class="pill">Actual result: {_esc(row.get("actual_score"))}</span>' if row.get("actual_score") else "")
                + "</div></details></article>"
            )
        open_attr = " open" if round_name == "R32" else ""
        html += (
            f'<details class="group-summary"{open_attr}><summary class="group-summary">{_esc(round_label)} '
            f'({_fi(len(round_rows))} matches)</summary>{inner}</details>'
        )
    return html


def _render_played(data: dict) -> str:
    rows = _arr(data.get("played_matches"))
    if not rows:
        return '<p class="muted">No matches played yet.</p>'
    html = ""
    for row in rows:
        row = _obj(row)
        html += (
            f'<div class="match"><span>#{_fi(row.get("match_number"))} {_esc(row.get("team_a") or "—")} v '
            f'{_esc(row.get("team_b") or "—")}</span><span class="sc">{_fi(row.get("team_a_goals"))}–{_fi(row.get("team_b_goals"))}</span></div>'
        )
    return html


def _render_scoring(data: dict) -> str:
    summary = _obj(data.get("summary"))
    scoring = _obj(data.get("scoring_summary"))
    if float(summary.get("matches_played") or 0) <= 0:
        return '<p class="muted">No scoring summary yet.</p>'
    html = (
        f'<div class="bignum">{_fnum(scoring.get("total_points"))} pts</div>'
        f'<p class="muted">{_fi(scoring.get("played_matches"))} played · '
        f'{_fnum(scoring.get("possible_points_for_played_matches"))} possible · '
        f'{_fnum(scoring.get("points_missed"))} missed · '
        f'{_fnum(scoring.get("average_points_per_played_match"))} avg</p>'
    )
    html += (
        '<div class="stats">'
        f'<div class="stat"><b>{_fi(scoring.get("outcomes_correct"))}/{_fi(scoring.get("played_matches"))}</b><span>outcomes</span></div>'
        f'<div class="stat"><b>{_fi(scoring.get("goal_differences_correct"))}/{_fi(scoring.get("played_matches"))}</b><span>goal diffs</span></div>'
        f'<div class="stat"><b>{_fi(scoring.get("exact_scores_correct"))}/{_fi(scoring.get("played_matches"))}</b><span>exact scores</span></div></div>'
    )
    return html


def _render_remaining(data: dict) -> str:
    rows = _arr(data.get("remaining_matches"))
    total = data.get("remaining_matches_total")
    if not rows:
        return '<p class="muted">No remaining matches.</p>'
    html = ""
    for row in rows:
        row = _obj(row)
        html += (
            f'<div class="match"><span>#{_fi(row.get("match_number"))} {_esc(row.get("team_a") or "—")} v '
            f'{_esc(row.get("team_b") or "—")}</span><span class="muted">{_esc(row.get("date") or "—")}</span></div>'
        )
    try:
        if total is not None and float(total) > len(rows):
            html += f'<p class="muted">... {_fi(float(total) - len(rows))} more</p>'
    except (TypeError, ValueError):
        pass
    return html


def _render_live_results(data: dict) -> str:
    summary = _obj(data.get("summary"))
    instructions = (
        '<p><b>Score input instructions:</b> Use the GitHub Actions workflow or an issue comment '
        "with a <code>/WK-SCORES</code> block (group matches 1-72, knockout 73-104).</p>"
        '<p class="muted">Actual results update live tables, scoring and projections only — never '
        "the submitted predictions.</p>"
    )
    audit = (
        '<details class="audit-details"><summary>Audit &amp; validation notes</summary>'
        '<ul><li>Submitted group predictions are frozen; only live tables, scoring and '
        "projections move.</li><li>Tie-breaks use points → GD → GF → name.</li></ul></details>"
    )
    return (
        _subcard("Score input instructions", instructions)
        + _subcard("Played matches", _render_played(data), f'{_fi(summary.get("matches_played"))}/72')
        + _subcard("Points earned / scoring summary", _render_scoring(data))
        + _subcard("Remaining matches", _render_remaining(data), f'{_fi(summary.get("matches_remaining"))} left')
        + audit
    )


def _render_pva(data: dict) -> str:
    rows = _arr(data.get("prediction_vs_actual"))
    if not rows:
        return '<p class="muted">No prediction-vs-actual data yet.</p>'
    html = (
        '<table class="aligned">'
        '<colgroup><col class="numcol"><col><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"></colgroup>'
        '<tr><th class="num">#</th><th class="team">Match</th><th class="num">Submitted</th>'
        '<th class="num">Actual</th><th class="num">Exact</th><th class="num">Points</th></tr>'
    )
    for row in rows:
        row = _obj(row)
        points = row.get("points_earned")
        if points is None:
            points = row.get("total_points")
        html += (
            f'<tr><td class="num">{_fi(row.get("match_number"))}</td>'
            f'<td class="team">{_esc(row.get("team_a") or "—")} v {_esc(row.get("team_b") or "—")}</td>'
            f'<td class="num">{_esc(row.get("submitted_score") or row.get("predicted_score") or "—")}</td>'
            f'<td class="num">{_esc(row.get("actual_score") or "—")}</td>'
            f'<td class="num">{"yes" if row.get("exact_score_correct") else "no"}</td>'
            f'<td class="num">{_fnum(points)}</td></tr>'
        )
    return html + "</table>"


def _render_live_group_tables(data: dict) -> str:
    groups = _obj(data.get("groups"))
    intro = (
        '<p class="muted">Live group table built from actual played results. Your submitted '
        "group standings above stay locked.</p>"
    )
    keys = sorted(groups)
    body = ""
    for group in keys:
        rows = _arr(groups.get(group))
        if not rows:
            continue
        body += (
            f'<div class="grp">Group {_esc(group)}</div><table class="aligned">'
            '<colgroup><col><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"><col class="numcol"></colgroup>'
            '<tr><th class="team">Team</th><th class="num">P</th><th class="num">W</th><th class="num">D</th>'
            '<th class="num">L</th><th class="num">GF</th><th class="num">GA</th><th class="num">GD</th><th class="num">Pts</th></tr>'
        )
        for row in rows:
            row = _obj(row)
            gd = row.get("goal_difference")
            try:
                gd_txt = ("+" if float(gd) >= 0 else "") + _fi(gd)
            except (TypeError, ValueError):
                gd_txt = "—"
            body += (
                f'<tr><td class="team"><span class="rank">{_fi(row.get("rank"))}</span>{_esc(row.get("team") or "—")}</td>'
                f'<td class="num">{_fi(row.get("played"))}</td><td class="num">{_fi(row.get("won"))}</td>'
                f'<td class="num">{_fi(row.get("drawn"))}</td><td class="num">{_fi(row.get("lost"))}</td>'
                f'<td class="num">{_fi(row.get("goals_for"))}</td><td class="num">{_fi(row.get("goals_against"))}</td>'
                f'<td class="num">{gd_txt}</td><td class="num">{_fi(row.get("points"))}</td></tr>'
            )
        body += "</table>"
    if not body:
        body = '<p class="muted">No live group table data yet.</p>'
    return intro + body


def _render_incentives(data: dict) -> str:
    diagnostics = _obj(data.get("incentive_diagnostics"))
    matches = _arr(diagnostics.get("matches"))
    teams = _arr(diagnostics.get("teams"))
    intro = (
        '<p class="muted">Final group-match incentive diagnostics are live context only. '
        "They label possible rotation, must-win, draw-likely-enough and opponent-qualified "
        "situations after actual prior group results are known. Submitted predictions stay locked.</p>"
    )
    if not matches and not teams:
        return intro + '<p class="muted">No final group-match incentive notes yet. These usually appear after matchday 2.</p>'
    html = intro
    if matches:
        html += (
            '<table class="aligned">'
            '<colgroup><col class="numcol"><col class="numcol"><col><col><col></colgroup>'
            '<tr><th class="num">#</th><th class="num">G</th><th class="team">Match</th><th>Team A notes</th><th>Team B notes</th></tr>'
        )
        for row in matches:
            row = _obj(row)
            a_notes = "; ".join(_arr(row.get("team_a_notes"))) or "—"
            b_notes = "; ".join(_arr(row.get("team_b_notes"))) or "—"
            html += (
                f'<tr><td class="num">{_fi(row.get("match_number"))}</td><td class="num">{_esc(row.get("group") or "—")}</td>'
                f'<td class="team">{_esc(row.get("team_a") or "—")} v {_esc(row.get("team_b") or "—")}</td>'
                f'<td>{_esc(a_notes)}</td><td>{_esc(b_notes)}</td></tr>'
            )
        html += "</table>"
    if teams:
        html += '<details class="audit-details"><summary>Team-level incentive notes</summary>'
        for row in teams:
            row = _obj(row)
            notes = "; ".join(_arr(row.get("notes"))) or "—"
            html += (
                f'<div class="match"><span>#{_fi(row.get("match_number"))} {_esc(row.get("team") or "—")}'
                f' vs {_esc(row.get("opponent") or "—")}</span><span class="muted">{_esc(notes)}</span></div>'
            )
        html += "</details>"
    return html


def _render_advancement(data: dict) -> str:
    rows = _arr(data.get("advancement"))
    intro = (
        '<p class="muted">Advancement combines actual played results with frozen simulations of '
        "unplayed matches.</p>"
    )
    body = ""
    if rows:
        by_group: dict[str, list] = {}
        for row in rows:
            by_group.setdefault(_obj(row).get("group") or "?", []).append(_obj(row))
        for group in sorted(by_group):
            body += f'<div class="grp">Group {_esc(group)}</div>'
            for row in by_group[group]:
                cls = "adv" if float(row.get("p_advance") or 0) >= 0.5 else ("out" if float(row.get("p_advance") or 0) <= 0.05 else "")
                body += (
                    '<div style="margin:.35rem 0"><div style="display:flex;justify-content:space-between;gap:.5rem">'
                    f'<span class="{cls}">{_esc(row.get("team") or "—")}</span>'
                    f'<span class="muted">adv {_fpct(row.get("p_advance"))} · win {_fpct(row.get("p_rank1"))}</span></div>'
                    f'<div class="bar"><i style="width:{_pctw(row.get("p_advance"))}%"></i></div></div>'
                )
    else:
        body = '<p class="muted">No advancement probabilities yet.</p>'
    bracket = _obj(data.get("actual_bracket_state"))
    future = (
        '<p class="muted"><b>Projected future bracket.</b> Submitted group-stage predictions stay '
        "locked; actual results pin only played matches and future projections use only unplayed "
        "fixtures.</p>"
        '<p class="muted"><b>Future recommendation:</b> recommendations are future-only and apply '
        "only to matches that have not been played.</p>"
        f'<p class="muted">Bracket state: {_esc(bracket.get("status") or "pending_group_stage")}. '
        f'{_esc(bracket.get("note") or "")}</p>'
    )
    return intro + body + future


def render_app_html(data: dict) -> str:
    """Server-render every dashboard section as static HTML (no JS required)."""
    knockout = _obj(data.get("knockout_predictions"))
    knockout_count = len(_arr(knockout.get("matches")))
    return "".join(
        [
            _section("overview", "Overview", "live", _render_overview(data)),
            _section("submitted-scores", "Scores to fill in", f"{_fi(len(_arr(data.get('submission_score_predictions'))))} matches", _render_submission_scores(data)),
            _section("group-standings", "Group standings to fill in", "submitted", _render_submission_standings(data)),
            _section("last8", "Last-8 to fill in", "submitted", _render_submission_last8(data)),
            _section("knockout-predictions", "Knockout predictions", f"{_fi(knockout_count)} matches", _render_knockout(data)),
            _section("live-results", "Live results", None, _render_live_results(data)),
            _section("prediction-vs-actual", "Prediction vs Actual", f"{_fi(len(_arr(data.get('prediction_vs_actual'))))} scored", _render_pva(data)),
            _section("live-group-tables", "Live group tables", None, _render_live_group_tables(data)),
            _section("incentives", "Final group incentives", None, _render_incentives(data)),
            _section("advancement", "Advancement / bracket projections", None, _render_advancement(data)),
        ]
    )


def render_meta_text(data: dict) -> str:
    summary = _obj(data.get("summary"))
    n_sims = summary.get("n_sims")
    try:
        sims = f"{int(n_sims):,} sims" if n_sims is not None and math.isfinite(float(n_sims)) else "no sims"
    except (TypeError, ValueError):
        sims = "no sims"
    return f"Updated {_esc(data.get('generated_at') or '—')} · {_fi(summary.get('matches_played'))}/72 played · {_esc(sims)}"


def render_cand_text(data: dict) -> str:
    candidate = _obj(data.get("active_candidate"))
    return f"Active candidate: {_esc(candidate.get('name') or '—')} ({_esc(candidate.get('active_candidate_dir') or '—')})"


def render_html(payload: dict) -> str:
    payload_json = _payload_json(payload).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("<!--PAYLOAD_JSON_PLACEHOLDER-->", payload_json)
    # Server-side pre-render so the page works with JavaScript disabled or broken.
    html = html.replace("<!--APP_PLACEHOLDER-->", render_app_html(payload))
    html = html.replace("<!--META_PLACEHOLDER-->", render_meta_text(payload))
    html = html.replace("<!--CAND_PLACEHOLDER-->", render_cand_text(payload))
    return html


def main() -> None:
    with guard_frozen_submission("build_mobile_dashboard.py"):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        payload = build_payload()
        payload_json = _payload_json(payload, indent=2)
        (LIVE_DIR / "mobile_dashboard_data.json").write_text(payload_json, encoding="utf-8")
        html = render_html(payload)
        (LIVE_DIR / "mobile_dashboard.html").write_text(html, encoding="utf-8")
        shutil.copyfile(LIVE_DIR / "mobile_dashboard.html", DOCS_DIR / "index.html")
        shutil.copyfile(LIVE_DIR / "mobile_dashboard_data.json", DOCS_DIR / "mobile_dashboard_data.json")
        print(
            f"Built mobile dashboard for {payload['active_candidate']['name']}: "
            f"{payload['summary']['matches_played']}/72 played, "
            f"{payload['scoring_summary']['total_points']:g} points. Wrote HTML + JSON."
        )


if __name__ == "__main__":
    main()
