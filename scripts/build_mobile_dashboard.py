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
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate
from src.live.scores_override import utc_now_iso

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


def _submission_score_rows(scores: pd.DataFrame, date_map: dict[int, str]) -> tuple[list[dict], str]:
    rows: list[dict] = []
    copy_lines: list[str] = []
    for row in scores.where(pd.notna(scores), None).to_dict(orient="records"):
        team_a_goals, team_b_goals = _score_parts(row.get("final_recommended_score"))
        match_number = int(row.get("match_number")) if row.get("match_number") is not None else None
        group = row.get("group") or ""
        team_a = row.get("team_a") or ""
        team_b = row.get("team_b") or ""
        final_score = row.get("final_recommended_score") or ""
        date = date_map.get(match_number) if match_number is not None else None
        rows.append(
            {
                "match_number": match_number,
                "group": group,
                "date": date,
                "team_a": team_a,
                "team_b": team_b,
                "final_recommended_score": final_score,
                "predicted_team_a_goals": team_a_goals,
                "predicted_team_b_goals": team_b_goals,
                "safe_score": row.get("safe_score"),
                "ev_score": row.get("ev_score"),
                "auto_policy_decision": row.get("auto_policy_decision"),
                "reason": row.get("reason"),
                "manual_review_flag_original": bool(row.get("manual_review_flag_original")),
                "manual_review_resolved_auto": bool(row.get("manual_review_resolved_auto")),
            }
        )
        copy_lines.append(f"{match_number}. {team_a} {final_score} {team_b}")
    return rows, "\n".join(copy_lines)


def _compare_strings(v1: list[dict], v2: list[dict], keys: list[str]) -> int:
    changes = 0
    for left, right in zip(v1, v2):
        if any(left.get(key) != right.get(key) for key in keys):
            changes += 1
    return changes


def build_payload() -> dict:
    candidate_obj = load_active_candidate()
    candidate = candidate_obj.as_dict()

    live_tables = _read_json(LIVE_DIR / "live_group_tables.json")
    sim = _read_json(LIVE_DIR / "live_group_stage_simulation_summary.json")
    pva = _read_json(LIVE_DIR / "prediction_vs_actual.json")
    scoring = _read_json(LIVE_DIR / "scoring_summary.json")

    played = _read_csv_records(LIVE_DIR / "played_matches.csv")
    remaining = _read_csv_records(LIVE_DIR / "remaining_matches.csv")
    date_map = _match_date_map(played, remaining)

    scores = candidate_obj.load_score_predictions()
    standings_df = candidate_obj.load_standing_predictions()
    last8_df = candidate_obj.load_last8_predictions()
    standings = standings_df.to_dict(orient="records")
    last8 = last8_df.to_dict(orient="records")
    submission_scores, submission_copy_text = _submission_score_rows(scores, date_map)
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
        "played_matches": played,
        "remaining_matches": remaining[:24],
        "remaining_matches_total": len(remaining),
        "groups": live_tables.get("groups", {}),
        "advancement": sim.get("teams", []),
        "prediction_vs_actual": pva.get("matches", []),
        "scoring_summary": scoring_summary,
        "submission_summary": submission_summary,
        "submission_score_predictions": submission_scores,
        "submission_score_copy_text": submission_copy_text,
        "submission_group_standings": standings,
        "submission_last8_picks": last8,
        "final_group_standings": standings,
        "last8_picks": last8,
        "manual_review": review,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC2026 Travel Mode</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0 0 3rem; background: #0f1116; color: #e7e9ee; line-height: 1.4; font-size: 16px; }
  header { position: sticky; top: 0; background: #161a23; padding: .8rem 1rem; border-bottom: 1px solid #2a2f3a; z-index: 5; }
  header h1 { margin: 0; font-size: 1.15rem; }
  header .meta { font-size: .78rem; color: #9aa3b2; margin-top: .15rem; }
  header .cand { font-size: .78rem; color: #7dd3fc; margin-top: .15rem; }
  nav.tabs { position: sticky; top: 3.9rem; z-index: 4; display:flex; gap:.4rem; overflow-x:auto; padding:.55rem .8rem; background:#11151d; border-bottom:1px solid #2a2f3a; }
  nav.tabs a { white-space:nowrap; text-decoration:none; padding:.35rem .65rem; border-radius:999px; background:#1b212c; color:#cfd6e4; font-size:.78rem; border:1px solid #2a2f3a; }
  nav.tabs a:hover { background:#222836; }
  main { padding: 0 .8rem; max-width: 720px; margin: 0 auto; }
  section { margin-top: 1.1rem; background: #161a23; border: 1px solid #2a2f3a; border-radius: 12px; padding: .85rem; }
  section > h2 { margin: 0 0 .55rem; font-size: 1rem; color: #cfd6e4; display: flex; align-items: center; gap: .4rem; }
  .subcard { background:#1b212c; border:1px solid #2a2f3a; border-radius:10px; padding:.75rem; margin:.6rem 0; }
  .subcard h3 { margin:.1rem 0 .45rem; font-size:.92rem; color:#dbe3f0; display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
  .pill { display:inline-block; font-size:.7rem; padding:.1rem .45rem; border-radius:999px; background:#222836; color:#9aa3b2; }
  table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  th, td { text-align: right; padding: .28rem .3rem; border-bottom: 1px solid #232836; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8b93a4; font-weight: 600; }
  .grp { font-weight: 700; color: #aab3c5; margin: .6rem 0 .2rem; font-size: .9rem; }
  .adv { color: #4ade80; }
  .out { color: #f87171; }
  .bar { height: 6px; border-radius: 4px; background:#2a3142; overflow:hidden; margin-top:2px; }
  .bar > i { display:block; height:100%; background:#4ade80; }
  .match { display:flex; justify-content:space-between; padding:.3rem 0; border-bottom:1px solid #232836; font-size:.85rem; }
  .match .sc { font-weight:700; }
  .error { background:#5a1f1f; border: 1px solid #8b3f3f; color:#ff9999; padding:1rem; border-radius:8px; margin:1rem 0; }
  .error h3 { color:#ffb3b3; margin-top:0; }
  .error pre { background:#3a1515; color:#ffcccc; max-height:300px; overflow-y:auto; font-size:.7rem; }
  .warn { background:#3a2a13; border-color:#5a431f; }
  .warn h2 { color:#fbbf24; }
  .muted { color:#9aa3b2; font-size:.8rem; }
  code { background:#222836; padding:.05rem .3rem; border-radius:4px; font-size:.8rem; }
  pre { background:#222836; padding:.6rem; border-radius:8px; overflow:auto; font-size:.75rem; }
  a { color:#7dd3fc; }
  details { margin-top:.4rem; }
  summary { cursor:pointer; color:#cfd6e4; }
  .filelinks a { display:inline-block; margin:.15rem .4rem .15rem 0; font-size:.8rem; }
  ol { padding-left:1.2rem; } ol li { margin:.25rem 0; }
  .bignum { font-size:1.8rem; font-weight:800; color:#4ade80; }
  .stats { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:.4rem; }
  .stat { flex:1 1 30%; background:#1b212c; border-radius:8px; padding:.5rem; text-align:center; }
  .stat b { display:block; font-size:1.2rem; color:#e7e9ee; }
  .stat span { font-size:.72rem; color:#9aa3b2; }
  .summary-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:.55rem; }
  .summary-card { background:#1b212c; border:1px solid #2a2f3a; border-radius:10px; padding:.7rem; }
  .summary-card b { display:block; font-size:1rem; margin-bottom:.15rem; }
  .summary-card span { color:#9aa3b2; font-size:.74rem; }
  .policy-list { margin:.6rem 0 0; padding-left:1.2rem; }
  .policy-list li { margin:.25rem 0; }
  .copybar { display:flex; align-items:center; justify-content:space-between; gap:.5rem; margin:.5rem 0 .65rem; flex-wrap:wrap; }
  .copybar button { background:#7dd3fc; color:#082032; border:0; border-radius:999px; padding:.4rem .75rem; font-weight:700; cursor:pointer; }
  .copyblock { white-space:pre-wrap; word-break:break-word; background:#11151d; border:1px solid #2a2f3a; border-radius:10px; padding:.7rem; font-size:.78rem; }
  .group-summary { cursor:pointer; }
  .row-card { background:#11151d; border:1px solid #2a2f3a; border-radius:10px; padding:.7rem; margin:.55rem 0; }
  .row-top { display:flex; justify-content:space-between; gap:.6rem; align-items:flex-start; }
  .row-title { font-weight:800; font-size:.95rem; }
  .score-big { font-size:1.8rem; font-weight:900; color:#4ade80; letter-spacing:-0.02em; }
  .row-meta { display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.5rem; }
  .row-meta .pill { background:#1f2937; }
  .row-copy { margin-top:.55rem; font-size:.78rem; color:#9aa3b2; }
  .label { color:#8b93a4; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
</style>
</head>
<body>
<header>
  <h1>🏆 WC2026 Travel Mode</h1>
  <div class="meta" id="meta">Loading…</div>
  <div class="cand" id="cand"></div>
</header>
<nav class="tabs" id="tabs">
  <a href="#overview">Overview</a>
  <a href="#submit-scores">Submit Scores</a>
  <a href="#group-standings">Group Standings</a>
  <a href="#last8">Last-8</a>
  <a href="#live-results">Live Results</a>
  <a href="#prediction-vs-actual">Prediction vs Actual</a>
</nav>
<main id="app"></main>
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

  function buildSection(id, title, pill, body) {
    return '<section id="' + id + '"><h2>' + title + (pill ? '<span class="pill">' + pill + '</span>' : '') + '</h2>' + body + '</section>';
  }

  function buildSubcard(title, body, rightMeta) {
    return '<div class="subcard"><h3><span>' + title + '</span>' + (rightMeta ? '<span class="pill">' + rightMeta + '</span>' : '') + '</h3>' + body + '</div>';
  }

  function showError(title, err, data) {
    const keys = data && typeof data === 'object' && !Array.isArray(data) ? Object.keys(data) : [];
    const message = err && err.message ? err.message : String(err || 'Unknown error');
    const stack = err && err.stack ? '<pre>' + esc(err.stack) + '</pre>' : '';
    if (metaEl) metaEl.textContent = 'Dashboard error';
    if (candEl) candEl.textContent = '';
    app.innerHTML = '<section class="error"><h3>❌ ' + esc(title) + '</h3><p>' + esc(message) + '</p><p><b>Loaded keys:</b> ' + esc(keys.length ? keys.join(', ') : 'none') + '</p>' + stack + '</section>';
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
    const requiredObjects = ['active_candidate', 'summary', 'groups', 'scoring_summary'];
    const requiredArrays = ['played_matches', 'remaining_matches', 'advancement', 'prediction_vs_actual', 'final_group_standings', 'last8_picks', 'manual_review'];
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
      played_matches: safeArray(data.played_matches),
      remaining_matches: remainingMatches,
      remaining_matches_total: Number.isFinite(Number(data.remaining_matches_total)) ? Number(data.remaining_matches_total) : remainingMatches.length,
      groups: safeObject(data.groups),
      advancement: safeArray(data.advancement),
      prediction_vs_actual: safeArray(data.prediction_vs_actual),
      scoring_summary: safeObject(data.scoring_summary),
      final_group_standings: safeArray(data.final_group_standings),
      last8_picks: safeArray(data.last8_picks),
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
    let table = '<table><tr><th>#</th><th>Match</th><th>Pred</th><th>Actual</th><th>O</th><th>GD</th><th>Exact</th><th>Pts</th></tr>';
    let details = '';
    for (let i = 0; i < rows.length; i += 1) {
      const m = safeObject(rows[i]);
      table += '<tr><td>' + fmtInt(m.match_number) + '</td><td style="text-align:left">' + esc(m.team_a || '—') + ' v ' + esc(m.team_b || '—') + '</td><td>' + esc(m.predicted_score || '—') + '</td><td>' + esc(m.actual_score || '—') + '</td><td>' + (m.outcome_correct ? '✅' : '❌') + '</td><td>' + (m.goal_difference_correct ? '✅' : '❌') + '</td><td>' + (m.exact_score_correct ? '✅' : '❌') + '</td><td>' + fmtNum(m.total_points) + '</td></tr>';
      details += '<details><summary>#' + fmtInt(m.match_number) + ' ' + esc(m.team_a || '—') + ' v ' + esc(m.team_b || '—') + ' - ' + fmtNum(m.total_points) + ' pts</summary><p class="muted">' + esc(m.scoring_explanation || 'No explanation available.') + '</p></details>';
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
      html += '<div class="grp">Group ' + esc(groupName) + '</div><table><tr><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GD</th><th>Pts</th></tr>';
      for (let j = 0; j < rows.length; j += 1) {
        const row = safeObject(rows[j]);
        const gd = Number(row.goal_difference);
        const gdText = Number.isFinite(gd) ? (gd >= 0 ? '+' : '') + fmtInt(gd) : '—';
        html += '<tr><td>' + fmtInt(row.rank) + '. ' + esc(row.team || '—') + '</td><td>' + fmtInt(row.played) + '</td><td>' + fmtInt(row.won) + '</td><td>' + fmtInt(row.drawn) + '</td><td>' + fmtInt(row.lost) + '</td><td>' + gdText + '</td><td>' + fmtInt(row.points) + '</td></tr>';
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
    const activeCandidate = safeObject(data.active_candidate);
    const cards = [
      ['Active candidate', activeCandidate.name || '—', activeCandidate.active_candidate_dir || '—'],
      ['Total matches', fmtInt(summary.total_matches), 'all 72 picks are present'],
      ['Auto-resolved', fmtInt(summary.manual_review_rows_auto_resolved), 'science-only policy'],
      ['EV accepted', fmtInt(summary.ev_overrides_accepted), 'policy kept EV out'],
      ['EV rejected', fmtInt(summary.ev_overrides_rejected), 'safe score kept'],
      ['Safe scores kept', fmtInt(summary.safe_scores_kept), 'recommended submission stayed with safe score'],
      ['Scores changed vs v1', fmtInt(summary.score_changes_vs_v1), 'should remain zero'],
      ['Group standings changed vs v1', fmtInt(summary.group_standings_changed_vs_v1), 'two groups changed'],
      ['Last-8 changed vs v1', fmtInt(summary.last8_changed_vs_v1), 'unchanged'],
    ];
    let html = '<div class="summary-grid">';
    for (let i = 0; i < cards.length; i += 1) {
      const card = cards[i];
      html += '<div class="summary-card"><b>' + esc(card[0]) + '</b><span>' + esc(card[1]) + '</span><div class="muted" style="margin-top:.35rem">' + esc(card[2]) + '</div></div>';
    }
    html += '</div>';
    html += '<p class="muted" style="margin-top:.75rem">Science-only submission: auto-resolved by policy, with safe scores kept where EV was rejected.</p>';
    html += '<ul class="policy-list">';
    html += '<li>Original manual-review flags are preserved for audit only.</li>';
    html += '<li>The submission sheet is fully auto-resolved by science-only policy.</li>';
    html += '<li>Use the Submit Scores section below to copy the 72 recommended results.</li>';
    html += '</ul>';
    return html;
  }

  function renderSubmissionScores(rows, copyText) {
    if (!rows.length) {
      return '<p class="muted">No score predictions available.</p>';
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
    let html = '<p class="muted"><b>Scores to fill in.</b> Fill in these 72 picks. The score line shown in green is the recommended submission; the safe score is kept when EV is rejected by policy.</p>';
    html += '<div class="copybar"><button type="button" onclick="navigator.clipboard.writeText(document.getElementById(&quot;scores-copy&quot;).innerText)">Copy all scores</button><span class="muted">' + fmtInt(total) + ' lines</span></div>';
    html += '<pre class="copyblock" id="scores-copy">' + esc(copyText) + '</pre>';
    for (let i = 0; i < groupKeys.length; i += 1) {
      const groupName = groupKeys[i];
      const groupRows = byGroup[groupName];
      let groupHtml = '<div class="copybar"><span class="muted">Copy-friendly line format: <code>1. Mexico 1-0 South Africa</code></span></div>';
      for (let j = 0; j < groupRows.length; j += 1) {
        const row = safeObject(groupRows[j]);
        const score = row.final_recommended_score || '—';
        const policy = String(row.auto_policy_decision || '');
        let policyLabel = 'science-only policy';
        if (policy === 'ev_override_accepted') {
          policyLabel = 'EV accepted by policy';
        } else if (String(row.safe_score || '') === String(row.ev_score || '')) {
          policyLabel = 'all scientific sources agree';
        } else {
          policyLabel = 'EV rejected by policy';
        }
        const originalManual = row.manual_review_flag_original ? 'yes' : 'no';
        groupHtml += '<article class="row-card"><div class="row-top"><div><div class="label">Match ' + fmtInt(row.match_number) + (row.date ? ' · ' + esc(row.date) : '') + '</div><div class="row-title">' + esc(row.team_a || '—') + ' vs ' + esc(row.team_b || '—') + '</div></div><div class="score-big">' + esc(score) + '</div></div><div class="row-meta"><span class="pill">predicted A: ' + fmtInt(row.predicted_team_a_goals) + '</span><span class="pill">predicted B: ' + fmtInt(row.predicted_team_b_goals) + '</span><span class="pill">safe: ' + esc(row.safe_score || '—') + '</span><span class="pill">EV: ' + esc(row.ev_score || '—') + '</span><span class="pill">policy: ' + esc(row.auto_policy_decision || '—') + '</span><span class="pill">' + esc(policyLabel) + '</span><span class="pill">original manual flag: ' + esc(originalManual) + '</span></div><div class="row-copy">' + esc(fmtInt(row.match_number) + '. ' + row.team_a + ' ' + score + ' ' + row.team_b) + '<br><span class="muted">reason: ' + esc(row.reason || '—') + '</span></div></article>';
      }
      html += '<details class="group-summary"><summary class="group-summary">Group ' + esc(groupName) + ' (' + fmtInt(groupRows.length) + ' matches)</summary>' + groupHtml + '</details>';
    }
    return html;
  }

  function renderSubmissionStandings(rows) {
    if (!rows.length) {
      return '<p class="muted">No group standings predictions available.</p>';
    }
    let html = '<p class="muted"><b>Group standings to fill in.</b> Fill in these group standings exactly as shown.</p>';
    html += '<table><tr><th>G</th><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr>';
    for (let i = 0; i < rows.length; i += 1) {
      const row = safeObject(rows[i]);
      html += '<tr><td>' + esc(row.group || '—') + '</td><td>' + esc(row.rank_1 || '—') + '</td><td>' + esc(row.rank_2 || '—') + '</td><td>' + esc(row.rank_3 || '—') + '</td><td>' + esc(row.rank_4 || '—') + '</td></tr>';
    }
    return html + '</table>';
  }

  function renderSubmissionLast8(rows) {
    if (!rows.length) {
      return '<p class="muted">No Last-8 picks available.</p>';
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
    let html = '<p class="muted"><b>Last-8 / progression picks to fill in.</b> These are unchanged from the prior candidate.</p>';
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

  let rawData = null;
  try {
    const rawInline = parseInlinePayload();
    if (rawInline === null) {
      fetchPayload()
        .then(function(raw) {
          renderDashboard(validatePayload(raw));
        })
        .catch(function(err) {
          showError('Could not load dashboard', err, null);
        });
      return;
    }

    rawData = rawInline;
    const data = validatePayload(rawData);
    renderDashboard(data);
  } catch (err) {
    showError('Initialization failed', err, rawData);
    return;
  }

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
      const advancementHtml = renderAdvancement(advancement);
      const remainingHtml = renderRemainingMatches(remainingMatches, data.remaining_matches_total);
      const auditHtml = renderValidationNotes({ tie_break_note: data.tie_break_note, sim_tie_break_note: data.sim_tie_break_note, manual_review: manualReview });
      const liveHtml = [
        buildSubcard('Score input instructions', instructionsHtml),
        buildSubcard('Played matches', playedHtml, fmtInt(summary.matches_played) + '/72'),
        buildSubcard('Points earned / scoring summary', scoringHtml),
        buildSubcard('Live group tables', groupsHtml),
        buildSubcard('Advancement probabilities', advancementHtml, Number.isFinite(Number(summary.n_sims)) ? Number(summary.n_sims).toLocaleString() + ' sims' : null),
        buildSubcard('Remaining matches', remainingHtml, fmtInt(summary.matches_remaining) + ' left'),
        buildSubcard('Science-only policy notes', auditHtml)
      ].join('');

      const filesHtml = '<div class="filelinks"><a href="prediction_vs_actual.csv">pva.csv</a> <a href="scoring_summary.csv">score.csv</a> <a href="live_group_tables.csv">tables.csv</a> <a href="mobile_dashboard_data.json">data.json</a></div>';

      app.innerHTML = [
        buildSection('overview', 'Overview', 'live', overviewHtml),
        buildSection('submit-scores', 'Submit Scores', fmtInt(submissionScores.length) + ' matches', submitHtml),
        buildSection('group-standings', 'Group Standings', 'to fill in', standingsHtml),
        buildSection('last8', 'Last-8', 'to fill in', last8Html),
        buildSection('live-results', 'Live Results', null, liveHtml + filesHtml),
        buildSection('prediction-vs-actual', 'Prediction vs Actual', fmtInt(predictionVsActual.length) + ' scored', pvaHtml)
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


def render_html(payload: dict) -> str:
    payload_json = _payload_json(payload).replace("</", "<\\/")
    # Replace the payload placeholder with actual JSON.
    return HTML_TEMPLATE.replace("<!--PAYLOAD_JSON_PLACEHOLDER-->", payload_json)


def main() -> None:
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
