#!/usr/bin/env python3
"""Build the static mobile Travel Mode dashboard (Travel Mode, Task H).

Consolidates the live outputs (group tables, simulations, prediction-vs-actual
scoring) and the *active candidate's* frozen picks into a single JSON payload,
then renders a self-contained, mobile-friendly HTML file. The data is embedded
directly in the HTML so the page works straight off the filesystem or GitHub
Pages -- no server, no build step, no external fetch.

Inputs (all already produced by the other Travel Mode scripts):
    data/live/active_candidate.yml -> active candidate prediction files
    outputs/live/live_group_tables.json
    outputs/live/live_group_stage_simulation_summary.json
    outputs/live/prediction_vs_actual.json
    outputs/live/scoring_summary.json
    outputs/live/played_matches.csv
    outputs/live/remaining_matches.csv

Outputs:
    outputs/live/mobile_dashboard_data.json
    outputs/live/mobile_dashboard.html

Does NOT retrain, fetch APIs, or change any prediction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.live.active_candidate import load_active_candidate
from src.live.scores_override import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "outputs" / "live"

# GitHub repo for the "update from phone" deep link. Override via the env var
# TRAVEL_MODE_REPO ("owner/name") in the workflow if the repo is renamed.
REPO_SLUG = os.environ.get("TRAVEL_MODE_REPO", "")


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


def build_payload() -> dict:
    candidate_obj = load_active_candidate()
    candidate = candidate_obj.as_dict()

    live_tables = _read_json(LIVE_DIR / "live_group_tables.json")
    sim = _read_json(LIVE_DIR / "live_group_stage_simulation_summary.json")
    pva = _read_json(LIVE_DIR / "prediction_vs_actual.json")
    scoring = _read_json(LIVE_DIR / "scoring_summary.json")

    standings = candidate_obj.load_standing_predictions().to_dict(orient="records")
    last8 = candidate_obj.load_last8_predictions().to_dict(orient="records")
    scores = candidate_obj.load_score_predictions()
    review = _review_rows(scores)

    played = _read_csv_records(LIVE_DIR / "played_matches.csv")
    remaining = _read_csv_records(LIVE_DIR / "remaining_matches.csv")

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
        "final_group_standings": standings,
        "last8_picks": last8,
        "manual_review": review,
    }


# --------------------------------------------------------------------------- #
# HTML rendering (no framework, inline CSS, embedded data + small vanilla JS).
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC2026 Travel Mode</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 0 0 3rem; background: #0f1116; color: #e7e9ee;
         line-height: 1.4; font-size: 16px; }}
  header {{ position: sticky; top: 0; background: #161a23; padding: .8rem 1rem;
           border-bottom: 1px solid #2a2f3a; z-index: 5; }}
  header h1 {{ margin: 0; font-size: 1.15rem; }}
  header .meta {{ font-size: .78rem; color: #9aa3b2; margin-top: .15rem; }}
  header .cand {{ font-size: .78rem; color: #7dd3fc; margin-top: .15rem; }}
  main {{ padding: 0 .8rem; max-width: 720px; margin: 0 auto; }}
  section {{ margin-top: 1.1rem; background: #161a23; border: 1px solid #2a2f3a;
            border-radius: 12px; padding: .85rem; }}
  section > h2 {{ margin: 0 0 .55rem; font-size: 1rem; color: #cfd6e4;
                 display: flex; align-items: center; gap: .4rem; }}
  .pill {{ display:inline-block; font-size:.7rem; padding:.1rem .45rem;
          border-radius:999px; background:#222836; color:#9aa3b2; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  th, td {{ text-align: right; padding: .28rem .3rem; border-bottom: 1px solid #232836; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: #8b93a4; font-weight: 600; }}
  .grp {{ font-weight: 700; color: #aab3c5; margin: .6rem 0 .2rem; font-size: .9rem; }}
  .adv {{ color: #4ade80; }}
  .out {{ color: #f87171; }}
  .bar {{ height: 6px; border-radius: 4px; background:#2a3142; overflow:hidden; margin-top:2px;}}
  .bar > i {{ display:block; height:100%; background:#4ade80; }}
  .match {{ display:flex; justify-content:space-between; padding:.3rem 0;
           border-bottom:1px solid #232836; font-size:.85rem; }}
  .match .sc {{ font-weight:700; }}
  .warn {{ background:#3a2a13; border-color:#5a431f; }}
  .warn h2 {{ color:#fbbf24; }}
  .muted {{ color:#9aa3b2; font-size:.8rem; }}
  code {{ background:#222836; padding:.05rem .3rem; border-radius:4px; font-size:.8rem; }}
  pre {{ background:#222836; padding:.6rem; border-radius:8px; overflow:auto; font-size:.75rem; }}
  a {{ color:#7dd3fc; }}
  details {{ margin-top:.4rem; }}
  summary {{ cursor:pointer; color:#cfd6e4; }}
  .filelinks a {{ display:inline-block; margin:.15rem .4rem .15rem 0; font-size:.8rem; }}
  ol {{ padding-left:1.2rem; }} ol li {{ margin:.25rem 0; }}
  .bignum {{ font-size:1.8rem; font-weight:800; color:#4ade80; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-top:.4rem; }}
  .stat {{ flex:1 1 30%; background:#1b212c; border-radius:8px; padding:.5rem; text-align:center; }}
  .stat b {{ display:block; font-size:1.2rem; color:#e7e9ee; }}
  .stat span {{ font-size:.72rem; color:#9aa3b2; }}
</style>
</head>
<body>
<header>
  <h1>🏆 WC2026 Travel Mode</h1>
  <div class="meta" id="meta"></div>
  <div class="cand" id="cand"></div>
</header>
<main id="app"></main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const app = document.getElementById('app');
const el = (h) => {{ const d=document.createElement('div'); d.innerHTML=h; return d.firstElementChild; }};
const esc = (s) => String(s==null?'':s).replace(/[&<>]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
const pct = (x) => x==null ? '–' : (x*100).toFixed(0)+'%';
const tick = (b) => b ? '✅' : '❌';
const num = (x) => x==null ? '–' : (Math.round(x*100)/100);

document.getElementById('meta').textContent =
  `Updated ${{D.generated_at}} · ${{D.summary.matches_played}}/72 played · `
  + (D.summary.n_sims ? D.summary.n_sims.toLocaleString()+' sims' : 'no sims');
document.getElementById('cand').textContent =
  `Active candidate: ${{D.active_candidate.name}} (${{D.active_candidate.active_candidate_dir}})`;

function section(title, pill, bodyHtml) {{
  return `<section><h2>${{title}}${{pill?` <span class="pill">${{pill}}</span>`:''}}</h2>${{bodyHtml}}</section>`;
}}

// 1. Active candidate detail
const candHtml = `<table>
  <tr><td>Directory</td><td>${{esc(D.active_candidate.active_candidate_dir)}}</td></tr>
  <tr><td>Scores file</td><td>${{esc(D.active_candidate.score_predictions_file)}}</td></tr>
  <tr><td>Standings file</td><td>${{esc(D.active_candidate.standing_predictions_file)}}</td></tr>
  <tr><td>Last-8 file</td><td>${{esc(D.active_candidate.last8_predictions_file)}}</td></tr>
</table><p class="muted">All Travel Mode outputs are computed against this frozen candidate. Switch via data/live/active_candidate.yml.</p>`;

// 2. Score input instructions
const repoLink = D.repo_slug
  ? `<p><a href="https://github.com/${{D.repo_slug}}/actions/workflows/travel_mode_update.yml">▶ Open the single-match Travel Mode workflow</a></p>`
  : '<p class="muted">Set <code>TRAVEL_MODE_REPO=owner/name</code> when building to get direct workflow links here.</p>';
const howHtml = repoLink + `<p><b>One match (GitHub Actions):</b></p><ol>
  <li>GitHub <b>Actions</b> → <b>Travel Mode update</b> → <b>Run workflow</b>.</li>
  <li>Enter <code>match_number</code>, <code>team_a_goals</code>, <code>team_b_goals</code>; leave status <code>played</code>.</li>
  <li>Run it; reload this page after ~1 minute.</li>
</ol>
<p><b>Several matches (GitHub Issue comment):</b> post a comment containing:</p>
<pre>/WK-SCORES
match_number,team_a_goals,team_b_goals,status,notes
1,2,1,played,Mexico v South Africa
2,1,1,played,Korea Republic v Czechia
/END-WK-SCORES</pre>
<p class="muted">To fix a wrong score, re-enter the same match number with corrected goals. To clear a result, set status to <code>scheduled</code>.</p>`;

// 4. Matches played
let playedHtml = D.played_matches.length
  ? D.played_matches.map(m => `<div class="match"><span>#${{m.match_number}} ${{esc(m.team_a)}} v ${{esc(m.team_b)}} <span class="muted">(${{esc(m.group)}})</span></span><span class="sc">${{m.team_a_goals}}–${{m.team_b_goals}}</span></div>`).join('')
  : '<p class="muted">No matches entered yet. The dashboard shows the frozen pre-tournament picks until you enter scores.</p>';

// 5 + 6 + 7 + 8. Prediction vs actual + scoring
const S = D.scoring_summary;
let scoreHtml = `<div class="bignum">${{num(S.total_points)}} pts</div>`
  + `<p class="muted">earned across ${{S.played_matches}} played match(es) · `
  + `${{num(S.possible_points_for_played_matches)}} possible · ${{num(S.points_missed)}} missed · `
  + `${{num(S.average_points_per_played_match)}} avg/match</p>`
  + `<div class="stats">`
  + `<div class="stat"><b>${{S.outcomes_correct}}/${{S.played_matches}}</b><span>outcomes ✅</span></div>`
  + `<div class="stat"><b>${{S.goal_differences_correct}}/${{S.played_matches}}</b><span>goal diff ✅</span></div>`
  + `<div class="stat"><b>${{S.exact_scores_correct}}/${{S.played_matches}}</b><span>exact score ✅</span></div>`
  + `</div>`;
const grp = S.total_by_group || {{}};
if (Object.keys(grp).length) {{
  scoreHtml += `<div class="grp">Points by group</div><table><tr><th>Grp</th><th>MP</th><th>Pts</th><th>Poss</th><th>Missed</th></tr>`
    + Object.keys(grp).sort().map(g => `<tr><td>${{esc(g)}}</td><td>${{grp[g].played_matches}}</td><td>${{num(grp[g].total_points)}}</td><td>${{num(grp[g].possible_points)}}</td><td>${{num(grp[g].points_missed)}}</td></tr>`).join('')
    + `</table>`;
}}

let pvaHtml;
if (D.prediction_vs_actual.length) {{
  pvaHtml = `<table><tr><th>#</th><th>Match</th><th>Pred</th><th>Act</th><th>O</th><th>GD</th><th>Exact</th><th>Pts</th></tr>`
    + D.prediction_vs_actual.map(m => `<tr><td>${{m.match_number}}</td><td style="text-align:left">${{esc(m.team_a)}} v ${{esc(m.team_b)}}</td><td>${{esc(m.predicted_score)}}</td><td>${{esc(m.actual_score)}}</td><td>${{tick(m.outcome_correct)}}</td><td>${{tick(m.goal_difference_correct)}}</td><td>${{tick(m.exact_score_correct)}}</td><td>${{num(m.total_points)}}</td></tr>`).join('')
    + `</table>`
    + D.prediction_vs_actual.map(m => `<details><summary>#${{m.match_number}} ${{esc(m.team_a)}} v ${{esc(m.team_b)}} — ${{num(m.total_points)}} pts</summary><p class="muted">${{esc(m.scoring_explanation)}}</p></details>`).join('');
}} else {{
  pvaHtml = '<p class="muted">No played matches yet — enter scores to compare predicted vs actual and see points earned.</p>';
}}

// 9. Live group tables
let tablesHtml = '';
for (const [g, rows] of Object.entries(D.groups)) {{
  tablesHtml += `<div class="grp">Group ${{g}}</div><table><tr><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GD</th><th>Pts</th></tr>` +
    rows.map(r => `<tr><td>${{r.rank}}. ${{esc(r.team)}}</td><td>${{r.played}}</td><td>${{r.won}}</td><td>${{r.drawn}}</td><td>${{r.lost}}</td><td>${{r.goal_difference>=0?'+':''}}${{r.goal_difference}}</td><td>${{r.points}}</td></tr>`).join('') +
    `</table>`;
}}
tablesHtml += `<p class="muted">${{esc(D.tie_break_note)}}</p>`;

// 10. Advancement probabilities
const byGroup = {{}};
D.advancement.forEach(t => {{ (byGroup[t.group]=byGroup[t.group]||[]).push(t); }});
let advHtml = '';
for (const [g, rows] of Object.entries(byGroup)) {{
  advHtml += `<div class="grp">Group ${{g}}</div>` + rows.map(t => {{
    const cls = t.p_advance>=0.5?'adv':(t.p_advance<=0.05?'out':'');
    return `<div style="margin:.35rem 0"><div style="display:flex;justify-content:space-between"><span class="${{cls}}">${{esc(t.team)}}</span><span class="muted">adv ${{pct(t.p_advance)}} · win ${{pct(t.p_rank1)}}</span></div><div class="bar"><i style="width:${{(t.p_advance*100).toFixed(0)}}%"></i></div></div>`;
  }}).join('');
}}

// 11. Remaining matches
let remHtml = D.remaining_matches.length
  ? D.remaining_matches.map(m => `<div class="match"><span>#${{m.match_number}} ${{esc(m.team_a)}} v ${{esc(m.team_b)}} <span class="muted">(${{esc(m.group)}})</span></span><span class="muted">${{esc(m.date)}}</span></div>`).join('')
    + (D.remaining_matches_total > D.remaining_matches.length ? `<p class="muted">… ${{D.remaining_matches_total - D.remaining_matches.length}} more</p>` : '')
  : '<p class="muted">All group matches played.</p>';

// 6b. Final candidate group-standing picks
let standHtml = `<table><tr><th>Grp</th><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr>` +
  D.final_group_standings.map(s => `<tr><td>${{esc(s.group)}}</td><td>${{esc(s.rank_1)}}</td><td>${{esc(s.rank_2)}}</td><td>${{esc(s.rank_3)}}</td><td>${{esc(s.rank_4)}}</td></tr>`).join('') +
  `</table><p class="muted">Frozen picks from ${{esc(D.active_candidate.name)}} — not changed by score entry.</p>`;

// 12. Last-8 picks
const stages = {{}};
D.last8_picks.forEach(p => {{ (stages[p.stage]=stages[p.stage]||[]).push(p); }});
let last8Html = '';
for (const [st, rows] of Object.entries(stages)) {{
  last8Html += `<div class="grp">${{esc(st.replace(/_/g,' '))}}</div>` +
    rows.map(r => `<div class="match"><span>${{r.rank}}. ${{esc(r.team)}}</span><span class="muted">${{pct(r.probability)}}</span></div>`).join('');
}}

// 13. Manual review / warnings
let warnHtml = '';
if (D.manual_review.length) {{
  warnHtml += `<p>${{D.manual_review.length}} fixture(s) were flagged for manual review in the frozen pack (safe vs EV disagreement):</p>` +
    D.manual_review.map(m => `<div class="match"><span>#${{m.match_number}} ${{esc(m.team_a)}} v ${{esc(m.team_b)}}</span><span class="muted">${{esc(m.final_recommended_score)}}</span></div>`).join('');
}} else {{ warnHtml = '<p class="muted">No flagged fixtures.</p>'; }}
warnHtml += `<details><summary>Simulation tie-break note</summary><p class="muted">${{esc(D.sim_tie_break_note)}}</p></details>`;

// File links
const fileLinks = `<div class="filelinks">
  <a href="prediction_vs_actual.csv">prediction_vs_actual.csv</a>
  <a href="scoring_summary.csv">scoring_summary.csv</a>
  <a href="live_group_tables.csv">group_tables.csv</a>
  <a href="live_group_stage_simulation_summary.csv">simulation_summary.csv</a>
  <a href="played_matches.csv">played.csv</a>
  <a href="remaining_matches.csv">remaining.csv</a>
  <a href="mobile_dashboard_data.json">dashboard_data.json</a>
</div>`;

app.appendChild(el(section('🧭 Active candidate', D.active_candidate.name, candHtml)));
app.appendChild(el(section('📱 How to enter scores', null, howHtml)));
app.appendChild(el(section('📋 Matches played', `${{D.summary.matches_played}}/72`, playedHtml)));
app.appendChild(el(section('🏅 Points earned', null, scoreHtml)));
app.appendChild(el(section('🎯 Prediction vs actual', `${{D.prediction_vs_actual.length}} scored`, pvaHtml)));
app.appendChild(el(section('📊 Live group tables', null, tablesHtml)));
app.appendChild(el(section('📈 Advancement probabilities', D.summary.n_sims?`${{D.summary.n_sims.toLocaleString()}} sims`:null, advHtml)));
app.appendChild(el(section('⏭️ Remaining matches', `${{D.summary.matches_remaining}} left`, remHtml)));
app.appendChild(el(section('🥇 Final candidate group picks', 'frozen', standHtml)));
app.appendChild(el(section('🏁 Last-8 picks', 'frozen', last8Html)));
const warnSection = el(section('⚠️ Manual review &amp; warnings', null, warnHtml));
warnSection.classList.add('warn');
app.appendChild(warnSection);
app.appendChild(el(section('🔗 Raw files', null, fileLinks)));
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload).replace("</", "<\\/")
    return HTML_TEMPLATE.format(payload_json=payload_json)


def main() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (LIVE_DIR / "mobile_dashboard_data.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    html = render_html(payload)
    (LIVE_DIR / "mobile_dashboard.html").write_text(html, encoding="utf-8")
    print(
        f"Built mobile dashboard for {payload['active_candidate']['name']}: "
        f"{payload['summary']['matches_played']}/72 played, "
        f"{payload['scoring_summary']['total_points']:g} points, "
        f"{len(payload['advancement'])} team rows. Wrote HTML + JSON to outputs/live/."
    )


if __name__ == "__main__":
    main()
