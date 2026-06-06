"""Validate the late-residual reference data files.

Validation rules:
1. Required columns exist for all four CSVs.
2. event_date and source_date parse as dates where present.
3. role_importance_0_5 is between 0 and 5.
4. already_reflected_score_0_5 is between 0 and 5.
5. penalties are <= 0 and >= -1.25.
6. source_url/source_name required for non-placeholder rows (
   placeholder = event_date is empty or source_reliability = 'uncertain').
7. Messi must not be encoded as key_absence or withdrawn unless status changes.
8. Karl penalty must not exceed -0.25 unless role_tier is upgraded and explicitly
   documented (notes must mention the upgrade).
9. No row may set applies_to_score_candidate=true unless an objective promoted rule
   exists (v3 manifest promotion_gate_passed = True).
10. No prediction files are modified.

Produces:
  outputs/reports/late_residual_input_validation_report.md
  outputs/predictions/late_residual_input_validation_issues.csv
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REF = ROOT / "data" / "reference"
LATE_AVAIL = REF / "late_availability_events.csv"
EXPECTED_XI = REF / "expected_xi_status.csv"
GK_STATUS = REF / "goalkeeper_status.csv"
FRIENDLY_DISTORTION = REF / "friendly_lineup_distortion_events.csv"

V2_MANIFEST = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"
V3_MANIFEST = ROOT / "outputs" / "final_candidate_v3_objective_residual" / "FROZEN_MANIFEST.json"

REPORT_PATH = ROOT / "outputs" / "reports" / "late_residual_input_validation_report.md"
ISSUES_PATH = ROOT / "outputs" / "predictions" / "late_residual_input_validation_issues.csv"

LATE_AVAIL_REQUIRED = [
    "event_id", "team", "player", "player_position", "event_type", "status",
    "event_date", "source_date", "source_url", "source_name", "source_reliability",
    "timing_category", "role_tier", "role_importance_0_5",
    "already_reflected_score_0_5", "raw_penalty", "recommended_residual_penalty",
    "applies_to_score_candidate", "reason", "notes", "needs_review",
]

EXPECTED_XI_REQUIRED = [
    "team", "player", "position", "expected_xi_status", "role_tier",
    "role_importance_0_5", "evidence_source", "source_date", "source_url",
    "notes", "needs_review",
]

GK_REQUIRED = [
    "team", "goalkeeper", "expected_rank", "availability_status",
    "caps_or_experience_note", "club_2026", "source_date", "source_url",
    "crisis_flag", "crisis_reason", "needs_review",
]

FRIENDLY_REQUIRED = [
    "match_date", "team", "opponent", "friendly_match_id_if_known",
    "distortion_type", "affected_players", "estimated_strength_distortion",
    "should_downweight_for_form", "source_date", "source_url", "notes", "needs_review",
]

VALID_EVENT_TYPES = {
    "injury", "illness", "suspension", "ruled_out", "withdrawn", "doubtful",
    "minutes_limited", "returned_to_training", "fit_for_opener", "other",
}
VALID_STATUSES = {
    "available", "active_monitoring", "doubtful", "minutes_limited",
    "ruled_out", "withdrawn", "unknown",
}
VALID_SOURCE_RELIABILITY = {"official", "tier1_media", "tier2_media", "uncertain"}
VALID_TIMING = {"long_known", "medium", "late", "matchweek", "unknown"}
VALID_ROLE_TIER = {
    "star_carry", "starter_core", "starter_likely", "rotation", "depth", "unknown",
}

MESSI_FORBIDDEN_EVENT_TYPES = {"withdrawal", "withdrawn", "ruled_out"}
MESSI_FORBIDDEN_STATUSES = {"withdrawn", "ruled_out"}


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        import math
        if math.isnan(v):
            return False
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes"}


def _parse_date(s) -> bool:
    if pd.isna(s) or str(s).strip() == "":
        return True
    try:
        datetime.strptime(str(s).strip(), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _v3_promotion_passed() -> bool:
    if not V3_MANIFEST.exists():
        return False
    try:
        return bool(json.loads(V3_MANIFEST.read_text()).get("promotion_gate_passed"))
    except Exception:
        return False


def validate() -> tuple[list[dict], list[dict]]:
    issues: list[dict] = []
    summary: list[dict] = []

    def issue(file: str, row_id, field: str, rule: int, msg: str) -> None:
        issues.append({
            "file": file,
            "row_id": str(row_id),
            "field": field,
            "rule": rule,
            "message": msg,
        })

    # ── Rule 1: required columns ──────────────────────────────────────────────
    specs = [
        (LATE_AVAIL, LATE_AVAIL_REQUIRED, "late_availability_events.csv"),
        (EXPECTED_XI, EXPECTED_XI_REQUIRED, "expected_xi_status.csv"),
        (GK_STATUS, GK_REQUIRED, "goalkeeper_status.csv"),
        (FRIENDLY_DISTORTION, FRIENDLY_REQUIRED, "friendly_lineup_distortion_events.csv"),
    ]
    frames: dict[str, pd.DataFrame] = {}
    for path, required, label in specs:
        if not path.exists():
            issue(label, "—", "—", 1, f"file not found: {path}")
            frames[label] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        frames[label] = df
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            issue(label, "—", ",".join(missing_cols), 1,
                  f"missing required columns: {missing_cols}")
        summary.append({"file": label, "rows": len(df), "columns": len(df.columns)})

    la = frames.get("late_availability_events.csv", pd.DataFrame())

    if la.empty:
        return issues, summary

    for idx, row in la.iterrows():
        row_id = row.get("event_id") or idx

        # ── Rule 2: dates parse ───────────────────────────────────────────────
        for dcol in ("event_date", "source_date"):
            if dcol in row.index and not _parse_date(row[dcol]):
                issue("late_availability_events.csv", row_id, dcol, 2,
                      f"{dcol} '{row[dcol]}' is not a valid YYYY-MM-DD date")

        # ── Rule 3: role_importance_0_5 in [0,5] ─────────────────────────────
        if "role_importance_0_5" in row.index:
            try:
                v = float(row["role_importance_0_5"])
                if not (0 <= v <= 5):
                    issue("late_availability_events.csv", row_id,
                          "role_importance_0_5", 3, f"out of range [0,5]: {v}")
            except (ValueError, TypeError):
                issue("late_availability_events.csv", row_id,
                      "role_importance_0_5", 3, "not numeric")

        # ── Rule 4: already_reflected_score_0_5 in [0,5] ─────────────────────
        if "already_reflected_score_0_5" in row.index:
            try:
                v = float(row["already_reflected_score_0_5"])
                if not (0 <= v <= 5):
                    issue("late_availability_events.csv", row_id,
                          "already_reflected_score_0_5", 4, f"out of range [0,5]: {v}")
            except (ValueError, TypeError):
                issue("late_availability_events.csv", row_id,
                      "already_reflected_score_0_5", 4, "not numeric")

        # ── Rule 5: penalties in [-1.25, 0] ──────────────────────────────────
        for pcol in ("raw_penalty", "recommended_residual_penalty"):
            if pcol in row.index:
                try:
                    p = float(row[pcol])
                    if not (-1.25 <= p <= 0):
                        issue("late_availability_events.csv", row_id, pcol, 5,
                              f"penalty {p} outside allowed range [-1.25, 0]")
                except (ValueError, TypeError):
                    issue("late_availability_events.csv", row_id, pcol, 5,
                          "not numeric")

        # ── Rule 6: source required for non-placeholder ───────────────────────
        is_placeholder = (
            (pd.isna(row.get("event_date")) or str(row.get("event_date")).strip() == "")
            or str(row.get("source_reliability", "")).strip() == "uncertain"
        )
        if not is_placeholder:
            if not row.get("source_url") and not row.get("source_name"):
                issue("late_availability_events.csv", row_id,
                      "source_url/source_name", 6,
                      "non-placeholder row missing both source_url and source_name")

        # ── Rule 7: Messi not encoded as key_absence / withdrawn ─────────────
        if (
            str(row.get("team", "")).strip() == "Argentina"
            and str(row.get("player", "")).strip() == "Lionel Messi"
        ):
            evt = str(row.get("event_type", "")).strip()
            sts = str(row.get("status", "")).strip()
            if evt in MESSI_FORBIDDEN_EVENT_TYPES:
                issue("late_availability_events.csv", row_id,
                      "event_type", 7,
                      f"Messi encoded as '{evt}'; must not be key_absence/withdrawn "
                      "unless status changes. Use 'injury' + status='active_monitoring'.")
            if sts in MESSI_FORBIDDEN_STATUSES:
                issue("late_availability_events.csv", row_id,
                      "status", 7,
                      f"Messi status '{sts}' implies key absence; expected "
                      "'active_monitoring' for in_squad_fitness_risk.")

        # ── Rule 8: Karl penalty cap at -0.25 unless upgraded ────────────────
        if (
            str(row.get("team", "")).strip() == "Germany"
            and str(row.get("player", "")).strip() == "Lennart Karl"
        ):
            try:
                resid = float(row.get("recommended_residual_penalty", 0))
                if resid < -0.25:
                    notes_text = str(row.get("notes", "")).lower()
                    if "upgraded" not in notes_text and "upgrade" not in notes_text:
                        issue("late_availability_events.csv", row_id,
                              "recommended_residual_penalty", 8,
                              f"Karl residual {resid} < -0.25 without documented "
                              "role_tier upgrade in notes.")
            except (ValueError, TypeError):
                pass

        # ── Rule 9: applies_to_score_candidate=true only when rule promoted ──
        if _parse_bool(row.get("applies_to_score_candidate")):
            if not _v3_promotion_passed():
                issue("late_availability_events.csv", row_id,
                      "applies_to_score_candidate", 9,
                      "applies_to_score_candidate=true but no promoted objective "
                      "rule found (v3 promotion_gate_passed is False or absent).")

    # ── Rule 10: no prediction files modified ────────────────────────────────
    import hashlib
    for manifest_path in (V2_MANIFEST, V3_MANIFEST):
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get("files", []):
            path = ROOT / entry["path"]
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry["sha256"]:
                    issue(entry["path"], "—", "sha256", 10,
                          f"prediction file modified: {entry['path']}")

    # ── Friendly distortion date check (rule 2 extended) ─────────────────────
    fd = frames.get("friendly_lineup_distortion_events.csv", pd.DataFrame())
    if not fd.empty:
        for idx, row in fd.iterrows():
            if "match_date" in row.index and not _parse_date(row.get("match_date")):
                issue("friendly_lineup_distortion_events.csv", idx,
                      "match_date", 2,
                      f"match_date '{row['match_date']}' is not YYYY-MM-DD")

    return issues, summary


def write_report(issues: list[dict], summary: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ISSUES_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_errors = len([i for i in issues if i["rule"] != 6])
    n_warn = len([i for i in issues if i["rule"] == 6])
    status = "PASS" if n_errors == 0 else "FAIL"

    lines = [
        "# Late Residual Input Validation Report",
        "",
        f"**Status: {status}**  ({n_errors} errors, {n_warn} source-completeness warnings)",
        "",
        "## File summary",
        "",
        "| file | rows | columns |",
        "| --- | --- | --- |",
    ]
    for s in summary:
        lines.append(f"| {s['file']} | {s['rows']} | {s['columns']} |")

    lines += ["", "## Validation rules checked", ""]
    rule_desc = {
        1: "required columns present",
        2: "date fields parse as YYYY-MM-DD",
        3: "role_importance_0_5 in [0, 5]",
        4: "already_reflected_score_0_5 in [0, 5]",
        5: "penalties in [-1.25, 0]",
        6: "source_url/source_name present on non-placeholder rows",
        7: "Messi not encoded as key_absence / withdrawn",
        8: "Karl penalty <= -0.25 unless role_tier upgrade documented",
        9: "applies_to_score_candidate=true only when objective rule promoted",
        10: "no frozen prediction files modified",
    }
    for r, desc in rule_desc.items():
        errs = [i for i in issues if i["rule"] == r]
        mark = "FAIL" if errs else "pass"
        lines.append(f"- Rule {r} ({desc}): **{mark}**" + (f" — {len(errs)} issue(s)" if errs else ""))

    if issues:
        lines += ["", "## Issues", ""]
        lines.append("| rule | file | row | field | message |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i in issues:
            lines.append(f"| {i['rule']} | {i['file']} | {i['row_id']} | {i['field']} | {i['message']} |")
    else:
        lines += ["", "_No issues found._"]

    REPORT_PATH.write_text("\n".join(lines) + "\n")

    pd.DataFrame(issues if issues else [{"rule": "", "file": "", "row_id": "",
                                          "field": "", "message": "no issues"}]).to_csv(
        ISSUES_PATH, index=False
    )


def main() -> list[dict]:
    issues, summary = validate()
    write_report(issues, summary)
    n_err = len([i for i in issues if i["rule"] != 6])
    n_warn = len([i for i in issues if i["rule"] == 6])
    status = "PASS" if n_err == 0 else "FAIL"
    print(f"Late residual input validation: {status} ({n_err} errors, {n_warn} warnings)")
    return issues


if __name__ == "__main__":
    main()
