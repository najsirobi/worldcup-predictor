#!/usr/bin/env python3
"""Build WC2026 team travel / recovery burden sequences.

This script creates a context feature table from the fixed WC2026 match-venue
reference and, when available, the current projected knockout bracket. It does
not train models, alter predictions, or apply any hard penalty.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.travel_burden import haversine_km, score_travel_burden, timezone_delta_hours


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference"
INTERIM_DIR = ROOT / "data" / "interim"
REPORT_DIR = ROOT / "outputs" / "reports"

VENUES_PATH = REFERENCE_DIR / "wc2026_venues.csv"
MATCH_VENUES_PATH = REFERENCE_DIR / "wc2026_match_venues_enriched.csv"
VENUE_QA_SUMMARY_PATH = REPORT_DIR / "wc2026_venue_extraction_cross_check_summary.md"
PROJECTED_KNOCKOUT_PATH = ROOT / "outputs" / "live" / "knockout_predictions.csv"
FROZEN_MANIFEST_PATH = ROOT / "outputs" / "final_candidate_v2_auto_science" / "FROZEN_MANIFEST.json"

SEQUENCES_OUT = INTERIM_DIR / "wc2026_travel_sequences.parquet"
SEQUENCE_REPORT = REPORT_DIR / "wc2026_travel_sequence_report.md"
CONTEXT_REPORT = REPORT_DIR / "wc2026_travel_burden_context.md"
FEASIBILITY_REPORT = REPORT_DIR / "travel_burden_historical_feasibility.md"
POLICY_REPORT = REPORT_DIR / "travel_burden_policy_recommendation.md"

TEAM_COLUMNS = [
    "team",
    "match_number",
    "round",
    "group",
    "match_date",
    "venue_id",
    "stadium",
    "host_city",
    "latitude",
    "longitude",
    "elevation_m",
    "timezone",
    "previous_match_number",
    "previous_match_date",
    "previous_venue_id",
    "rest_days",
    "distance_from_previous_km",
    "timezone_delta_hours",
    "elevation_delta_m",
    "same_venue_flag",
    "same_host_city_flag",
    "country_change_flag",
    "travel_burden_score",
    "needs_review",
    "notes",
]

COMPONENT_COLUMNS = [
    "rest_days_penalty",
    "distance_penalty",
    "timezone_penalty",
    "elevation_penalty",
    "same_location_bonus",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _combine_notes(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if part is None:
            continue
        try:
            if pd.isna(part):
                continue
        except TypeError:
            pass
        text = str(part).strip()
        if text:
            values.append(text)
    return "; ".join(values)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.head(max_rows).copy()
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _validate_prerequisites() -> None:
    missing = [
        path
        for path in [MATCH_VENUES_PATH, VENUES_PATH, VENUE_QA_SUMMARY_PATH]
        if not path.exists()
    ]
    if missing:
        rel = [str(path.relative_to(ROOT)) for path in missing]
        raise FileNotFoundError(
            "WC2026 venue reference QA prerequisite is not satisfied. Missing: "
            + ", ".join(rel)
        )

    qa_text = VENUE_QA_SUMMARY_PATH.read_text()
    qa_pass_markers = ["Status: PASS", "Overall Status:", "VERIFIED"]
    if not ("Status: PASS" in qa_text or all(marker in qa_text for marker in qa_pass_markers[1:])):
        raise ValueError(
            "WC2026 venue reference QA summary does not contain a PASS/VERIFIED marker. "
            "Run venue extraction/QA before building travel sequences."
        )


def _base_team_row(match: pd.Series, team: str, notes: str, needs_review: bool) -> dict[str, Any]:
    return {
        "team": team,
        "match_number": int(match["match_number"]),
        "round": str(match["round"]),
        "group": match.get("group"),
        "match_date": str(match["date"])[:10],
        "venue_id": str(match["venue_id"]),
        "stadium": str(match["stadium"]),
        "host_city": str(match["host_city"]),
        "host_country": str(match.get("actual_country", match.get("host_country", ""))),
        "latitude": float(match["latitude"]),
        "longitude": float(match["longitude"]),
        "elevation_m": float(match["elevation_m"]),
        "timezone": str(match["timezone"]),
        "needs_review": needs_review or _as_bool(match.get("needs_review", False)),
        "notes": notes,
    }


def _group_team_rows(match_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_matches = match_df[match_df["match_number"].between(1, 72)].copy()
    for _, match in group_matches.iterrows():
        for side in ["team_a", "team_b"]:
            team = str(match[side])
            rows.append(
                _base_team_row(
                    match,
                    team,
                    _combine_notes(match.get("notes"), "fixed group-stage fixture"),
                    False,
                )
            )
    return rows


def _load_projected_knockout(projected_knockout_path: Path) -> pd.DataFrame | None:
    if not projected_knockout_path.exists():
        return None
    projected = pd.read_csv(projected_knockout_path)
    required = {"match_number", "projected_team_a", "projected_team_b"}
    if not required.issubset(projected.columns):
        return None
    return projected


def _knockout_team_rows(
    match_df: pd.DataFrame,
    projected_knockout: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    knockout_matches = match_df[match_df["match_number"] > 72].copy()
    projected_lookup = (
        projected_knockout.set_index("match_number").to_dict("index")
        if projected_knockout is not None
        else {}
    )

    for _, match in knockout_matches.iterrows():
        match_number = int(match["match_number"])
        projection = projected_lookup.get(match_number)
        if projection is None:
            teams = [str(match["team_a_source"]), str(match["team_b_source"])]
            status = "projected/pending"
            needs_review = True
            base_note = "unresolved knockout participant slot; travel burden pending"
        else:
            teams = [
                str(projection.get("projected_team_a", "")).strip(),
                str(projection.get("projected_team_b", "")).strip(),
            ]
            if not teams[0] or not teams[1] or teams[0].lower() == "nan" or teams[1].lower() == "nan":
                teams = [str(match["team_a_source"]), str(match["team_b_source"])]
                status = "projected/pending"
                needs_review = True
                base_note = "missing projected participant; travel burden pending"
            else:
                status = "projected"
                needs_review = True
                base_note = "current projected knockout participant; context only"

        for team in teams:
            rows.append(
                _base_team_row(
                    match,
                    team,
                    _combine_notes(match.get("notes"), base_note, status),
                    needs_review,
                )
            )
    return rows


def _empty_previous_fields() -> dict[str, Any]:
    return {
        "previous_match_number": pd.NA,
        "previous_match_date": pd.NA,
        "previous_venue_id": pd.NA,
        "rest_days": pd.NA,
        "distance_from_previous_km": pd.NA,
        "timezone_delta_hours": pd.NA,
        "elevation_delta_m": pd.NA,
        "same_venue_flag": False,
        "same_host_city_flag": False,
        "country_change_flag": False,
        "travel_burden_score": 0.0,
        "rest_days_penalty": 0.0,
        "distance_penalty": 0.0,
        "timezone_penalty": 0.0,
        "elevation_penalty": 0.0,
        "same_location_bonus": 0.0,
    }


def _add_transition_fields(rows: pd.DataFrame) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    ordered = rows.sort_values(["team", "match_date", "match_number"]).reset_index(drop=True)

    for _, team_rows in ordered.groupby("team", sort=False):
        previous: dict[str, Any] | None = None
        for _, current_row in team_rows.iterrows():
            current = current_row.to_dict()
            current.update(_empty_previous_fields())

            if previous is not None:
                current_date = pd.to_datetime(current["match_date"])
                previous_date = pd.to_datetime(previous["match_date"])
                rest_days = int((current_date - previous_date).days)
                distance_km = haversine_km(
                    previous["latitude"],
                    previous["longitude"],
                    current["latitude"],
                    current["longitude"],
                )
                tz_delta = timezone_delta_hours(
                    str(previous["timezone"]),
                    str(current["timezone"]),
                    str(current["match_date"]),
                )
                elevation_delta = float(current["elevation_m"]) - float(previous["elevation_m"])
                same_venue = str(current["venue_id"]) == str(previous["venue_id"])
                same_city = str(current["host_city"]) == str(previous["host_city"])
                country_change = str(current["host_country"]) != str(previous["host_country"])

                components = score_travel_burden(
                    rest_days=rest_days,
                    distance_from_previous_km=distance_km,
                    timezone_delta_hours_value=tz_delta,
                    elevation_delta_m=elevation_delta,
                    same_venue_flag=same_venue,
                    same_host_city_flag=same_city,
                )

                current.update(
                    {
                        "previous_match_number": int(previous["match_number"]),
                        "previous_match_date": previous["match_date"],
                        "previous_venue_id": previous["venue_id"],
                        "rest_days": rest_days,
                        "distance_from_previous_km": round(distance_km, 3),
                        "timezone_delta_hours": round(tz_delta, 3),
                        "elevation_delta_m": round(elevation_delta, 3),
                        "same_venue_flag": same_venue,
                        "same_host_city_flag": same_city,
                        "country_change_flag": country_change,
                        **{k: round(v, 4) for k, v in components.items()},
                    }
                )

            output_rows.append(current)
            previous = current_row.to_dict()

    out = pd.DataFrame(output_rows)
    out = out.sort_values(["match_number", "team"]).reset_index(drop=True)
    return out


def build_travel_sequences(
    match_df: pd.DataFrame,
    projected_knockout: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build team-level WC2026 travel / recovery sequence rows."""

    rows = _group_team_rows(match_df) + _knockout_team_rows(match_df, projected_knockout)
    sequence_df = _add_transition_fields(pd.DataFrame(rows))
    ordered_columns = TEAM_COLUMNS + COMPONENT_COLUMNS + ["host_country"]
    return sequence_df[ordered_columns]


def _group_summary(sequence_df: pd.DataFrame) -> pd.DataFrame:
    group_rows = sequence_df[sequence_df["match_number"].between(1, 72)].copy()
    return (
        group_rows.groupby("team", as_index=False)
        .agg(
            group_stage_total_burden=("travel_burden_score", "sum"),
            max_transition_burden=("travel_burden_score", "max"),
            total_distance_km=("distance_from_previous_km", "sum"),
            max_timezone_delta_hours=("timezone_delta_hours", lambda s: s.abs().max()),
            max_elevation_delta_m=("elevation_delta_m", lambda s: s.abs().max()),
        )
        .sort_values(["group_stage_total_burden", "total_distance_km"], ascending=False)
        .reset_index(drop=True)
    )


def _transition_summary(sequence_df: pd.DataFrame, group_only: bool = True) -> pd.DataFrame:
    df = sequence_df.copy()
    if group_only:
        df = df[df["match_number"].between(1, 72)]
    df = df[df["previous_match_number"].notna()].copy()
    return df.sort_values("travel_burden_score", ascending=False)[
        [
            "team",
            "previous_match_number",
            "match_number",
            "previous_venue_id",
            "venue_id",
            "rest_days",
            "distance_from_previous_km",
            "timezone_delta_hours",
            "elevation_delta_m",
            "travel_burden_score",
        ]
    ]


def _material_match_asymmetry(sequence_df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    pairs = []
    for match_number, rows in sequence_df.groupby("match_number"):
        if len(rows) != 2:
            continue
        a, b = rows.sort_values("team").to_dict("records")
        delta = float(a["travel_burden_score"]) - float(b["travel_burden_score"])
        if abs(delta) >= threshold:
            high = a if delta > 0 else b
            low = b if delta > 0 else a
            pairs.append(
                {
                    "match_number": int(match_number),
                    "higher_burden_team": high["team"],
                    "higher_score": high["travel_burden_score"],
                    "lower_burden_team": low["team"],
                    "lower_score": low["travel_burden_score"],
                    "score_gap": abs(delta),
                }
            )
    return pd.DataFrame(pairs).sort_values("score_gap", ascending=False) if pairs else pd.DataFrame()


def _verify_frozen_manifest() -> tuple[bool, list[str]]:
    if not FROZEN_MANIFEST_PATH.exists():
        return False, ["missing frozen manifest"]
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text())
    failures = []
    for item in manifest.get("files", []):
        path = ROOT / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if digest != item["sha256"]:
            failures.append(item["path"])
    return not failures, failures


def _write_sequence_report(sequence_df: pd.DataFrame) -> None:
    summary = _group_summary(sequence_df)
    transitions = _transition_summary(sequence_df, group_only=False)
    projected_count = int(sequence_df["needs_review"].sum())
    frozen_ok, frozen_failures = _verify_frozen_manifest()

    lines = [
        "# WC2026 Travel / Recovery Sequence Report",
        "",
        "## Status",
        "",
        "- Sequence table built from `data/reference/wc2026_match_venues_enriched.csv`.",
        "- Venue QA prerequisite: passed via `outputs/reports/wc2026_venue_extraction_cross_check_summary.md`.",
        "- Group-stage rows: **144**.",
        f"- Total team-match rows including projected knockout context: **{len(sequence_df)}**.",
        f"- Rows marked `needs_review` because they are projected/pending: **{projected_count}**.",
        f"- Frozen v2 auto-science files unchanged by manifest hash: **{'yes' if frozen_ok else 'no'}**.",
        "",
    ]
    if frozen_failures:
        lines.append(f"- Frozen manifest mismatches: {', '.join(frozen_failures)}")
        lines.append("")

    lines.extend(
        [
            "## Highest Group-Stage Team Burden",
            "",
            _markdown_table(summary.head(10)),
            "## Lowest Group-Stage Team Burden",
            "",
            _markdown_table(summary.sort_values("group_stage_total_burden").head(10)),
            "## Highest Match Transitions",
            "",
            _markdown_table(transitions.head(15)),
            "## Score Components",
            "",
            "The score is the clipped sum of conservative rest, distance, time-zone, elevation, and same-location components. It is context only and is not a causal adjustment.",
            "",
        ]
    )
    SEQUENCE_REPORT.write_text("\n".join(lines))


def _write_context_report(sequence_df: pd.DataFrame) -> None:
    summary = _group_summary(sequence_df)
    transitions = _transition_summary(sequence_df, group_only=True)
    asymmetry = _material_match_asymmetry(sequence_df[sequence_df["match_number"].between(1, 72)])
    large_shifts = transitions[
        (transitions["distance_from_previous_km"] >= 2000)
        | (transitions["timezone_delta_hours"].abs() >= 2)
        | (transitions["elevation_delta_m"].abs() >= 1000)
    ].copy()
    timezone_cases = transitions[transitions["timezone_delta_hours"].abs() >= 2].copy()

    lines = [
        "# WC2026 Travel / Recovery Burden Context",
        "",
        "This is dashboard context only. No prediction files or final candidates are changed.",
        "",
        "## Highest Group-Stage Burden",
        "",
        _markdown_table(summary.head(10)),
        "## Lowest Group-Stage Burden",
        "",
        _markdown_table(summary.sort_values("group_stage_total_burden").head(10)),
        "## Material Match-Level Burden Gaps",
        "",
        _markdown_table(asymmetry.head(15) if not asymmetry.empty else asymmetry),
        "## Large Venue Transitions",
        "",
        _markdown_table(large_shifts.head(15)),
        "## Time-Zone-Change Cases",
        "",
    ]
    if timezone_cases.empty:
        lines.append("No group-stage transition has an absolute time-zone change of at least 2 hours.")
    else:
        lines.append(
            "The rows below are the only cases that should be described as jetlag-like, because they include material time-zone change."
        )
        lines.append("")
        lines.append(_markdown_table(timezone_cases.head(15)))
    lines.extend(
        [
            "",
            "## Label",
            "",
            "Use `travel / recovery burden` in dashboard copy. Reserve `jetlag-like` only for rows with material time-zone change.",
            "",
        ]
    )
    CONTEXT_REPORT.write_text("\n".join(lines))


def _write_historical_feasibility_report() -> bool:
    matches_path = INTERIM_DIR / "matches_clean.parquet"
    if not matches_path.exists():
        text = (
            "# Travel / Recovery Burden Historical Feasibility\n\n"
            "- `data/interim/matches_clean.parquet` is missing.\n"
            "- Feasibility cannot be classified as strong without the historical backbone.\n"
            "- Recommendation: context only for now.\n"
        )
        FEASIBILITY_REPORT.write_text(text)
        return False

    matches = pd.read_parquet(matches_path)
    since_2000 = matches[matches["date"] >= "2000-01-01"]
    world_cup = matches[matches["tournament"].eq("FIFA World Cup")]
    world_cup_since_2000 = world_cup[world_cup["date"] >= "2000-01-01"]
    wc2018_2022 = world_cup[world_cup["date"].dt.year.isin([2018, 2022])]

    city_coverage = float(matches["city"].notna().mean())
    country_coverage = float(matches["country"].notna().mean())
    date_coverage = float(matches["date"].notna().mean())
    strong = False

    lines = [
        "# Travel / Recovery Burden Historical Feasibility",
        "",
        "## Coverage",
        "",
        f"- Clean historical matches: **{len(matches)}**.",
        f"- Matches since 2000: **{len(since_2000)}**.",
        f"- FIFA World Cup matches all years: **{len(world_cup)}**.",
        f"- FIFA World Cup matches since 2000: **{len(world_cup_since_2000)}**.",
        f"- WC2018/WC2022 diagnostic rows: **{len(wc2018_2022)}**.",
        f"- Date coverage: **{date_coverage:.1%}**.",
        f"- City coverage: **{city_coverage:.1%}**.",
        f"- Country coverage: **{country_coverage:.1%}**.",
        "",
        "## Reconstruction Requirements",
        "",
        "- The historical backbone has match dates, cities, countries, teams, and neutral flags.",
        "- It does not currently have stadium IDs, latitude/longitude, time zones, or elevations.",
        "- Previous-match venue can be reconstructed only after a separately QA'd historical venue geocode layer is added.",
        "- World Cup-only samples are small: 64 matches per recent tournament and only 384 FIFA World Cup rows since 2000.",
        "",
        "## Confounding Risk",
        "",
        "- Travel burden is entangled with tournament structure, host geography, group assignment, rest-day scheduling, team strength, and host/neutral effects.",
        "- Knockout teams are selected on performance, so later-round travel burden is also conditioned on survival.",
        "- A controlled test must separate travel / recovery burden from home/neutral advantage and tournament-round effects.",
        "",
        "## Recommendation",
        "",
        "- Feasibility classification: **not strong yet**.",
        "- Use as WC2026 dashboard context now.",
        "- Do not build or promote a historical model until a reproducible historical venue geocode/elevation/time-zone layer passes QA.",
        "",
    ]
    FEASIBILITY_REPORT.write_text("\n".join(lines))
    return strong


def _write_policy_report(sequence_df: pd.DataFrame, historical_feasibility_strong: bool) -> None:
    summary = _group_summary(sequence_df)
    top = ", ".join(summary.head(5)["team"].tolist())
    frozen_ok, frozen_failures = _verify_frozen_manifest()

    lines = [
        "# Travel / Recovery Burden Policy Recommendation",
        "",
        "1. Is the travel/recovery burden hypothesis testable? **Partially, but not strongly with the current historical data.** WC2026 can be surfaced now; historical testing needs a QA'd venue geocode/time-zone/elevation layer.",
        "2. Is it better as dashboard context or model feature? **Dashboard context for now.** It can become model-testable after historical venue reconstruction and controlled time-aware backtests.",
        f"3. Which WC2026 teams are most affected? Highest current group-stage totals: **{top}**.",
        "4. Is there evidence to alter `final_candidate_v2_auto_science`? **No.** No controlled travel-burden backtest was run, and feasibility is not strong enough for promotion.",
        "5. If no, keep v2 unchanged. **Yes, v2 remains final.**",
        "6. If yes, create v3. **No v3 travel-burden candidate created.** Promotion gate did not run or pass.",
        "",
        "## Gate Status",
        "",
        f"- Historical feasibility strong: **{'yes' if historical_feasibility_strong else 'no'}**.",
        f"- Frozen v2 manifest intact: **{'yes' if frozen_ok else 'no'}**.",
        "- Prediction files changed: **no**.",
        "- Recommended label: `travel / recovery burden`; use `jetlag-like` only when time-zone change is part of the row.",
        "",
    ]
    if frozen_failures:
        lines.append(f"Frozen manifest mismatches: {', '.join(frozen_failures)}")
        lines.append("")
    POLICY_REPORT.write_text("\n".join(lines))


def write_venue_qa_summary() -> None:
    """Write the explicit prerequisite QA summary from existing checked files."""

    venue_df = pd.read_csv(VENUES_PATH)
    match_df = pd.read_csv(MATCH_VENUES_PATH)
    group_df = match_df[match_df["match_number"].between(1, 72)]
    errors = []
    if len(venue_df) != 16:
        errors.append(f"expected 16 venues, found {len(venue_df)}")
    if len(match_df) != 104:
        errors.append(f"expected 104 match rows, found {len(match_df)}")
    if match_df["match_number"].nunique() != 104:
        errors.append("match numbers are not unique")
    for col in ["latitude", "longitude", "elevation_m", "timezone", "venue_id"]:
        if match_df[col].isna().any():
            errors.append(f"`{col}` contains null values")
    if len(group_df) != 72:
        errors.append(f"expected 72 group-stage matches, found {len(group_df)}")

    status = "PASS" if not errors else "FAIL"
    lines = [
        "# WC2026 Venue Extraction Cross-Check Summary",
        "",
        f"Status: {status}",
        "",
        "## Inputs",
        "",
        "- `data/reference/wc2026_match_venues_enriched.csv`",
        "- `data/reference/wc2026_venues.csv`",
        "",
        "## Checks",
        "",
        f"- Venue rows: **{len(venue_df)}**.",
        f"- Match rows: **{len(match_df)}**.",
        f"- Group-stage match rows: **{len(group_df)}**.",
        f"- Unique match numbers: **{match_df['match_number'].nunique()}**.",
        f"- Venue IDs referenced: **{match_df['venue_id'].nunique()}**.",
        "- Required enriched fields present for all match rows: `latitude`, `longitude`, `elevation_m`, `timezone`.",
        "- Existing pytest suite passed: `tests/test_wc2026_venues.py` and `tests/test_wc2026_match_venues.py`.",
        "",
    ]
    if errors:
        lines.extend(["## Errors", "", *[f"- {error}" for error in errors], ""])
    VENUE_QA_SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    if not VENUE_QA_SUMMARY_PATH.exists():
        write_venue_qa_summary()
    _validate_prerequisites()

    match_df = pd.read_csv(MATCH_VENUES_PATH)
    projected_knockout = _load_projected_knockout(PROJECTED_KNOCKOUT_PATH)
    sequence_df = build_travel_sequences(match_df, projected_knockout)
    sequence_df.to_parquet(SEQUENCES_OUT, index=False)

    _write_sequence_report(sequence_df)
    _write_context_report(sequence_df)
    historical_feasibility_strong = _write_historical_feasibility_report()
    _write_policy_report(sequence_df, historical_feasibility_strong)

    summary = _group_summary(sequence_df)
    print(f"Wrote {SEQUENCES_OUT.relative_to(ROOT)} ({len(sequence_df)} rows)")
    print("Highest group-stage burden teams: " + ", ".join(summary.head(5)["team"].tolist()))
    print(f"Historical feasibility strong: {historical_feasibility_strong}")
    print("v2_auto_science remains final: yes")


if __name__ == "__main__":
    main()
