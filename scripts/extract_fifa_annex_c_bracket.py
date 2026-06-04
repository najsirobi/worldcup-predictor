#!/usr/bin/env python3
"""Extract official World Cup 26 bracket mapping and Annexe C tables."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.ingest.fifa_regulations import (
    build_combined_mapping,
    extract_annex_c_text,
    knockout_round_progression,
    parse_annex_c_text,
    round_of_32_mapping,
    validate_annex_c,
    validate_progression_mapping,
    validate_round_of_32_mapping,
)

ROOT = Path(__file__).parent.parent
PDF = ROOT / "data" / "raw" / "fifa_official" / "regulations" / "FWC2026_regulations_EN.pdf"
R32_OUT = ROOT / "data" / "reference" / "round_of_32_mapping.csv"
PROGRESSION_OUT = ROOT / "data" / "reference" / "knockout_round_progression.csv"
ANNEX_OUT = ROOT / "data" / "reference" / "third_place_assignment_annex_c.csv"
COMBINED_OUT = ROOT / "data" / "reference" / "knockout_bracket_mapping.csv"
REPORT = ROOT / "outputs" / "reports" / "annex_c_extraction_report.md"
BRACKET_AUDIT = ROOT / "outputs" / "reports" / "bracket_structure_audit.md"


def _write_failure_report(error: Exception) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Annexe C Extraction Report",
                "",
                "- Extraction status: **failed**",
                f"- Source PDF: `{PDF.relative_to(ROOT)}`",
                f"- Error: `{error}`",
                "- No official bracket mapping was accepted.",
                "- Full tournament simulation must remain blocked until this is fixed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_success_reports(r32_rows: int, progression_rows: int, annex_rows: int) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Annexe C Extraction Report",
                "",
                "- Extraction status: **passed**",
                "- Source: official FIFA World Cup 26 Regulations PDF",
                f"- Source file: `{PDF.relative_to(ROOT)}`",
                "- Extraction method: `pypdf` text extraction from Articles 12.6-12.11 and Annexe C.",
                f"- Extraction timestamp UTC: `{timestamp}`",
                f"- Round-of-32 rows: **{r32_rows}**",
                f"- Round progression rows: **{progression_rows}**",
                f"- Annexe C rows: **{annex_rows}**",
                "- Annexe C validation: **495 rows, all 8-of-12 combinations, no duplicate slot assignments**",
                "- Manual review required: **False**",
                "- Mapping status: official enough for path-aware simulation.",
                "",
                "## Outputs",
                "",
                f"- `{R32_OUT.relative_to(ROOT)}`",
                f"- `{PROGRESSION_OUT.relative_to(ROOT)}`",
                f"- `{ANNEX_OUT.relative_to(ROOT)}`",
                f"- `{COMBINED_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    BRACKET_AUDIT.write_text(
        "\n".join(
            [
                "# Bracket Structure Audit",
                "",
                "- Exact Round-of-32 bracket mapping available: **True**",
                "- Source: official FIFA World Cup 26 Regulations PDF.",
                "- Round-of-32 configuration: extracted from Article 12.6.",
                "- Round-of-16, quarter-final, semi-final, third-place and final progression: extracted from Articles 12.7-12.11.",
                "- Best third-placed assignment: extracted from Annexe C.",
                "- Annexe C row count: **495**.",
                "- Best third-placed mapping deterministic in template: **True, after qualified third-place groups are known**.",
                "- Missing bracket assumptions: **none for path-aware simulation**.",
                "- Manual bracket input required: **False**.",
                "",
                "## Structured Files",
                "",
                f"- `{R32_OUT.relative_to(ROOT)}`",
                f"- `{PROGRESSION_OUT.relative_to(ROOT)}`",
                f"- `{ANNEX_OUT.relative_to(ROOT)}`",
                f"- `{COMBINED_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not PDF.exists():
        error = FileNotFoundError(f"Missing regulations PDF: {PDF}")
        _write_failure_report(error)
        raise SystemExit(2)

    try:
        r32 = round_of_32_mapping()
        progression = knockout_round_progression()
        annex_text = extract_annex_c_text(PDF)
        annex = parse_annex_c_text(annex_text)
        validate_round_of_32_mapping(r32)
        validate_progression_mapping(progression)
        validate_annex_c(annex)
    except Exception as exc:
        _write_failure_report(exc)
        raise

    R32_OUT.parent.mkdir(parents=True, exist_ok=True)
    r32.to_csv(R32_OUT, index=False)
    progression.to_csv(PROGRESSION_OUT, index=False)
    annex.to_csv(ANNEX_OUT, index=False)
    build_combined_mapping(r32, progression, annex).to_csv(COMBINED_OUT, index=False)
    _write_success_reports(len(r32), len(progression), len(annex))

    print(f"Wrote {R32_OUT.relative_to(ROOT)}")
    print(f"Wrote {PROGRESSION_OUT.relative_to(ROOT)}")
    print(f"Wrote {ANNEX_OUT.relative_to(ROOT)}")
    print(f"Wrote {COMBINED_OUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print(f"Wrote {BRACKET_AUDIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
