#!/usr/bin/env python3
"""Fetch the official FIFA World Cup 2026 squad-list PDF."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
SOURCE_URL = "https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf"
OUT = ROOT / "data" / "raw" / "fifa_official" / "squads" / "SquadLists-English.pdf"
REPORT = ROOT / "outputs" / "reports" / "fifa_squad_pdf_fetch_report.md"


def write_report(lines: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    status: int | str = "not attempted"
    content_type = ""
    error = ""

    try:
        response = requests.get(SOURCE_URL, timeout=60)
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        response.raise_for_status()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(response.content)
    except Exception as exc:  # pragma: no cover - exercised by integration run
        error = f"{type(exc).__name__}: {exc}"

    exists = OUT.exists()
    size = OUT.stat().st_size if exists else 0
    is_pdf = exists and OUT.read_bytes()[:5] == b"%PDF-"
    is_official = "fdp.fifa.org" in SOURCE_URL and "/assetspublic/" in SOURCE_URL

    lines = [
        "# FIFA Squad PDF Fetch Report",
        "",
        f"- Source URL: `{SOURCE_URL}`",
        f"- Fetch timestamp UTC: `{fetched_at}`",
        f"- HTTP status: `{status}`",
        f"- Content type: `{content_type or 'unknown'}`",
        f"- Output file: `{OUT.relative_to(ROOT)}`",
        f"- File size bytes: **{size}**",
        f"- PDF exists: **{exists}**",
        f"- PDF signature valid: **{is_pdf}**",
        f"- Official FIFA/FDP asset URL: **{is_official}**",
        f"- Error: `{error or 'none'}`",
    ]
    write_report(lines)

    if error or not exists or not is_pdf:
        raise SystemExit("Official FIFA squad PDF fetch failed; see fetch report.")


if __name__ == "__main__":
    main()
