#!/usr/bin/env python3
"""Fetch the official FIFA World Cup 26 regulations PDF."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
SOURCE_URL = "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf"
OUT = ROOT / "data" / "raw" / "fifa_official" / "regulations" / "FWC2026_regulations_EN.pdf"
REPORT = ROOT / "outputs" / "reports" / "fifa_regulations_fetch_report.md"


def write_report(lines: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    status: int | str = "not attempted"
    content_type = ""
    error = ""
    cache_used_after_error = False

    try:
        response = requests.get(SOURCE_URL, timeout=90)
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        response.raise_for_status()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(response.content)
    except Exception as exc:  # pragma: no cover - integration/network path
        error = f"{type(exc).__name__}: {exc}"

    exists = OUT.exists()
    size = OUT.stat().st_size if exists else 0
    is_pdf = exists and OUT.read_bytes()[:5] == b"%PDF-"
    if error and exists and is_pdf:
        cache_used_after_error = True
    official = "digitalhub.fifa.com" in SOURCE_URL and "FWC2026_regulations_EN.pdf" in SOURCE_URL

    write_report(
        [
            "# FIFA Regulations PDF Fetch Report",
            "",
            f"- Source URL: `{SOURCE_URL}`",
            f"- Fetch timestamp UTC: `{fetched_at}`",
            f"- HTTP status: `{status}`",
            f"- Content type: `{content_type or 'unknown'}`",
            f"- Output file: `{OUT.relative_to(ROOT)}`",
            f"- File size bytes: **{size}**",
            f"- PDF exists: **{exists}**",
            f"- PDF signature valid: **{is_pdf}**",
            f"- Source treated as official FIFA regulations: **{official}**",
            f"- Valid cached PDF used after network error: **{cache_used_after_error}**",
            f"- Error: `{error or 'none'}`",
        ]
    )

    if (error and not cache_used_after_error) or not exists or not is_pdf:
        raise SystemExit("Official FIFA regulations PDF fetch failed; see fetch report.")


if __name__ == "__main__":
    main()
