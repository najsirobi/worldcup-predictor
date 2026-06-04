"""Parse FIFA World Cup 2026 official squad-list PDF text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

SNAPSHOT_DATE = date(2026, 6, 11)
VALID_POSITIONS = {"GK", "DF", "MF", "FW"}


@dataclass(frozen=True)
class ParsedSquads:
    players: pd.DataFrame
    exceptions: pd.DataFrame


def normalize_whitespace(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_extracted_name(value: object) -> str:
    text = normalize_whitespace(value)
    return re.sub(r"\b([A-Z])\s+([A-Z]{2,})", r"\1\2", text)


def parse_date(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(normalize_whitespace(value), format="%d/%m/%Y", errors="coerce")
    return parsed


def age_on(date_of_birth: object, *, on_date: date = SNAPSHOT_DATE) -> float:
    dob = parse_date(date_of_birth)
    if pd.isna(dob):
        return float("nan")
    born = dob.date()
    return float(on_date.year - born.year - ((on_date.month, on_date.day) < (born.month, born.day)))


def club_country_code(club: object) -> str:
    match = re.search(r"\(([A-Z]{3})\)\s*$", normalize_whitespace(club))
    return match.group(1) if match else ""


def _looks_upper_token(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return bool(letters) and token.upper() == token


def _fallback_player_name(names_blob: str) -> str:
    tokens = normalize_whitespace(names_blob).split()
    if not tokens:
        return ""
    upper_count = 0
    for token in tokens:
        if _looks_upper_token(token):
            upper_count += 1
            continue
        break
    if upper_count >= 2:
        return clean_extracted_name(" ".join(tokens[:upper_count]))
    if upper_count == 1 and len(tokens) >= 2:
        given = re.sub(r"([a-zà-ÿ])([A-ZÀ-Þ])", r"\1", tokens[1])
        return clean_extracted_name(f"{tokens[0]} {given}")
    return clean_extracted_name(tokens[0])


def source_footer(text: str) -> tuple[str, str]:
    footer = ""
    version = ""
    for line in text.splitlines():
        if "Version" in line and "Page" in line:
            footer = normalize_whitespace(line)
            match = re.search(r"Version\s+(\d+)", footer)
            version = f"Version {match.group(1)}" if match else ""
            break
    return version, footer


def parse_player_line(line: str, *, team: str, fifa_code: str, head_coach: str, source_file: str, source_version: str, source_footer_date: str) -> dict[str, object]:
    normalized = line.rstrip()
    match = re.match(r"^\s*(\d{1,2})\s*(GK|DF|MF|FW)\s+(.+?)\s*$", normalized)
    if not match:
        raise ValueError("line does not start with shirt number and valid position")

    player_number = int(match.group(1))
    position = match.group(2)
    rest = match.group(3)
    parts = [normalize_whitespace(part) for part in re.split(r"\s{2,}", rest) if normalize_whitespace(part)]
    parse_note = "pypdf_layout_text; no inferred squad membership"
    if len(parts) >= 5:
        player_name = clean_extracted_name(parts[0])
        first_names = clean_extracted_name(parts[1])
        last_names = clean_extracted_name(parts[2])
        date_part_indexes = [idx for idx, part in enumerate(parts) if re.search(r"\d{2}/\d{2}/\d{4}", part)]
        if date_part_indexes:
            date_idx = date_part_indexes[0]
            height_raw = parts[-1] if date_idx < len(parts) - 1 else ""
            name_on_shirt = clean_extracted_name(parts[3]) if date_idx > 3 else ""
            dob_and_club = " ".join(parts[date_idx:-1]) if date_idx < len(parts) - 1 else parts[date_idx]
            parse_note = (
                "pypdf_layout_text; shifted_date_column_fallback; no inferred squad membership"
                if date_idx != 4
                else parse_note
            )
        else:
            height_raw = parts[-1] if len(parts) > 5 else ""
            name_on_shirt = clean_extracted_name(parts[3])
            dob_and_club = " ".join(parts[4:-1]) if len(parts) > 5 else parts[4]
    else:
        date_match = re.search(r"\d{2}/\d{2}/\d{4}", rest)
        if not date_match:
            raise ValueError("DOB not found in row")
        names_blob = rest[: date_match.start()]
        dob_and_club = rest[date_match.start() :]
        height_match_fallback = re.search(r"\s(\d{2,3})\s*$", dob_and_club)
        height_raw = height_match_fallback.group(1) if height_match_fallback else ""
        if height_match_fallback:
            dob_and_club = dob_and_club[: height_match_fallback.start()].strip()
        player_name = _fallback_player_name(names_blob)
        first_names = ""
        last_names = ""
        name_on_shirt = ""
        parse_note = "pypdf_layout_text; collapsed_name_columns_fallback; no inferred squad membership"

    dob_match = re.search(r"(\d{2}/\d{2}/\d{4})(.*)$", dob_and_club)
    if not dob_match:
        raise ValueError("DOB not found in row")
    dob_text = dob_match.group(1)
    club = normalize_whitespace(dob_match.group(2))
    height_match = re.match(r"^\d{2,3}$", height_raw)
    height = int(height_raw) if height_match else pd.NA

    return {
        "team": team,
        "fifa_code": fifa_code,
        "player_number": player_number,
        "position": position,
        "player_name": player_name,
        "first_names": first_names,
        "last_names": last_names,
        "name_on_shirt": name_on_shirt,
        "date_of_birth": parse_date(dob_text).date().isoformat(),
        "age_on_2026_06_11": age_on(dob_text),
        "club": club,
        "club_country_code": club_country_code(club),
        "height_cm": height,
        "head_coach": head_coach,
        "source_file": source_file,
        "source_version": source_version,
        "source_timestamp_or_footer_date": source_footer_date,
        "parse_notes": parse_note,
    }


def parse_head_coach(text: str) -> str:
    for line in text.splitlines():
        if re.match(r"^\s*Head coach\s+", line):
            parts = [normalize_whitespace(part) for part in re.split(r"\s{2,}", line) if normalize_whitespace(part)]
            if len(parts) >= 2:
                return clean_extracted_name(parts[1])
    return ""


def parse_page_text(text: str, *, source_file: str = "") -> ParsedSquads:
    version, footer = source_footer(text)
    team = ""
    fifa_code = ""
    for line in text.splitlines():
        match = re.match(r"^\s*(.+?)\s+\(([A-Z]{3})\)\s*$", normalize_whitespace(line))
        if match:
            team = normalize_whitespace(match.group(1))
            fifa_code = match.group(2)
            break
    if not team:
        raise ValueError("Team header not found in page text")

    head_coach = parse_head_coach(text)
    rows: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d{1,2}\s*(GK|DF|MF|FW)\s+", line):
            continue
        try:
            rows.append(
                parse_player_line(
                    line,
                    team=team,
                    fifa_code=fifa_code,
                    head_coach=head_coach,
                    source_file=source_file,
                    source_version=version,
                    source_footer_date=footer,
                )
            )
        except Exception as exc:
            exceptions.append(
                {
                    "team": team,
                    "fifa_code": fifa_code,
                    "raw_line": normalize_whitespace(line),
                    "parse_error": f"{type(exc).__name__}: {exc}",
                    "source_file": source_file,
                    "source_version": version,
                    "source_timestamp_or_footer_date": footer,
                }
            )
    return ParsedSquads(pd.DataFrame(rows), pd.DataFrame(exceptions))


def parse_pdf(path: str | Path) -> ParsedSquads:
    source_path = Path(path)
    reader = PdfReader(str(source_path))
    players: list[pd.DataFrame] = []
    exceptions: list[pd.DataFrame] = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        parsed = parse_page_text(unicodedata.normalize("NFC", text), source_file=str(source_path))
        players.append(parsed.players)
        if not parsed.exceptions.empty:
            exceptions.append(parsed.exceptions)

    players_df = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    exceptions_df = pd.concat(exceptions, ignore_index=True) if exceptions else pd.DataFrame(
        columns=[
            "team",
            "fifa_code",
            "raw_line",
            "parse_error",
            "source_file",
            "source_version",
            "source_timestamp_or_footer_date",
        ]
    )
    return ParsedSquads(players_df, exceptions_df)
