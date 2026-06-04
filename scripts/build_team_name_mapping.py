#!/usr/bin/env python3
"""Team-name mapping expansion support (Phase 3, Task B).

- Audits backbone vs. rating-source team names.
- Adds ONLY safe exact-match mappings (model-era) to team_name_map.csv with
  evidence, preserving existing rows.
- Writes alias / renamed / disputed candidates to team_name_map_candidates.csv
  with needs_review=true (NOT applied to production joins).
"""
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.ingest.common import select_dataset_csv
from src.ingest.team_names import load_team_name_map, normalize_team_whitespace

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
REF = REPO_ROOT / "data" / "reference"
KAGGLE = REPO_ROOT / "data" / "raw" / "kaggle"
MAP_PATH = REF / "team_name_map.csv"
CAND_PATH = REF / "team_name_map_candidates.csv"
TEMPLATE_PATH = REF / "fif8a_group_stage_template.csv"
MATCHES_PATH = REPO_ROOT / "data" / "interim" / "matches_clean.parquet"
AUDIT_PATH = REPO_ROOT / "outputs" / "reports" / "team_name_mapping_audit.md"

ELO_SOURCE_KEY = "international_elo"
FIFA_SOURCE_KEY = "fifa_ranking"

# Curated, source-reviewable alias dictionary: backbone name -> rating-source
# alias for well-known renames where the names are NOT byte-identical. These are
# emitted as CANDIDATES (needs_review=true), never auto-applied. A backbone team
# may be exact in one source and aliased in another; the per-source logic below
# only emits where the source actually contains the alias.
CURATED_ALIASES = {
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "United States": "USA",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "China PR": "China",              # backbone "China PR" exact in FIFA; Elo calls it "China"
    "Republic of Ireland": "Ireland",  # exact in FIFA; Elo calls it "Ireland"
    "Brunei": "Brunei Darussalam",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent / Grenadines",
}


def _norm(s: str) -> str:
    """Casefold + de-accent + strip non-alphanumerics (spelling-variant key)."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    matches = pd.read_parquet(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home_team"] = normalize_team_whitespace(matches["home_team"])
    matches["away_team"] = normalize_team_whitespace(matches["away_team"])
    backbone_teams = set(matches["home_team"]) | set(matches["away_team"])

    m2000 = matches[matches["date"] >= "2000-01-01"]
    teams_2000 = set(m2000["home_team"]) | set(m2000["away_team"])

    template = pd.read_csv(TEMPLATE_PATH)
    template_teams = set(normalize_team_whitespace(template["team_a"])) | set(normalize_team_whitespace(template["team_b"]))

    # Raw rating-source names (pre-canonical); normalize whitespace (Elo uses NBSP).
    elo_raw = pd.read_csv(select_dataset_csv(KAGGLE / "international_elo", preferred_names=["eloratings.csv"]))
    elo_names = set(normalize_team_whitespace(elo_raw["team"]))
    fifa_file = select_dataset_csv(KAGGLE / "fifa_world_ranking", pick="last")
    fifa_raw = pd.read_csv(fifa_file)
    fifa_raw["country_full"] = normalize_team_whitespace(fifa_raw["country_full"])
    fifa_names = set(fifa_raw["country_full"])
    fifa_code = dict(zip(fifa_raw["country_full"], fifa_raw["country_abrv"].astype(str).str.strip()))
    rating_names = elo_names | fifa_names
    rating_norm = {}
    for nm in rating_names:
        rating_norm.setdefault(_norm(nm), nm)

    existing_map = load_team_name_map()
    existing_keys = set(zip(existing_map["source"], existing_map["raw_name"]))

    # ---- SAFE exact-match additions (model-era teams only) ----
    safe_rows = []
    for src_key, names in ((ELO_SOURCE_KEY, elo_names), (FIFA_SOURCE_KEY, fifa_names)):
        for team in sorted(teams_2000 & names):
            if (src_key, team) in existing_keys:
                continue
            code = fifa_code.get(team, "")
            safe_rows.append({
                "source": src_key,
                "raw_name": team,
                "canonical_team_name": team,  # identity: exact match is unambiguous
                "country_code": code,
                "notes": f"safe exact match: present in {('eloratings.csv' if src_key==ELO_SOURCE_KEY else fifa_file.name)} "
                         f"and matches_clean backbone (identity canonical)",
            })

    if safe_rows:
        add_df = pd.DataFrame(safe_rows, columns=existing_map.columns)
        combined = pd.concat([existing_map, add_df], ignore_index=True)
        combined = combined.drop_duplicates(["source", "raw_name"], keep="first")
        combined.to_csv(MAP_PATH, index=False)
    logger.info(f"  ✓ Safe exact mappings added to team_name_map.csv: {len(safe_rows)}")

    # ---- CANDIDATES (needs_review): per-source alias of backbone teams ----
    # Each candidate maps a RATING-source raw name -> the backbone canonical, so
    # it is directly promotable: adding (source, raw_name, canonical) to
    # team_name_map.csv makes that source join the backbone team.
    src_norm = {
        ELO_SOURCE_KEY: {},
        FIFA_SOURCE_KEY: {},
    }
    for nm in elo_names:
        src_norm[ELO_SOURCE_KEY].setdefault(_norm(nm), nm)
    for nm in fifa_names:
        src_norm[FIFA_SOURCE_KEY].setdefault(_norm(nm), nm)

    candidates = []
    seen = set()
    focus_teams = sorted((teams_2000 | template_teams))
    for team in focus_teams:
        for src_key, names in ((ELO_SOURCE_KEY, elo_names), (FIFA_SOURCE_KEY, fifa_names)):
            if team in names:
                continue  # exact join with this source
            alias, method = None, None
            if team in CURATED_ALIASES and CURATED_ALIASES[team] in names:
                alias, method = CURATED_ALIASES[team], "curated_alias"
            elif _norm(team) in src_norm[src_key] and src_norm[src_key][_norm(team)] != team:
                alias, method = src_norm[src_key][_norm(team)], "deaccent_case"
            if alias is None or alias == team:
                continue
            key = (alias, src_key)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "raw_name": alias,
                "source": src_key,
                "candidate_canonical_team_name": team,
                "candidate_country_code": fifa_code.get(alias, "") if src_key == FIFA_SOURCE_KEY else "",
                "match_method": method,
                "confidence": "high",
                "evidence": f"backbone '{team}' not in {src_key}; source has '{alias}' "
                            f"({method}); maps source name -> backbone canonical",
                "needs_review": True,
                "notes": f"promote to team_name_map.csv as ({src_key}, '{alias}', '{team}') to join",
            })

    cand_df = pd.DataFrame(candidates, columns=[
        "raw_name", "source", "candidate_canonical_team_name", "candidate_country_code",
        "match_method", "confidence", "evidence", "needs_review", "notes",
    ]).sort_values(["candidate_canonical_team_name", "source"])
    cand_df.to_csv(CAND_PATH, index=False)
    logger.info(f"  ✓ Candidate mappings (needs_review): {len(cand_df)} -> {CAND_PATH.name}")

    # Teams (model-era / template) with NO rating match at all -> report only.
    teams_with_candidate = set(cand_df["candidate_canonical_team_name"])
    no_match = sorted(
        t for t in focus_teams
        if t not in elo_names and t not in fifa_names
        and t not in teams_with_candidate
    )

    # ---- AUDIT REPORT ----
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "w") as f:
        f.write("# Team-Name Mapping Audit\n\n")
        f.write("## Inputs\n\n")
        f.write(f"- Backbone: `data/interim/matches_clean.parquet` "
                f"({len(backbone_teams)} distinct teams; {len(teams_2000)} in matches from 2000+)\n")
        f.write(f"- Elo names: `eloratings.csv` ({len(elo_names)} teams)\n")
        f.write(f"- FIFA names: `{fifa_file.name}` ({len(fifa_names)} teams)\n")
        f.write(f"- Existing map: `team_name_map.csv` ({len(existing_map)} rows before this run)\n")
        f.write(f"- FIF8A template teams: {len(template_teams)}\n\n")

        f.write("## Exact-match coverage (safe, identity joins)\n\n")
        f.write(f"- Backbone teams exactly present in Elo: **{len(backbone_teams & elo_names)}**\n")
        f.write(f"- Backbone teams exactly present in FIFA: **{len(backbone_teams & fifa_names)}**\n")
        f.write(f"- Model-era (2000+) teams exact in Elo: **{len(teams_2000 & elo_names)}** / {len(teams_2000)}\n")
        f.write(f"- Model-era (2000+) teams exact in FIFA: **{len(teams_2000 & fifa_names)}** / {len(teams_2000)}\n\n")

        f.write("## Safe mappings added to team_name_map.csv\n\n")
        f.write(f"- Added **{len(safe_rows)}** explicit exact-match (identity) rows "
                "for sources `international_elo` / `fifa_ranking` (model-era teams), with evidence.\n")
        f.write("- Existing mappings preserved; no mappings deleted.\n")
        f.write("- Rule: only exact cross-source matches are auto-added; aliases are NOT.\n\n")

        f.write("## Candidate mappings (needs_review = true)\n\n")
        f.write(f"- **{len(cand_df)}** candidates written to `team_name_map_candidates.csv`.\n")
        f.write("- These are renamed/aliased FIFA members (e.g. Cape Verde→Cabo Verde, "
                "South Korea→Korea Republic, Iran→IR Iran). They are **not** used in production "
                "joins until a human promotes them to `team_name_map.csv`.\n\n")
        if len(cand_df):
            f.write("| backbone name | candidate canonical | method | confidence |\n")
            f.write("|---|---|---|---|\n")
            for _, r in cand_df.iterrows():
                f.write(f"| {r['raw_name']} | {r['candidate_canonical_team_name']} | "
                        f"{r['match_method']} | {r['confidence']} |\n")
            f.write("\n")

        f.write("## Model-era / template teams with NO rating-source match\n\n")
        f.write(f"- **{len(no_match)}** teams (mostly non-FIFA / CONIFA / regional / disputed) "
                "have no Elo or FIFA entry and are intentionally left unmapped (no guessing).\n")
        if no_match:
            f.write("\n<details><summary>list</summary>\n\n" + ", ".join(no_match[:80]) + "\n\n</details>\n")
        f.write("\n## Status\n\n")
        f.write("- No fuzzy mapping was applied to production joins. No model trained.\n")

    logger.info(f"  ✓ Wrote {AUDIT_PATH}")
    logger.info(f"\nsafe_added={len(safe_rows)} candidates={len(cand_df)} no_match={len(no_match)}")


if __name__ == "__main__":
    main()
