#!/usr/bin/env python3
"""Phase 3 Task C+D: clean rating tables + strict no-leakage as-of ratings join.

Outputs:
- data/interim/elo_ratings_clean.parquet
- data/interim/fifa_rankings_clean.parquet
- data/interim/matches_with_ratings.parquet
- outputs/reports/ratings_join_report.md

For each match, attaches the latest Elo / FIFA rating dated STRICTLY BEFORE the
match (merge_asof, allow_exact_matches=False). Missing teams -> NaN + missing
flags (never guessed). No model trained; no other features added.
"""
import logging
from pathlib import Path

import pandas as pd

from src.ingest.team_names import load_team_name_map, canonicalize_team_series, normalize_team_whitespace
from src.ingest.ratings import (
    clean_elo_ratings, clean_fifa_rankings, validate_clean_ratings, asof_rating_join,
    ELO_CLEAN_COLUMNS, FIFA_CLEAN_COLUMNS,
)
from src.features.rating_momentum import (
    MOMENTUM_FEATURES,
    add_rating_momentum_features,
    validate_rating_momentum_no_leakage,
)
from src.ingest.rules_and_scoring import load_scoring_rules, validate_scoring_rules
from src.ingest.fif8a_template import load_fif8a_group_template, validate_fif8a_group_template

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
INTERIM = ROOT / "data" / "interim"
MATCHES_CLEAN = INTERIM / "matches_clean.parquet"
ELO_OUT = INTERIM / "elo_ratings_clean.parquet"
FIFA_OUT = INTERIM / "fifa_rankings_clean.parquet"
JOIN_OUT = INTERIM / "matches_with_ratings.parquet"
REPORT = ROOT / "outputs" / "reports" / "ratings_join_report.md"
TEMPLATE_PATH = ROOT / "data" / "reference" / "fif8a_group_stage_template.csv"


def _missing_reason(team_series, valued_series, rating_canon_set, earliest_date_by_team, match_dates):
    """Classify why a rating is missing: 'absent_team' vs 'no_prior_date'."""
    absent = 0
    no_prior = 0
    for team, val, mdate in zip(team_series, valued_series, match_dates):
        if pd.notna(val):
            continue
        if team not in rating_canon_set:
            absent += 1
        else:
            earliest = earliest_date_by_team.get(team)
            if earliest is None or earliest >= mdate:
                no_prior += 1
            else:
                no_prior += 1  # shouldn't happen, but classify safely
    return absent, no_prior


def main():
    mapping_df = load_team_name_map()

    # ---- Task C: clean rating tables ----
    logger.info("Cleaning Elo ratings...")
    elo = clean_elo_ratings(mapping_df)
    validate_clean_ratings(elo, "rating_date", ["elo_rating"])
    elo.to_parquet(ELO_OUT, index=False)
    logger.info(f"  ✓ {ELO_OUT} ({len(elo)} rows)")

    logger.info("Cleaning FIFA rankings...")
    fifa = clean_fifa_rankings(mapping_df)
    validate_clean_ratings(fifa, "ranking_date", ["fifa_rank", "fifa_points"])
    fifa.to_parquet(FIFA_OUT, index=False)
    logger.info(f"  ✓ {FIFA_OUT} ({len(fifa)} rows)")

    # ---- Load matches, canonicalize team names (whitespace-normalized) ----
    m = pd.read_parquet(MATCHES_CLEAN)
    m["date"] = pd.to_datetime(m["date"])
    home_ws = normalize_team_whitespace(m["home_team"])
    away_ws = normalize_team_whitespace(m["away_team"])
    m["home_canon"], _ = canonicalize_team_series(home_ws, "international_results", mapping_df)
    m["away_canon"], _ = canonicalize_team_series(away_ws, "international_results", mapping_df)

    # ---- Task D: strict as-of joins (rating_date < match_date) ----
    out = m.copy()

    he = asof_rating_join(m, elo, "home_canon", ["elo_rating"], "home", rating_date_col="rating_date")
    ae = asof_rating_join(m, elo, "away_canon", ["elo_rating"], "away", rating_date_col="rating_date")
    out["home_elo"] = he["home_elo_rating"].values
    out["away_elo"] = ae["away_elo_rating"].values
    out["home_elo_rating_date"] = he["home_rating_date"].values
    out["away_elo_rating_date"] = ae["away_rating_date"].values

    hf = asof_rating_join(m, fifa, "home_canon", ["fifa_rank", "fifa_points"], "home", rating_date_col="ranking_date")
    af = asof_rating_join(m, fifa, "away_canon", ["fifa_rank", "fifa_points"], "away", rating_date_col="ranking_date")
    out["home_fifa_rank"] = hf["home_fifa_rank"].values
    out["home_fifa_points"] = hf["home_fifa_points"].values
    out["home_fifa_ranking_date"] = hf["home_rating_date"].values
    out["away_fifa_rank"] = af["away_fifa_rank"].values
    out["away_fifa_points"] = af["away_fifa_points"].values
    out["away_fifa_ranking_date"] = af["away_rating_date"].values

    # diffs (home - away); NaN if either side missing
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["fifa_rank_diff"] = out["home_fifa_rank"] - out["away_fifa_rank"]
    out["fifa_points_diff"] = out["home_fifa_points"] - out["away_fifa_points"]

    # presence flags (FIFA presence keyed on points, the more complete field)
    out["has_home_elo"] = out["home_elo"].notna()
    out["has_away_elo"] = out["away_elo"].notna()
    out["has_home_fifa"] = out["home_fifa_points"].notna()
    out["has_away_fifa"] = out["away_fifa_points"].notna()
    out["has_complete_elo"] = out["has_home_elo"] & out["has_away_elo"]
    out["has_complete_fifa"] = out["has_home_fifa"] & out["has_away_fifa"]
    out["has_complete_ratings"] = out["has_complete_elo"] & out["has_complete_fifa"]

    out = add_rating_momentum_features(out, elo, fifa)
    validate_rating_momentum_no_leakage(out)

    # leakage assertion: every used rating date must be strictly before match date
    for dcol in ["home_elo_rating_date", "away_elo_rating_date",
                 "home_fifa_ranking_date", "away_fifa_ranking_date"]:
        used = out[dcol].notna()
        if (out.loc[used, dcol] >= out.loc[used, "date"]).any():
            raise ValueError(f"LEAKAGE: {dcol} not strictly before match date for some rows")

    out = out.drop(columns=["home_canon", "away_canon"])
    out.to_parquet(JOIN_OUT, index=False)
    logger.info(f"  ✓ {JOIN_OUT} ({len(out)} rows)")

    _write_report(out, elo, fifa)

    # ---- Task E: confirm scoring + template still validate ----
    rules = load_scoring_rules(); validate_scoring_rules(rules)
    tmpl = load_fif8a_group_template(); validate_fif8a_group_template(tmpl, require_full=True)
    logger.info("  ✓ scoring_rules.yml valid; fif8a_group_stage_template.csv valid")


def _coverage(df, label):
    n = len(df)
    if n == 0:
        return f"| {label} | 0 | – | – | – |\n"
    return (f"| {label} | {n} | {df['has_complete_elo'].mean()*100:.1f}% | "
            f"{df['has_complete_fifa'].mean()*100:.1f}% | {df['has_complete_ratings'].mean()*100:.1f}% |\n")


def _write_report(out, elo, fifa):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    wc = out[out["tournament"] == "FIFA World Cup"]
    m1992 = out[out["date"] >= "1992-01-01"]
    m2000 = out[out["date"] >= "2000-01-01"]

    # missing teams (rows missing complete elo): which canonical teams
    def top_missing(side_flag, home_team_col="home_team", away_team_col="away_team", val_home=None, val_away=None):
        miss = pd.concat([
            out.loc[~out[val_home], home_team_col],
            out.loc[~out[val_away], away_team_col],
        ])
        return miss.value_counts().head(15)

    miss_elo = top_missing(None, val_home="has_home_elo", val_away="has_away_elo")
    miss_fifa = top_missing(None, val_home="has_home_fifa", val_away="has_away_fifa")

    elo_canon = set(elo["canonical_team_name"]); fifa_canon = set(fifa["canonical_team_name"])
    elo_earliest = elo.groupby("canonical_team_name")["rating_date"].min().to_dict()
    fifa_earliest = fifa.groupby("canonical_team_name")["ranking_date"].min().to_dict()

    # reason classification on home side (representative)
    elo_absent, elo_noprior = _missing_reason(
        out["home_team"].where(out["has_home_elo"].eq(False)), out["home_elo"],
        elo_canon, elo_earliest, out["date"])
    fifa_absent, fifa_noprior = _missing_reason(
        out["home_team"].where(out["has_home_fifa"].eq(False)), out["home_fifa_points"],
        fifa_canon, fifa_earliest, out["date"])

    examples = out[out["has_complete_elo"]].head(5)

    with open(REPORT, "w") as f:
        f.write("# Ratings Join Report\n\n")
        f.write("As-of join of Elo + FIFA ratings onto the historical match backbone, using "
                "the **strict no-future-leakage rule**: each rating is the latest dated "
                "**strictly before** the match (`merge_asof(..., allow_exact_matches=False)`).\n\n")

        f.write("## Inputs\n\n")
        f.write(f"- Matches: `data/interim/matches_clean.parquet` ({len(out)} rows)\n")
        f.write(f"- Elo (cleaned): {len(elo)} rows, {elo['rating_date'].min().date()} → {elo['rating_date'].max().date()}, "
                f"{elo['canonical_team_name'].nunique()} teams\n")
        f.write(f"- FIFA (cleaned): {len(fifa)} rows, {fifa['ranking_date'].min().date()} → {fifa['ranking_date'].max().date()}, "
                f"{fifa['canonical_team_name'].nunique()} teams\n\n")

        f.write("## Coverage (complete = both teams have a rating)\n\n")
        f.write("| Segment | Rows | Complete Elo | Complete FIFA | Complete Both |\n")
        f.write("|---|--:|--:|--:|--:|\n")
        f.write(_coverage(out, "All history"))
        f.write(_coverage(m1992, "1992 onward"))
        f.write(_coverage(m2000, "2000 onward"))
        f.write(_coverage(wc, "FIFA World Cup matches"))
        f.write("\n")
        f.write(f"- Rows with complete Elo: **{int(out['has_complete_elo'].sum())}** / {len(out)}\n")
        f.write(f"- Rows with complete FIFA: **{int(out['has_complete_fifa'].sum())}** / {len(out)}\n")
        f.write(f"- Rows with complete Elo **and** FIFA: **{int(out['has_complete_ratings'].sum())}** / {len(out)}\n\n")

        f.write("## Most common teams with MISSING Elo\n\n")
        f.write("```\n" + miss_elo.to_string() + "\n```\n\n")
        f.write("## Most common teams with MISSING FIFA\n\n")
        f.write("```\n" + miss_fifa.to_string() + "\n```\n\n")

        f.write("## Missing-rating reasons (home side)\n\n")
        f.write(f"- Elo missing: **{elo_absent}** absent team (no canonical match / unmapped alias), "
                f"**{elo_noprior}** no rating dated before the match (early history)\n")
        f.write(f"- FIFA missing: **{fifa_absent}** absent team, **{fifa_noprior}** no ranking before match "
                "(FIFA rankings start 1992-12-31, so all pre-1993 matches are 'no prior')\n\n")

        f.write("## Rating dates used\n\n")
        used_elo = pd.concat([out["home_elo_rating_date"], out["away_elo_rating_date"]]).dropna()
        used_fifa = pd.concat([out["home_fifa_ranking_date"], out["away_fifa_ranking_date"]]).dropna()
        if len(used_elo):
            f.write(f"- Elo rating dates used: {used_elo.min().date()} → {used_elo.max().date()}\n")
        if len(used_fifa):
            f.write(f"- FIFA ranking dates used: {used_fifa.min().date()} → {used_fifa.max().date()}\n\n")

        f.write("## Examples proving rating_date < match_date (no leakage)\n\n")
        f.write("| match_date | home | home_elo_date | away | away_elo_date | strictly_before |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in examples.iterrows():
            ok = (r["home_elo_rating_date"] < r["date"]) and (r["away_elo_rating_date"] < r["date"])
            f.write(f"| {r['date'].date()} | {r['home_team']} | {pd.Timestamp(r['home_elo_rating_date']).date()} | "
                    f"{r['away_team']} | {pd.Timestamp(r['away_elo_rating_date']).date()} | {ok} |\n")
        f.write("\n")

        f.write("## Rating momentum features\n\n")
        f.write("Momentum is computed as current pre-match rating minus latest rating strictly before the shifted cutoff date.\n\n")
        f.write("- FIFA momentum cutoff: match date minus 12 months.\n")
        f.write("- Elo momentum cutoffs: match date minus 6, 12, and 24 months.\n")
        f.write("- Prior dates are validated as strictly before the shifted cutoff, not merely before the match.\n\n")
        f.write("Features added: " + ", ".join(f"`{feature}`" for feature in MOMENTUM_FEATURES) + "\n\n")
        f.write("| Feature | Coverage 2000+ |\n|---|--:|\n")
        m2000 = out[out["date"] >= "2000-01-01"]
        for feature in MOMENTUM_FEATURES:
            f.write(f"| `{feature}` | {m2000[feature].notna().mean()*100:.1f}% |\n")
        f.write("\n")

        f.write("## Scoring-objective note\n\n")
        f.write("- These ratings joins are **future predictive features** (not yet used in any model).\n")
        f.write("- The final model must **optimise expected FIF8A points** under "
                "`Rules of the game/RULES_AND_SCORING.md` / `data/reference/scoring_rules.yml`.\n")
        f.write("- ⚠️ **No model has been trained in this phase.** No rolling-form, squad, coach, "
                "or country-context features were added; no tournament simulation was run.\n")

    logger.info(f"  ✓ {REPORT}")


if __name__ == "__main__":
    main()
