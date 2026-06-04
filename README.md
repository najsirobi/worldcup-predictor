# FIFA World Cup 2026 Prediction Model

A comprehensive data foundation and predictive modeling project for FIFA World Cup 2026 match and tournament probabilities.

## Project Overview

This project combines international football match history, team strength ratings, player/squad data, and country-context features to build predictive models for the 2026 FIFA World Cup.

**Status**: Data foundation phase (datasets downloaded, inspected, validated; ready for feature engineering).

## Phase 1: Data Foundation (Current)

- ✅ Download public Kaggle datasets
- ✅ Fetch/cache official FIFA 2026 pages
- ✅ Fetch World Bank country context indicators
- ✅ Inspect and validate data schemas
- ✅ Create explicit team name and country code mappings
- ✅ Build ingest modules and loaders
- ✅ Create country-context feature skeleton
- ⏳ *Phase 2 (next): Feature engineering and data integration*
- ⏳ *Phase 3: Model training (Elo baseline, logistic regression, goals model, gradient boosting)*
- ⏳ *Phase 4: Tournament simulation*

## Project Structure

```
worldcup-predictor/
├── data/                           # All data directories
│   ├── raw/                        # Raw downloaded data (git-ignored)
│   │   ├── kaggle/                 # Kaggle datasets
│   │   ├── fifa_official/          # Official FIFA 2026 cached pages
│   │   ├── world_bank/             # World Bank API responses
│   │   └── manual/                 # Manual import templates
│   ├── interim/                    # Processed intermediate data
│   ├── processed/                  # Model-ready features (phase 2+)
│   ├── reference/                  # Team name and country code mappings
│   └── MANIFEST.yml                # Dataset manifest (source of truth)
├── scripts/                        # Data download and processing scripts
│   ├── download_kaggle_data.py     # Kaggle dataset downloader
│   ├── fetch_fifa_2026.py          # FIFA 2026 official data fetcher
│   ├── fetch_world_bank_country_context.py  # World Bank API client
│   └── inspect_data.py             # Data inventory and inspection
├── src/                            # Python source code
│   ├── ingest/                     # Data loading modules
│   │   ├── common.py               # Shared utilities
│   │   ├── team_names.py           # Team name mapping
│   │   ├── country_codes.py        # Country code mapping
│   │   ├── matches.py              # International match loader
│   │   ├── ratings.py              # Elo and FIFA ranking loader
│   │   ├── world_cup.py            # World Cup data loader
│   │   ├── players.py              # Player data loader
│   │   ├── coaches.py              # Coach/manager data loader
│   │   └── country_context.py      # Country context loader
│   ├── features/                   # Feature engineering (skeleton)
│   │   └── country_context_features.py  # Country context features
│   ├── models/                     # Model modules (placeholder)
│   ├── simulation/                 # Tournament simulation (placeholder)
│   └── evaluation/                 # Model evaluation (placeholder)
├── tests/                          # Unit tests with fixture data
├── notebooks/                      # Jupyter notebooks (exploration)
├── outputs/                        # Generated reports and predictions
│   ├── reports/                    # Data and model reports
│   ├── predictions/                # Prediction outputs
│   └── charts/                     # Visualizations
├── AGENTS.md                       # Strict operational rules
├── README.md                       # This file
├── pyproject.toml                  # Python project configuration (incl. pytest config)
├── .gitignore                      # Git ignore patterns
└── .env.example                    # Environment variables template

```

## Quick Start

### 1. Clone and Setup

```bash
cd worldcup-predictor
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Configure Kaggle Credentials

The project downloads datasets from Kaggle. Set up authentication:

**Option A: Kaggle API Credentials File (Recommended)**

```bash
# Download from https://www.kaggle.com/account
# Place file at ~/.kaggle/kaggle.json
# Permissions: chmod 600 ~/.kaggle/kaggle.json
```

**Option B: Environment Variables**

```bash
cp .env.example .env
# Edit .env and add your Kaggle credentials:
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key
```

### 3. Download and Fetch Data

```bash
# Download Kaggle datasets
python scripts/download_kaggle_data.py

# Fetch official FIFA 2026 pages and create manual templates
python scripts/fetch_fifa_2026.py

# Fetch World Bank country context indicators
python scripts/fetch_world_bank_country_context.py
```

### 4. Inspect and Validate Data

```bash
# Generate data inventory report
python scripts/inspect_data.py

# Run tests on ingest modules
pytest tests/ -v
```

### 5. Review Reports

Generated reports are in `outputs/reports/`:

- `kaggle_download_report.md` — Kaggle dataset download status
- `fifa_2026_fetch_log.json` — Official FIFA page fetch log
- `fifa_2026_readiness_summary.md` — FIFA 2026 data readiness
- `world_bank_country_context_report.md` — World Bank data report
- `data_inventory.md` — Complete data inventory

## Datasets

### Kaggle Datasets

All Kaggle datasets are specified in `data/MANIFEST.yml` and downloaded automatically:

1. **international_results** — International match results 1872-2017
2. **international_elo** — Historical Elo ratings
3. **fifa_world_ranking** — Historical FIFA rankings
4. **world_cup_history** — Historical World Cup matches
5. **world_cup_database** — Rich WC entities (managers, players, goals, cards)
6. **transfermarkt_player_scores** — Player performance and market value
7. **fifa23_players_clean** — FIFA 23 player attributes (video-game proxy)
8. **world_cup_2022_player_data** — 2022 World Cup squad benchmarks

### Official FIFA 2026 Data

- Fixtures/scores (fetched and cached from fifa.com)
- Squads (manual template provided)
- Teams and coaches

**Note**: Parsing official FIFA pages is complex and requires manual inspection. Raw HTML is cached; manual CSV templates are created for data entry.

### World Bank Country Context

Macro indicators for all countries (2000-2023):

- Population
- GDP (PPP and nominal)
- Urbanization rate
- Education expenditure
- R&D expenditure

### Manual Import Templates

The following templates are created and can be filled with official data:

- `data/raw/manual/google_trends_football_interest_manual.csv` — Football search interest by country
- `data/raw/manual/fifa_professional_football_landscape_manual.csv` — Professional players/clubs per capita
- `data/raw/manual/fifa_forward_funding_manual.csv` — FIFA development funding

## Data Integrity Rules (AGENTS.md)

The project enforces strict data governance:

- **Raw data is read-only** — Never edit `data/raw/` manually.
- **All transformations are reproducible** — Via scripts, not manual steps.
- **No silent normalization** — Team names and country codes use explicit mappings.
- **No future leakage** — Features use only information available before the match.
- **Explicit mappings** — See `data/reference/team_name_map.csv` and `country_code_map.csv`.

## Data Readiness

After running the data scripts, check:

1. **Kaggle datasets**: `outputs/reports/kaggle_download_report.md`
2. **Official FIFA data**: `outputs/reports/fifa_2026_readiness_summary.md`
3. **World Bank data**: `outputs/reports/world_bank_country_context_report.md`
4. **Data inventory**: `outputs/reports/data_inventory.md`

## Feature Engineering (Phase 2)

Country context features are scaffolded in `src/features/country_context_features.py`:

- `log_population`
- `log_gdp_per_capita_ppp`
- `log_total_gdp_ppp`
- `urbanization_rate`
- `education_expenditure_share`
- `rd_expenditure_share`
- `pro_players_per_million_population`
- `pro_clubs_per_million_population`
- `registered_players_per_million_population`
- `football_culture_index` (Google Trends)
- `football_investment_capacity_index` (FIFA Forward funding)
- `support_infrastructure_capacity_index`

These will be integrated into match prediction models in Phase 2.

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_team_names.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

Tests use synthetic fixture data, not raw datasets, for fast and reliable testing.

## Important Notes

### Kaggle Authentication

If a Kaggle dataset requires consent (license agreement), the download will fail gracefully:

- Check the error message in `outputs/reports/kaggle_download_report.md`
- Visit the dataset page on Kaggle and accept the license
- Re-run `python scripts/download_kaggle_data.py`

### Official FIFA 2026 Data

Official FIFA pages are fetched and cached as raw HTML. Reliable parsing is complex:

- Raw HTML is saved to `data/raw/fifa_official/`
- Manual CSV templates are created in `data/raw/manual/`
- **Fill these templates with official 2026 data** from FIFA.com

### World Bank API

The World Bank API is public and requires no authentication. Data is fetched for all available countries and years (2000-2023).

## Next Steps (Phase 2)

1. **Fill manual templates** with official 2026 FIFA data
2. **Build match feature matrix** (rolling form, head-to-head, rating gaps)
3. **Build player-based features** (squad value, player count by position)
4. **Integrate country context features** into match models
5. **Handle missing data** and temporal boundaries
6. **Train baseline models** (Elo, logistic regression, Poisson goals model)
7. **Validate on historical World Cups** (2010, 2014, 2018, 2022)
8. **Tournament simulation** for 2026 predictions

## Development

Code style: `black`, `ruff`

```bash
black src/ tests/
ruff check src/ tests/
```

Type checking: `mypy`

```bash
mypy src/ --ignore-missing-imports
```

## License

[Specify license if applicable]

## Questions?

Refer to `AGENTS.md` for operational rules and `data/MANIFEST.yml` for dataset definitions.
