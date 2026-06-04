# Agent Rules for FIFA World Cup 2026 Prediction Project

## Data Integrity

- **Raw data must never be edited manually.** All raw data is placed under `data/raw/` and treated as read-only.
- **All transformations must be reproducible through scripts.** Manual edits destroy reproducibility.
- **Do not hardcode large datasets in source files.** Datasets must be in `data/raw/` or downloaded programmatically.
- **Use `data/MANIFEST.yml` as the source of truth for all datasets.** It documents what should exist, where it came from, and how it should be used.

## Team Names and Country Codes

- **Do not silently normalize team names.** Implicit normalization masks data quality issues.
- **Use explicit mapping in `data/reference/team_name_map.csv`.** Every team-name variant must be explicitly mapped to a canonical form.
- **Similarly for country codes:** use `data/reference/country_code_map.csv` for any country-name variant.
- **Mapping files are part of the source code review.** Changes to mappings must be explicit and justified.

## Feature Leakage

- **No future leakage.** Every feature must be computed only from information available before the match being predicted.
- **For team strength ratings:** use the latest available rating **strictly before** the match date.
- **For player/squad data:** use squads available **strictly before** or **at match time** (announced squads, not future roster changes).
- **Document any data-availability boundaries.** If a dataset has only current values, flag it as `current_only: true` and do not use it for historical backtesting unless explicitly allowed.

## Model Training

- **Do not train models in data-setup phase.** This task stops after data inspection and validation.
- **No feature engineering beyond infrastructure setup.** Feature logic is implemented but not applied to data.
- **No tournament simulation in this phase.** Simulation is a separate downstream task.

## Validation and Testing

- **Use clear validation errors.** Errors must explain what failed, why it failed, and what to fix.
- **Add tests using tiny fixture data, not raw Kaggle files.** Fixture data is synthetic or minimal real samples.
- **Tests validate schema, type, and constraint assumptions.** They do not require running the entire data-ingestion pipeline.

## Official Data Handling

- **If official FIFA pages cannot be fetched or parsed reliably, do not invent data.**
  - Cache raw HTML/JSON responses.
  - Create manual CSV templates with expected columns and example rows.
  - Document in `outputs/reports/data_readiness_summary.md` what still needs human input.

## Kaggle and Third-Party Datasets

- **If a dataset requires Kaggle authentication/consent, document it clearly.**
  - Log the error and reason.
  - Continue with other datasets.
  - Flag in the data-readiness report that this dataset is gated.
- **All Kaggle downloads use `kagglehub` library.**
- **Downloaded files are copied from cache into `data/raw/kaggle/<dataset_key>/`.**

## Reproducibility Checklist

- [ ] `data/raw/` contains only downloaded or explicitly-placed files.
- [ ] `data/reference/` contains all manual mappings (team names, country codes).
- [ ] `data/interim/` contains only script-generated files (can be regenerated).
- [ ] `data/processed/` reserved for model-ready features (not populated in this phase).
- [ ] All scripts are idempotent where possible (re-running produces same output).
- [ ] `MANIFEST.yml` is kept up-to-date.
- [ ] `outputs/reports/` contains inspection and readiness reports.
- [ ] Tests pass with fixture data.

## Scope Boundaries

- **In scope for this phase:**
  - Download public datasets (Kaggle, World Bank).
  - Cache official FIFA 2026 pages.
  - Inspect and validate schemas.
  - Create ingest/loading modules.
  - Generate data-readiness reports.
  - Create reference mappings and manual templates.

- **Out of scope:**
  - Building features beyond schema/constraint scaffolding.
  - Training any predictive models.
  - Tournament simulation.
  - Causal analysis (2SLS saved for separate notebook later).
