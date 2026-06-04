"""Resolve the active final-candidate prediction files (Travel Mode, Task A).

Travel Mode never hard-codes a candidate folder. Instead it reads
``data/live/active_candidate.yml`` to discover *which* frozen candidate is live
and the filenames of its four prediction artefacts. This lets the operator
switch from ``final_candidate_v1`` to ``final_candidate_v2_auto_science`` (or any
future candidate) by editing one small YAML file -- without retraining,
without touching the frozen candidate folders, and without code changes.

Behaviour:
    * If the config file is missing, fall back to ``final_candidate_v1`` with the
      default v1 filenames (documented below) so Travel Mode keeps working.
    * If the config points at files that do not exist, fail clearly with a
      message naming the offending path.

The frozen candidate folders are read-only as far as Travel Mode is concerned;
nothing here writes into them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "live" / "active_candidate.yml"

# Used verbatim when active_candidate.yml is absent.
DEFAULT_CONFIG = {
    "active_candidate_dir": "outputs/final_candidate_v1",
    "score_predictions_file": "final_group_score_predictions.csv",
    "standing_predictions_file": "final_group_standing_predictions.csv",
    "last8_predictions_file": "final_last8_predictions.csv",
    "submission_pack_file": "final_submission_pack.csv",
}

_REQUIRED_KEYS = (
    "active_candidate_dir",
    "score_predictions_file",
    "standing_predictions_file",
    "last8_predictions_file",
    "submission_pack_file",
)


@dataclass(frozen=True)
class ActiveCandidate:
    """Resolved, validated paths to the active candidate's prediction files."""

    name: str
    candidate_dir: Path
    score_predictions_path: Path
    standing_predictions_path: Path
    last8_predictions_path: Path
    submission_pack_path: Path
    config_path: Path
    config_existed: bool

    def load_score_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.score_predictions_path)

    def load_standing_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.standing_predictions_path)

    def load_last8_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.last8_predictions_path)

    def as_dict(self) -> dict:
        """JSON/report-friendly view (paths relative to the repo root)."""
        return {
            "name": self.name,
            "active_candidate_dir": _rel(self.candidate_dir),
            "score_predictions_file": self.score_predictions_path.name,
            "standing_predictions_file": self.standing_predictions_path.name,
            "last8_predictions_file": self.last8_predictions_path.name,
            "submission_pack_file": self.submission_pack_path.name,
            "config_path": _rel(self.config_path),
            "config_existed": self.config_existed,
        }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(config_path: Path = CONFIG_PATH) -> tuple[dict, bool]:
    """Return ``(config, existed)``; falls back to the v1 default if missing."""
    config_path = Path(config_path)
    if not config_path.exists():
        return dict(DEFAULT_CONFIG), False
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{config_path} must contain a YAML mapping of candidate settings."
        )
    missing = [k for k in _REQUIRED_KEYS if not raw.get(k)]
    if missing:
        raise ValueError(
            f"{config_path} is missing required key(s): {missing}. "
            f"Required keys: {list(_REQUIRED_KEYS)}."
        )
    return {k: raw[k] for k in _REQUIRED_KEYS}, True


def load_active_candidate(
    config_path: Path = CONFIG_PATH, root: Path = ROOT
) -> ActiveCandidate:
    """Resolve and validate the active candidate's prediction files.

    Raises ``FileNotFoundError`` (with the offending path) if the configured
    candidate directory or any configured prediction file does not exist.
    """
    config, existed = load_config(config_path)
    candidate_dir = (root / config["active_candidate_dir"]).resolve()
    if not candidate_dir.is_dir():
        raise FileNotFoundError(
            f"Active candidate directory not found: {config['active_candidate_dir']} "
            f"(resolved to {candidate_dir}). Check active_candidate_dir in "
            f"{_rel(Path(config_path))}."
        )

    paths = {}
    for key, attr in (
        ("score_predictions_file", "score"),
        ("standing_predictions_file", "standing"),
        ("last8_predictions_file", "last8"),
        ("submission_pack_file", "submission"),
    ):
        candidate_path = candidate_dir / config[key]
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Configured {key} not found: {config[key]} in "
                f"{config['active_candidate_dir']} (resolved to {candidate_path})."
            )
        paths[attr] = candidate_path

    return ActiveCandidate(
        name=candidate_dir.name,
        candidate_dir=candidate_dir,
        score_predictions_path=paths["score"],
        standing_predictions_path=paths["standing"],
        last8_predictions_path=paths["last8"],
        submission_pack_path=paths["submission"],
        config_path=Path(config_path),
        config_existed=existed,
    )
