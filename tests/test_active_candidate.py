"""Tests for the active-candidate pointer (Travel Mode, Task A)."""

from pathlib import Path

import pytest

from src.live.active_candidate import (
    DEFAULT_CONFIG,
    load_active_candidate,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "active_candidate.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_active_candidate_v1_loads():
    """The default repo config resolves final_candidate_v2_auto_science's four files."""
    candidate = load_active_candidate()
    assert candidate.name == "final_candidate_v2_auto_science"
    assert candidate.score_predictions_path.exists()
    assert candidate.standing_predictions_path.exists()
    assert candidate.last8_predictions_path.exists()
    # Loadable as frames.
    assert len(candidate.load_score_predictions()) == 72


def test_missing_config_falls_back_to_v1(tmp_path):
    """A missing config file defaults to v1 (does not raise)."""
    config, existed = load_config(tmp_path / "does_not_exist.yml")
    assert existed is False
    assert config == DEFAULT_CONFIG
    candidate = load_active_candidate(config_path=tmp_path / "does_not_exist.yml")
    assert candidate.name == "final_candidate_v1"
    assert candidate.config_existed is False


@pytest.mark.skipif(not V2_DIR.exists(), reason="auto-science candidate not present")
def test_active_candidate_v2_loads_if_present(tmp_path):
    cfg = _write_config(
        tmp_path,
        "active_candidate_dir: outputs/final_candidate_v2_auto_science\n"
        "score_predictions_file: final_group_score_predictions_auto.csv\n"
        "standing_predictions_file: final_group_standing_predictions_auto.csv\n"
        "last8_predictions_file: final_last8_predictions_auto.csv\n"
        "submission_pack_file: final_submission_pack_auto.csv\n",
    )
    candidate = load_active_candidate(config_path=cfg)
    assert candidate.name == "final_candidate_v2_auto_science"
    assert candidate.score_predictions_path.name == "final_group_score_predictions_auto.csv"
    assert len(candidate.load_score_predictions()) == 72


def test_missing_candidate_dir_fails_clearly(tmp_path):
    cfg = _write_config(
        tmp_path,
        "active_candidate_dir: outputs/final_candidate_does_not_exist\n"
        "score_predictions_file: x.csv\n"
        "standing_predictions_file: x.csv\n"
        "last8_predictions_file: x.csv\n"
        "submission_pack_file: x.csv\n",
    )
    with pytest.raises(FileNotFoundError, match="candidate directory not found"):
        load_active_candidate(config_path=cfg)


def test_missing_configured_file_fails_clearly(tmp_path):
    cfg = _write_config(
        tmp_path,
        "active_candidate_dir: outputs/final_candidate_v1\n"
        "score_predictions_file: not_a_real_file.csv\n"
        "standing_predictions_file: final_group_standing_predictions.csv\n"
        "last8_predictions_file: final_last8_predictions.csv\n"
        "submission_pack_file: final_submission_pack.csv\n",
    )
    with pytest.raises(FileNotFoundError, match="not_a_real_file.csv"):
        load_active_candidate(config_path=cfg)
