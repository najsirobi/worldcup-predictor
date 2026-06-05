"""Tests for immutable frozen-submission guardrails."""

from pathlib import Path

import pytest

from src.live.submission_guard import (
    FROZEN_SUBMISSION_FILES,
    FrozenSubmissionGuard,
    FrozenSubmissionIntegrityError,
    assert_not_frozen_path,
    verify_manifest,
    write_manifest,
)


def test_real_frozen_manifest_check_passes():
    result = verify_manifest()

    assert len(result.checked_files) == 4
    for filename in FROZEN_SUBMISSION_FILES:
        assert f"outputs/final_candidate_v2_auto_science/{filename}" in result.checked_files


def test_guard_catches_write_inside_frozen_candidate(tmp_path: Path):
    root = tmp_path
    candidate_dir = root / "outputs" / "final_candidate_v2_auto_science"
    legacy_dir = root / "outputs" / "final_candidate_v1"
    candidate_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    for filename in FROZEN_SUBMISSION_FILES:
        (candidate_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    manifest_path = candidate_dir / "FROZEN_MANIFEST.json"
    write_manifest(manifest_path, root=root)

    with pytest.raises(FrozenSubmissionIntegrityError, match="Added frozen path"):
        with FrozenSubmissionGuard(
            label="test live script",
            manifest_path=manifest_path,
            frozen_dirs=(candidate_dir, legacy_dir),
            root=root,
        ):
            (candidate_dir / "accidental_live_write.csv").write_text("bad\n", encoding="utf-8")


def test_assert_not_frozen_path_blocks_live_script_write_target():
    with pytest.raises(FrozenSubmissionIntegrityError, match="Refusing Travel Mode write"):
        assert_not_frozen_path(
            "outputs/final_candidate_v2_auto_science/final_group_score_predictions_auto.csv"
        )
