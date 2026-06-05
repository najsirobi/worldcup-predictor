"""Immutable-submission guardrails for Travel Mode.

Travel Mode may consume frozen candidate files, but it may not rewrite submitted
predictions. This module provides two protections:

* manifest verification for the submitted v2 candidate files;
* before/after snapshots of frozen candidate directories around live scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import TracebackType
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

ACTIVE_FROZEN_CANDIDATE_DIR = ROOT / "outputs" / "final_candidate_v2_auto_science"
LEGACY_FROZEN_CANDIDATE_DIR = ROOT / "outputs" / "final_candidate_v1"
FROZEN_DIRS = (ACTIVE_FROZEN_CANDIDATE_DIR, LEGACY_FROZEN_CANDIDATE_DIR)
FROZEN_MANIFEST_PATH = ACTIVE_FROZEN_CANDIDATE_DIR / "FROZEN_MANIFEST.json"

FROZEN_SUBMISSION_FILES = (
    "final_group_score_predictions_auto.csv",
    "final_group_standing_predictions_auto.csv",
    "final_last8_predictions_auto.csv",
    "final_submission_pack_auto.csv",
)


class FrozenSubmissionIntegrityError(RuntimeError):
    """Raised when a frozen submission file or directory changes unexpectedly."""


@dataclass(frozen=True)
class ManifestCheckResult:
    """Summary of a successful frozen-submission manifest check."""

    manifest_path: Path
    checked_files: tuple[str, ...]


def _resolve(path: str | Path, root: Path = ROOT) -> Path:
    path = Path(path)
    return (root / path if not path.is_absolute() else path).resolve()


def _rel(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path = ROOT) -> dict:
    """Build the expected manifest for the active frozen v2 submission."""
    candidate_dir = root / "outputs" / "final_candidate_v2_auto_science"
    files = []
    for filename in FROZEN_SUBMISSION_FILES:
        path = candidate_dir / filename
        if not path.exists():
            raise FrozenSubmissionIntegrityError(
                f"Frozen submission file missing while building manifest: {_rel(path, root)}"
            )
        files.append(
            {
                "path": _rel(path, root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "candidate_dir": "outputs/final_candidate_v2_auto_science",
        "protected_dirs": [
            "outputs/final_candidate_v2_auto_science",
            "outputs/final_candidate_v1",
        ],
        "rule": (
            "Submitted predictions are immutable. Live actual results may update "
            "only outputs/live, docs dashboard exports, and Travel Mode reports."
        ),
        "files": files,
    }


def write_manifest(path: Path = FROZEN_MANIFEST_PATH, root: Path = ROOT) -> dict:
    """Write a manifest from current frozen files. Use only for explicit freeze events."""
    manifest = build_manifest(root=root)
    path = _resolve(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _manifest_entries(manifest: dict) -> dict[str, dict]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise FrozenSubmissionIntegrityError("Frozen manifest must contain a files list.")
    by_path = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path"):
            raise FrozenSubmissionIntegrityError("Frozen manifest contains a malformed file entry.")
        by_path[str(entry["path"])] = entry
    return by_path


def verify_manifest(
    path: Path = FROZEN_MANIFEST_PATH,
    root: Path = ROOT,
) -> ManifestCheckResult:
    """Verify active frozen submission files against ``FROZEN_MANIFEST.json``."""
    path = _resolve(path, root)
    if not path.exists():
        raise FrozenSubmissionIntegrityError(
            f"Frozen manifest not found: {_rel(path, root)}. "
            "Run scripts/check_frozen_submission_integrity.py --write-manifest only "
            "when explicitly freezing a submitted candidate."
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = _manifest_entries(manifest)
    expected_paths = [
        f"outputs/final_candidate_v2_auto_science/{filename}"
        for filename in FROZEN_SUBMISSION_FILES
    ]
    errors: list[str] = []
    for rel_path in expected_paths:
        entry = entries.get(rel_path)
        file_path = root / rel_path
        if entry is None:
            errors.append(f"Missing manifest entry: {rel_path}")
            continue
        if not file_path.exists():
            errors.append(f"Frozen file missing: {rel_path}")
            continue
        actual_size = file_path.stat().st_size
        actual_hash = sha256_file(file_path)
        if int(entry.get("size_bytes", -1)) != actual_size:
            errors.append(
                f"Size mismatch for {rel_path}: manifest={entry.get('size_bytes')} "
                f"actual={actual_size}"
            )
        if str(entry.get("sha256")) != actual_hash:
            errors.append(
                f"SHA-256 mismatch for {rel_path}: manifest={entry.get('sha256')} "
                f"actual={actual_hash}"
            )

    if errors:
        raise FrozenSubmissionIntegrityError(
            "Frozen submission integrity check failed:\n- " + "\n- ".join(errors)
        )
    return ManifestCheckResult(path, tuple(expected_paths))


def snapshot_frozen_dirs(
    frozen_dirs: Iterable[Path] = FROZEN_DIRS,
    root: Path = ROOT,
) -> dict[str, dict]:
    """Snapshot files and directories under the frozen candidate directories."""
    snapshot: dict[str, dict] = {}
    for frozen_dir in frozen_dirs:
        frozen_dir = _resolve(frozen_dir, root)
        if not frozen_dir.exists():
            snapshot[_rel(frozen_dir, root)] = {"type": "missing"}
            continue
        snapshot[_rel(frozen_dir, root)] = {"type": "dir"}
        for path in sorted(frozen_dir.rglob("*")):
            rel_path = _rel(path, root)
            if path.is_dir():
                snapshot[rel_path] = {"type": "dir"}
            elif path.is_file():
                snapshot[rel_path] = {
                    "type": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
    return snapshot


def compare_snapshots(before: dict[str, dict], after: dict[str, dict]) -> list[str]:
    """Return human-readable frozen-directory changes between two snapshots."""
    messages: list[str] = []
    before_keys = set(before)
    after_keys = set(after)
    for rel_path in sorted(after_keys - before_keys):
        messages.append(f"Added frozen path: {rel_path}")
    for rel_path in sorted(before_keys - after_keys):
        messages.append(f"Removed frozen path: {rel_path}")
    for rel_path in sorted(before_keys & after_keys):
        if before[rel_path] != after[rel_path]:
            messages.append(f"Modified frozen path: {rel_path}")
    return messages


def assert_no_frozen_changes(
    before: dict[str, dict],
    *,
    frozen_dirs: Iterable[Path] = FROZEN_DIRS,
    root: Path = ROOT,
) -> None:
    """Raise if the frozen candidate directories changed since ``before``."""
    after = snapshot_frozen_dirs(frozen_dirs=frozen_dirs, root=root)
    changes = compare_snapshots(before, after)
    if changes:
        raise FrozenSubmissionIntegrityError(
            "Travel Mode attempted to modify frozen submitted predictions. "
            "Actual results must update only live outputs, scoring, tables, bracket "
            "state, or future projections.\n- "
            + "\n- ".join(changes)
        )


def assert_not_frozen_path(
    path: str | Path,
    *,
    frozen_dirs: Iterable[Path] = FROZEN_DIRS,
    root: Path = ROOT,
) -> None:
    """Fail before a Travel Mode write targets a frozen candidate directory."""
    target = _resolve(path, root)
    for frozen_dir in frozen_dirs:
        frozen_dir = _resolve(frozen_dir, root)
        if target == frozen_dir or _is_relative_to(target, frozen_dir):
            raise FrozenSubmissionIntegrityError(
                "Refusing Travel Mode write inside frozen submission directory: "
                f"{_rel(target, root)}. Write live results under outputs/live, docs, "
                "or outputs/reports instead."
            )


class FrozenSubmissionGuard:
    """Context manager that verifies frozen submissions before and after live work."""

    def __init__(
        self,
        *,
        label: str = "Travel Mode",
        manifest_path: Path = FROZEN_MANIFEST_PATH,
        frozen_dirs: Iterable[Path] = FROZEN_DIRS,
        root: Path = ROOT,
    ):
        self.label = label
        self.manifest_path = manifest_path
        self.frozen_dirs = tuple(frozen_dirs)
        self.root = root
        self._before: dict[str, dict] | None = None

    def __enter__(self) -> "FrozenSubmissionGuard":
        verify_manifest(self.manifest_path, root=self.root)
        self._before = snapshot_frozen_dirs(self.frozen_dirs, root=self.root)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        assert self._before is not None
        try:
            verify_manifest(self.manifest_path, root=self.root)
            assert_no_frozen_changes(
                self._before,
                frozen_dirs=self.frozen_dirs,
                root=self.root,
            )
        except FrozenSubmissionIntegrityError as guard_error:
            message = f"{self.label} failed frozen-submission guard: {guard_error}"
            if exc is not None:
                raise FrozenSubmissionIntegrityError(message) from exc
            raise FrozenSubmissionIntegrityError(message)
        return False


def guard_frozen_submission(label: str = "Travel Mode") -> FrozenSubmissionGuard:
    """Return the standard guard for Travel Mode scripts."""
    return FrozenSubmissionGuard(label=label)
