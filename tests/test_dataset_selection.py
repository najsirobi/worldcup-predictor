"""Test deterministic dataset CSV selection (guards against glob-order bugs)."""
import pytest

from src.ingest.common import select_dataset_csv


def _make_csv(dirpath, name):
    f = dirpath / name
    f.write_text("a,b\n1,2\n")
    return f


def test_prefers_named_file_over_glob_order(tmp_path):
    """Preferred file is selected regardless of filesystem ordering."""
    _make_csv(tmp_path, "goalscorers.csv")
    _make_csv(tmp_path, "results.csv")
    _make_csv(tmp_path, "shootouts.csv")

    chosen = select_dataset_csv(tmp_path, preferred_names=["results.csv"])
    assert chosen.name == "results.csv"


def test_pick_last_selects_latest_sorted(tmp_path):
    """pick='last' returns the latest sorted snapshot (e.g. newest ranking)."""
    _make_csv(tmp_path, "fifa_ranking-2023-07-20.csv")
    _make_csv(tmp_path, "fifa_ranking-2024-06-20.csv")
    _make_csv(tmp_path, "fifa_ranking-2024-04-04.csv")

    chosen = select_dataset_csv(tmp_path, pick="last")
    assert chosen.name == "fifa_ranking-2024-06-20.csv"


def test_single_file_returned(tmp_path):
    _make_csv(tmp_path, "eloratings.csv")
    chosen = select_dataset_csv(tmp_path, preferred_names=["eloratings.csv"])
    assert chosen.name == "eloratings.csv"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        select_dataset_csv(tmp_path / "does_not_exist")


def test_no_csv_files_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("not a csv")
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        select_dataset_csv(tmp_path)


def test_preferred_not_present_falls_back_deterministically(tmp_path):
    _make_csv(tmp_path, "b.csv")
    _make_csv(tmp_path, "a.csv")
    # Preferred file absent -> deterministic sorted fallback (first).
    chosen = select_dataset_csv(tmp_path, preferred_names=["results.csv"])
    assert chosen.name == "a.csv"
