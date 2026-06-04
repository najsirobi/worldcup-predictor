#!/usr/bin/env python3
"""Inspect and report on all data files."""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    return filepath.stat().st_size / (1024 * 1024)


def inspect_csv(filepath: Path) -> dict:
    """Inspect a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return {
            "format": "CSV",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": (df.duplicated().sum()),
            "date_columns": [col for col in df.columns if "date" in col.lower()],
            "error": None,
        }
    except Exception as e:
        return {"format": "CSV", "error": str(e)}


def inspect_parquet(filepath: Path) -> dict:
    """Inspect a Parquet file."""
    try:
        df = pd.read_parquet(filepath)
        return {
            "format": "Parquet",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": (df.duplicated().sum()),
            "date_columns": [col for col in df.columns if "date" in col.lower()],
            "error": None,
        }
    except Exception as e:
        return {"format": "Parquet", "error": str(e)}


def inspect_excel(filepath: Path) -> dict:
    """Inspect an Excel file."""
    try:
        xls = pd.ExcelFile(filepath)
        sheets = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet)
            sheets[sheet] = {
                "rows": len(df),
                "columns": len(df.columns),
            }
        return {
            "format": "Excel",
            "sheets": sheets,
            "error": None,
        }
    except Exception as e:
        return {"format": "Excel", "error": str(e)}


def inspect_json(filepath: Path) -> dict:
    """Inspect a JSON file."""
    try:
        import json
        with open(filepath) as f:
            data = json.load(f)
        return {
            "format": "JSON",
            "type": type(data).__name__,
            "size": len(data) if isinstance(data, (list, dict)) else None,
            "error": None,
        }
    except Exception as e:
        return {"format": "JSON", "error": str(e)}


def scan_data_directory() -> dict:
    """Scan all data files."""
    results = {}

    for filepath in sorted(DATA_DIR.rglob("*")):
        if not filepath.is_file():
            continue

        # Skip certain directories
        if any(part in filepath.parts for part in [".git", "__pycache__", ".pytest_cache"]):
            continue

        rel_path = filepath.relative_to(DATA_DIR)

        # Inspect by format
        info = {
            "path": str(rel_path),
            "size_mb": get_file_size_mb(filepath),
        }

        if filepath.suffix == ".csv":
            info.update(inspect_csv(filepath))
        elif filepath.suffix == ".parquet":
            info.update(inspect_parquet(filepath))
        elif filepath.suffix in [".xlsx", ".xls"]:
            info.update(inspect_excel(filepath))
        elif filepath.suffix == ".json":
            info.update(inspect_json(filepath))
        else:
            info["format"] = filepath.suffix or "unknown"

        results[str(rel_path)] = info

    return results


def main():
    """Inspect all data and generate report."""
    logger.info("Scanning data directory...\n")

    results = scan_data_directory()

    if not results:
        logger.warning("No data files found")
        return

    # Generate report
    report_dir = REPO_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "data_inventory.md"

    with open(report_path, "w") as f:
        f.write("# Data Inventory Report\n\n")
        f.write(f"Total files: {len(results)}\n\n")

        # Group by directory
        by_dir = {}
        for filepath, info in results.items():
            dir_name = str(Path(filepath).parent)
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append((filepath, info))

        for dir_name in sorted(by_dir.keys()):
            f.write(f"## {dir_name}\n\n")

            for filepath, info in sorted(by_dir[dir_name]):
                f.write(f"### {Path(filepath).name}\n\n")
                f.write(f"- **Path**: `{filepath}`\n")
                f.write(f"- **Format**: {info.get('format', 'unknown')}\n")
                f.write(f"- **Size**: {info.get('size_mb', 0):.2f} MB\n")

                if info.get("error"):
                    f.write(f"- **Error**: {info['error']}\n\n")
                    continue

                # Tabular data
                if "rows" in info:
                    f.write(f"- **Rows**: {info['rows']}\n")
                    f.write(f"- **Columns**: {info['columns']}\n")

                    if info.get("column_names"):
                        f.write(f"- **Columns**: {', '.join(info['column_names'][:10])}")
                        if len(info['column_names']) > 10:
                            f.write(f" (+ {len(info['column_names']) - 10} more)")
                        f.write("\n")

                    if info.get("duplicates") is not None:
                        f.write(f"- **Duplicate rows**: {info['duplicates']}\n")

                    if info.get("missing_values"):
                        missing_summary = {k: v for k, v in info['missing_values'].items() if v > 0}
                        if missing_summary:
                            f.write(f"- **Missing values**: {missing_summary}\n")

                    if info.get("date_columns"):
                        f.write(f"- **Date columns**: {', '.join(info['date_columns'])}\n")

                # JSON
                elif "type" in info:
                    f.write(f"- **Type**: {info['type']}\n")
                    if info.get("size"):
                        f.write(f"- **Size**: {info['size']}\n")

                # Excel
                elif "sheets" in info:
                    f.write(f"- **Sheets**: {len(info['sheets'])}\n")
                    for sheet, sheet_info in info['sheets'].items():
                        f.write(f"  - {sheet}: {sheet_info['rows']} rows, {sheet_info['columns']} columns\n")

                f.write("\n")

    logger.info(f"✓ Report written to {report_path}")


if __name__ == "__main__":
    main()
