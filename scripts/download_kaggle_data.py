#!/usr/bin/env python3
"""Download Kaggle datasets specified in data/MANIFEST.yml."""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub not installed. Run: pip install kagglehub")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "MANIFEST.yml"
RAW_DIR = DATA_DIR / "raw" / "kaggle"


def load_manifest() -> dict:
    """Load dataset manifest."""
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def download_dataset(slug: str, target_dir: Path) -> bool:
    """Download a dataset using kagglehub."""
    try:
        logger.info(f"Downloading {slug}...")
        cache_path = kagglehub.dataset_download(slug)
        logger.info(f"  → cached at {cache_path}")

        # Copy from cache to target
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(cache_path).iterdir():
            if item.is_file():
                dest = target_dir / item.name
                shutil.copy2(item, dest)
                logger.info(f"  ✓ copied {item.name}")
            elif item.is_dir():
                dest = target_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                logger.info(f"  ✓ copied directory {item.name}")

        return True
    except Exception as e:
        logger.error(f"  ✗ FAILED: {e}")
        return False


def main():
    """Download all Kaggle datasets from manifest."""
    manifest = load_manifest()
    kaggle_datasets = manifest.get("kaggle_datasets", {})

    if not kaggle_datasets:
        logger.warning("No Kaggle datasets found in manifest")
        return

    logger.info(f"Found {len(kaggle_datasets)} Kaggle datasets to download\n")

    results = {}
    for dataset_key, dataset_info in kaggle_datasets.items():
        slug = dataset_info.get("slug")
        if not slug:
            logger.warning(f"  {dataset_key}: no slug defined, skipping")
            continue

        target_dir = RAW_DIR / dataset_key
        success = download_dataset(slug, target_dir)
        results[dataset_key] = {
            "slug": slug,
            "success": success,
            "target_dir": str(target_dir)
        }
        logger.info("")

    # Generate report
    report_dir = REPO_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "kaggle_download_report.md"

    with open(report_path, "w") as f:
        f.write("# Kaggle Dataset Download Report\n\n")

        successful = [k for k, v in results.items() if v["success"]]
        failed = [k for k, v in results.items() if not v["success"]]

        f.write(f"## Summary\n\n")
        f.write(f"- Total datasets: {len(results)}\n")
        f.write(f"- Successful: {len(successful)}\n")
        f.write(f"- Failed: {len(failed)}\n\n")

        if successful:
            f.write(f"## Successfully Downloaded\n\n")
            for dataset_key in successful:
                v = results[dataset_key]
                f.write(f"- **{dataset_key}** ({v['slug']})\n")
                f.write(f"  - Location: `{v['target_dir']}`\n\n")

        if failed:
            f.write(f"## Failed Downloads\n\n")
            for dataset_key in failed:
                v = results[dataset_key]
                f.write(f"- **{dataset_key}** ({v['slug']})\n")
                f.write(f"  - Check Kaggle credentials or dataset availability\n\n")

    logger.info(f"\n✓ Report written to {report_path}")

    if failed:
        logger.warning(f"\n{len(failed)} dataset(s) failed to download")
        exit(1)


if __name__ == "__main__":
    main()
