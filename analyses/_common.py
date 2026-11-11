"""Helpers shared by the analysis scripts."""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT))  # allow running scripts without installing the package


def write_csv(name: str, header, rows) -> Path:
    """Write rows to results/<name> and return the path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")
    return path


def read_csv(name: str):
    with open(RESULTS / name) as fh:
        return list(csv.DictReader(fh))


def parallel_map(func, items, workers: int | None = None):
    """Map func over items in parallel (order preserved)."""
    workers = workers or int(os.environ.get("EVOSWITCH_WORKERS", os.cpu_count() or 1))
    if workers <= 1:
        return [func(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items, chunksize=1))
