#!/usr/bin/env python3
"""Copy paper-facing CSV tables into a clean reproduction directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reproduced")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "results/paper_tables").glob("*.csv"))
    for src in files:
        shutil.copy2(src, out / src.name)
    print(f"copied {len(files)} derived paper tables to {out}")


if __name__ == "__main__":
    main()
