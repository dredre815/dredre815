#!/usr/bin/env python3
"""Patch materialised experiments for Qlib panels encoded with index levels.

The frozen label artifact stores ``datetime`` and ``instrument`` as pandas
MultiIndex levels and ``label`` as the sole data column.  The experiment code
uses the neutral column names ``date``, ``instrument``, and ``label``.  This
small, versioned compatibility patch makes that conversion explicit and is
included in the workflow provenance hash.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OLD = '    df = pd.read_parquet(args.labels)[["date", "instrument", "label"]].dropna().copy()\n'
NEW = '''    df = pd.read_parquet(args.labels)\n    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:\n        df = df.reset_index()\n    if "datetime" in df.columns and "date" not in df.columns:\n        df = df.rename(columns={"datetime": "date"})\n    required = {"date", "instrument", "label"}\n    missing = required.difference(df.columns)\n    if missing:\n        raise RuntimeError(\n            f"label panel is missing {sorted(missing)}; columns={list(df.columns)}"\n        )\n    df = df[["date", "instrument", "label"]].dropna().copy()\n'''


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if OLD in text:
        text = text.replace(OLD, NEW, 1)
        path.write_text(text, encoding="utf-8")
        print({"patched": str(path)})
        return
    if 'required = {"date", "instrument", "label"}' in text:
        print({"already_patched": str(path)})
        return
    raise RuntimeError(f"expected Qlib read expression not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    for name in ("prepare_sample.py", "market_sweep.py"):
        patch(args.runtime / name)


if __name__ == "__main__":
    main()
