#!/usr/bin/env python3
"""Materialise the DelegationLeak experiment harness from deterministic text chunks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import tarfile
from pathlib import Path

EXPECTED_SHA256 = "bc1fe6f2036be7a5b1fb76267b77866eccf0674edd214c267c177126a9020406"


def safe_members(tf: tarfile.TarFile, destination: Path):
    root = destination.resolve()
    for member in tf.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"refusing archive path outside destination: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"refusing archive link: {member.name}")
        yield member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, default=Path("delegationleak/parts"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parts = sorted(args.parts.glob("part-*.b85"))
    if not parts:
        raise RuntimeError(f"no archive parts under {args.parts}")

    chunks: list[str] = []
    inventory = []
    for part in parts:
        raw = part.read_bytes()
        inventory.append(
            {
                "name": part.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        chunks.append(raw.decode("ascii").strip())

    encoded = "".join(chunks).encode("ascii")
    archive = base64.b85decode(encoded)
    observed = hashlib.sha256(archive).hexdigest()
    print({"parts": inventory, "encoded_bytes": len(encoded), "archive_bytes": len(archive), "sha256": observed})
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"archive hash mismatch: expected {EXPECTED_SHA256}, observed {observed}")

    args.out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tf:
        tf.extractall(args.out, members=safe_members(tf, args.out))

    runtime = args.out / "runtime"
    required = {
        "common.py",
        "prepare_sample.py",
        "langgraph_adapter.py",
        "autogen_adapter.py",
        "crewai_adapter.py",
        "tradingagents_adapter.py",
        "rdagentq_adapter.py",
        "market_sweep.py",
        "defense_and_scaling.py",
        "llm_audit.py",
        "forkleak_extension.py",
        "build_report.py",
    }
    observed_files = {p.name for p in runtime.glob("*.py")}
    missing = required - observed_files
    if missing:
        raise RuntimeError(f"materialised runtime is incomplete: {sorted(missing)}")
    print({"runtime": str(runtime), "python_files": sorted(observed_files)})


if __name__ == "__main__":
    main()
