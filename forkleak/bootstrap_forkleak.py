#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib, io, lzma, tarfile
from pathlib import Path
ARCHIVE_SHA256 = "b2568593957bb0e132145ada0c8a288dda91c97a5e629f2423ab0af9a217d73d"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parts = Path(__file__).with_name("parts")
    encoded = "".join(x.read_text().strip() for x in sorted(parts.glob("part-*.b85")))
    raw = base64.b85decode(encoded.encode())
    actual = hashlib.sha256(raw).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive hash mismatch: {actual}")
    a.out.mkdir(parents=True, exist_ok=True)
    data = lzma.decompress(raw)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        root = a.out.resolve()
        for member in tf.getmembers():
            target = (a.out / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(a.out, filter="data")
    print(f"materialized ForkLeak harness at {a.out}; sha256={actual}")

if __name__ == "__main__":
    main()
