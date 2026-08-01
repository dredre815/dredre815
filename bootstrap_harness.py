#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib, io, lzma, tarfile
from pathlib import Path
ARCHIVE_SHA256 = "7fb92d34055b2f42ed8b94795eab92a5167449192f82dfb685a60b70ab04bf10"
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parts = Path(__file__).with_name("bootstrap_parts")
    encoded = "".join(x.read_text().strip() for x in sorted(parts.glob("payload-*.b85")))
    raw = base64.b85decode(encoded.encode())
    actual = hashlib.sha256(raw).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive hash mismatch: {actual}")
    a.out.mkdir(parents=True, exist_ok=True)
    data = lzma.decompress(raw)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        for member in tf.getmembers():
            target = (a.out / member.name).resolve()
            if a.out.resolve() not in target.parents and target != a.out.resolve():
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(a.out, filter="data")
    print(f"extracted harness to {a.out}; sha256={actual}")
if __name__ == "__main__":
    main()
