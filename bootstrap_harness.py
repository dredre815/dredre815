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
    # Compatibility hotfix for MLflow >=3.5: the frozen Qlib target still uses the
    # legacy local file tracking backend. This changes storage plumbing only; it
    # does not alter factors, labels, predictions, metrics, or the attack logic.
    gate = a.out / "rdagent_gate.py"
    text = gate.read_text(encoding="utf-8")
    marker = "import os\n"
    replacement = 'import os\nos.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")\n'
    if replacement not in text:
        if marker not in text:
            raise RuntimeError("rdagent_gate.py import marker not found")
        gate.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    print(f"extracted harness to {a.out}; sha256={actual}; mlflow_file_store_hotfix=true")
if __name__ == "__main__":
    main()
