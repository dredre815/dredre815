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
    gate = a.out / "rdagent_gate.py"
    text = gate.read_text(encoding="utf-8")
    marker = "import os\n"
    replacement = 'import os\nos.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")\n'
    if replacement not in text:
        if marker not in text:
            raise RuntimeError("rdagent_gate.py import marker not found")
        text = text.replace(marker, replacement, 1)
    direct_marker = 'def direct_qlib_ic(conda_env: str, pred_path: Path, label_path: Path, work_dir: Path) -> dict:\n    script = work_dir / "direct_ic_check.py"'
    direct_replacement = 'def direct_qlib_ic(conda_env: str, pred_path: Path, label_path: Path, work_dir: Path) -> dict:\n    work_dir.mkdir(parents=True, exist_ok=True)\n    script = work_dir / "direct_ic_check.py"'
    if direct_replacement not in text:
        if direct_marker not in text:
            raise RuntimeError("direct_qlib_ic marker not found")
        text = text.replace(direct_marker, direct_replacement, 1)
    class_marker = 'from rdagent.scenarios.qlib.experiment.quant_experiment import QlibFactorExperiment, QlibModelExperiment'
    class_replacement = 'from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment\n    from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment'
    if class_replacement not in text:
        if class_marker not in text:
            raise RuntimeError("concrete experiment class import marker not found")
        text = text.replace(class_marker, class_replacement, 1)
    # RDAgentSettings has no RD_AGENT_ prefix; add the actual Pydantic field name
    # so local test-workspace objects are not serialized after successful runs.
    cache_env_marker = '"RD_AGENT_CACHE_WITH_PICKLE": "false",'
    cache_env_replacement = '"RD_AGENT_CACHE_WITH_PICKLE": "false",\n            "CACHE_WITH_PICKLE": "false",'
    if cache_env_replacement not in text:
        if cache_env_marker not in text:
            raise RuntimeError("cache environment marker not found")
        text = text.replace(cache_env_marker, cache_env_replacement, 1)
    gate.write_text(text, encoding="utf-8")
    print(f"extracted harness to {a.out}; sha256={actual}; compatibility_hotfixes=mlflow,direct_ic_dir,concrete_experiment_classes,cache_env")
if __name__ == "__main__":
    main()
