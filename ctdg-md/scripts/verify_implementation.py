#!/usr/bin/env python3
"""Fail-fast implementation audit intended for CI and pre-training use."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ctdg_md.config import load_config
from ctdg_md.data import verify_misato_10k

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--skip-data", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
checks = {}
checks["compile"] = subprocess.run(
    [sys.executable, "-m", "compileall", "-q", str(root / "src")], check=False
).returncode == 0
ruff = shutil.which("ruff")
checks["ruff"] = bool(ruff) and subprocess.run(
    [ruff, "check", "src", "tests", "scripts"],
    cwd=root,
    check=False,
).returncode == 0
checks["tests"] = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"], cwd=root, check=False
).returncode == 0
if not args.skip_data:
    report = verify_misato_10k(load_config(args.config).data)
    checks["real_data"] = report.__dict__
print(json.dumps(checks, indent=2))
if any(checks[name] is not True for name in ("compile", "ruff", "tests")):
    raise SystemExit(1)
