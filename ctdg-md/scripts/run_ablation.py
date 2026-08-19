#!/usr/bin/env python3
import argparse
import json

from ctdg_md.ablation import run_ablation
from ctdg_md.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
print(json.dumps(run_ablation(load_config(args.config)), indent=2))
