#!/usr/bin/env python3
"""One-command runtime smoke run; never substitutes for the real benchmark."""
from __future__ import annotations

import argparse

from ctdg_md.config import load_config
from ctdg_md.train import train_one

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument(
    "--variant",
    default="ctdg_tvn",
    choices=["constant", "schnet", "painn", "egnn", "egnn_tvn", "ctdg", "ctdg_tvn"],
)
args = parser.parse_args()
config = load_config(args.config)
config.training.epochs = 1
config.training.patience = 1
result, _ = train_one(config, variant=args.variant)
print(result)
