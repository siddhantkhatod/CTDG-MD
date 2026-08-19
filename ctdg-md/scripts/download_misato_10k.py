#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ctdg_md.download import download_misato_1000

parser = argparse.ArgumentParser(description="Download real MISATO_1000 data used by MISATO-10k")
parser.add_argument("--destination", default="data/MISATO_1000/raw")
parser.add_argument("--no-hash", action="store_true")
args = parser.parse_args()
print(download_misato_1000(args.destination, verify_hash=not args.no_hash))
