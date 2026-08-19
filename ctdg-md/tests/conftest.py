import os

# Importing the API in tests must never silently use an arbitrary user checkpoint.
os.environ.pop("CTDG_CHECKPOINT", None)
os.environ.pop("CTDG_ALLOW_UNTRAINED", None)
