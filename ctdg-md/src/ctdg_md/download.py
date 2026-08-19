from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/chao1224/NeuralMD/resolve/main/MISATO_1000/raw"
FILES = {
    "MD.hdf5": {
        "url": f"{BASE}/MD.hdf5?download=true",
        "size": 7_455_614_516,
        "sha256": "b202e1c3f596b1deb091769ea16b617c166891bc397a37f41935a0017deb8b4c",
    },
    "train_MD.txt": {"url": f"{BASE}/train_MD.txt?download=true", "size": 4_000},
    "val_MD.txt": {"url": f"{BASE}/val_MD.txt?download=true", "size": 500},
    "test_MD.txt": {"url": f"{BASE}/test_MD.txt?download=true", "size": 500},
}


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ctdg-md/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        downloaded = offset
        with partial.open(mode) as handle:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                percent = 100.0 * downloaded / expected_size
                print(
                    f"\r{destination.name}: {downloaded / 2**30:.2f}/{expected_size / 2**30:.2f} GiB "
                    f"({percent:5.1f}%)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    print(file=sys.stderr)
    actual = partial.stat().st_size
    if actual != expected_size:
        raise OSError(f"Incomplete {destination.name}: expected {expected_size} bytes, received {actual}")
    partial.replace(destination)


def download_misato_1000(destination: str | Path, verify_hash: bool = True) -> Path:
    """Download the published real MISATO_1000 subset and official split files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, metadata in FILES.items():
        target = destination / name
        expected_size = int(metadata["size"])
        if not target.is_file() or target.stat().st_size != expected_size:
            download_file(str(metadata["url"]), target, expected_size)
        else:
            print(f"Reusing complete {target}", file=sys.stderr)
        expected_hash = metadata.get("sha256")
        if verify_hash and expected_hash:
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                target.unlink(missing_ok=True)
                raise OSError(
                    f"SHA-256 mismatch for {target}: expected {expected_hash}, received {actual_hash}"
                )
    provenance = {
        "dataset": "MISATO_1000 derived subset released with NeuralMD",
        "source": "https://huggingface.co/datasets/chao1224/NeuralMD/tree/main/MISATO_1000/raw",
        "upstream": "MISATO, DOI 10.1038/s43588-024-00627-2; Zenodo 7711953",
        "training_contract": "first 80 train + 10 validation + 10 test complexes, 100 frames each",
        "real_frames": 10_000,
        "synthetic_frames": 0,
        "hdf5_sha256": FILES["MD.hdf5"]["sha256"],
    }
    (destination / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return destination / "MD.hdf5"
