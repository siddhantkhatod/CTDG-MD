from __future__ import annotations

import argparse
import json

from .ablation import run_ablation
from .config import load_config
from .data import verify_misato_10k
from .download import download_misato_1000
from .external import (
    evaluate_external_trajectory,
    export_same_target_extxyz,
    import_external_energy_csv,
)
from .publication import run_publication_study
from .train import train_one


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ctdg-md", description="CTDG-MD training and deployment")
    sub = root.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download the real MISATO_1000 benchmark subset")
    download.add_argument("--destination", default="data/MISATO_1000/raw")
    download.add_argument("--no-hash", action="store_true", help="skip the 7.45 GB SHA-256 pass")
    verify = sub.add_parser("verify", help="validate the exact 10,000-frame benchmark contract")
    verify.add_argument("--config", default="configs/default.yaml")
    verify.add_argument("--report", default="outputs/data_verification.json")
    train = sub.add_parser("train", help="train one model variant")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument(
        "--variant",
        choices=["constant", "schnet", "painn", "egnn", "egnn_tvn", "ctdg", "ctdg_tvn"],
    )
    train.add_argument("--seed", type=int)
    ablate = sub.add_parser("ablate", help="run all EGNN/CTDG/TVN ablations")
    ablate.add_argument("--config", default="configs/default.yaml")
    publication = sub.add_parser("publication", help="run claim-gated five-seed study")
    publication.add_argument("--config", default="configs/default.yaml")
    publication.add_argument("--aggregate-only", action="store_true")
    export = sub.add_parser("export-external", help="export aligned MACE/NequIP extxyz data")
    export.add_argument("--config", default="configs/default.yaml")
    export.add_argument("--output", required=True)
    external = sub.add_parser("import-external-energy", help="align external energy CSV")
    external.add_argument("--manifest", required=True)
    external.add_argument("--predictions", required=True)
    external.add_argument("--output", required=True)
    trajectory = sub.add_parser("evaluate-trajectory", help="evaluate NeuralMD/BioMD NPZ")
    trajectory.add_argument("--predictions", required=True)
    trajectory.add_argument("--output", required=True)
    serve = sub.add_parser("serve", help="serve the trained FastAPI project")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "download":
        path = download_misato_1000(args.destination, verify_hash=not args.no_hash)
        print(path)
    elif args.command == "verify":
        report = verify_misato_10k(load_config(args.config).data, args.report)
        print(json.dumps(report.__dict__, indent=2))
    elif args.command == "train":
        result, _ = train_one(load_config(args.config), variant=args.variant, seed=args.seed)
        print(json.dumps(result.__dict__, indent=2))
    elif args.command == "ablate":
        print(json.dumps(run_ablation(load_config(args.config)), indent=2))
    elif args.command == "publication":
        print(
            json.dumps(
                run_publication_study(load_config(args.config), train=not args.aggregate_only),
                indent=2,
            )
        )
    elif args.command == "export-external":
        print(json.dumps(export_same_target_extxyz(load_config(args.config).data, args.output), indent=2))
    elif args.command == "import-external-energy":
        print(import_external_energy_csv(args.manifest, args.predictions, args.output))
    elif args.command == "evaluate-trajectory":
        print(json.dumps(evaluate_external_trajectory(args.predictions, args.output), indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("ctdg_md.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
