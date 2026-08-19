.PHONY: install test audit download verify train ablate publication serve

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

audit:
	python scripts/verify_implementation.py --skip-data

download:
	ctdg-md download --destination data/MISATO_1000/raw

verify:
	ctdg-md verify --config configs/default.yaml

train:
	ctdg-md train --config configs/default.yaml --variant ctdg_tvn

ablate:
	ctdg-md ablate --config configs/default.yaml

publication:
	ctdg-md publication --config configs/default.yaml

serve:
	CTDG_CHECKPOINT=outputs/checkpoints/ctdg_tvn_seed42.pt ctdg-md serve
