from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=DEFAULT_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
