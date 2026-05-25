"""YAML configuration loading utilities."""

from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError("Config file must have a .yaml or .yml extension")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")

    return config
