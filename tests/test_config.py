import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def test_load_valid_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\nlearning_rate: 0.01\nnum_epochs: 10\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["seed"] == 42
    assert config["learning_rate"] == 0.01
    assert config["num_epochs"] == 10


def test_load_missing_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("missing_config.yaml")


def test_load_invalid_extension_raises_value_error(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(config_path))


def test_load_empty_yaml_returns_empty_dict(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_config(str(config_path)) == {}


def test_load_non_dict_yaml_raises_value_error(tmp_path):
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(config_path))
