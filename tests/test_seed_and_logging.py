import logging
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(42)
    first_array = np.random.rand(5)

    set_seed(42)
    second_array = np.random.rand(5)

    assert np.array_equal(first_array, second_array)


def test_invalid_seed():
    with pytest.raises(TypeError):
        set_seed("42")


def test_logger_creation():
    logger = get_logger("test_logger")

    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_logger_file_output(tmp_path):
    log_file = tmp_path / "logs" / "test.log"
    logger = get_logger("file_logger", log_file=str(log_file))

    logger.info("hello from the logger")

    assert log_file.exists()
    assert "hello from the logger" in log_file.read_text()
