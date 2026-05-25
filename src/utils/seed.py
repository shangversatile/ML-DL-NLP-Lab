"""Random seed helpers for reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    This function sets:
    - Python random seed
    - NumPy random seed
    - PYTHONHASHSEED environment variable
    """
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
