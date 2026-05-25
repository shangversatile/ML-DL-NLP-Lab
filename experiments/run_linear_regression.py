"""Week 1 smoke-test pipeline for linear regression data."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_linear_regression_data
from src.data.preprocessing import standardize_features, train_val_split
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "linear_regression.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("linear_regression_smoke_test", log_file=log_file)

    logger.info("Experiment: linear_regression_smoke_test")
    logger.info("Seed: %s", seed)

    X, y, true_weights, true_bias = make_linear_regression_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        noise=data_config["noise"],
        seed=seed,
    )
    logger.info("X shape: %s, y shape: %s", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=data_config["val_ratio"],
        seed=seed,
    )
    logger.info("X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)
    logger.info("X_val shape: %s, y_val shape: %s", X_val.shape, y_val.shape)
    logger.info("True weights: %s", true_weights)
    logger.info("True bias: %s", true_bias)

    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train, X_val)
    logger.info("Training feature mean after standardization: %s", X_train_scaled.mean(axis=0))
    logger.info("Training feature std after standardization: %s", X_train_scaled.std(axis=0))
    logger.info("No model is trained yet because this is a Week 1 pipeline smoke test.")

    # Keep these names visible for beginner-friendly inspection in a debugger.
    _ = X_val_scaled, mean, std


if __name__ == "__main__":
    main()
