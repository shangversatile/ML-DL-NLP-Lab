"""Train, checkpoint, load, and run inference for the digits MLP baseline."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.inference.digits_inference import predict_digit_batch
from src.models.checkpoint import (
    build_multiclass_mlp_metadata,
    load_multiclass_mlp_checkpoint,
    save_multiclass_mlp_checkpoint,
)
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_multiclass_mlp, train_multiclass_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def create_adam(optimizer_config: dict) -> ParameterAdam:
    if optimizer_config["name"].lower() != "adam":
        raise ValueError("Only the Adam optimizer is configured for this experiment.")

    return ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "digits_checkpoint_inference.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    checkpoint_config = config["checkpoint"]
    inference_config = config["inference"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("digits_checkpoint_inference", log_file=log_file)

    logger.info("Experiment: digits_checkpoint_inference")
    logger.info("Seed: %s", seed)

    X, y, _ = load_digits_dataset(
        scale_pixels=data_config["scale_pixels"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = (
        stratified_train_val_test_split(
            X,
            y,
            val_ratio=data_config["val_ratio"],
            test_ratio=data_config["test_ratio"],
            seed=seed,
        )
    )

    logger.info("Train shape: X=%s, y=%s", X_train.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val.shape, y_val.shape)
    logger.info("Test shape: X=%s, y=%s", X_test.shape, y_test.shape)

    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        seed=seed,
    )
    optimizer = create_adam(optimizer_config)

    history = train_multiclass_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        seed=seed,
        log_every=training_config["log_every"],
        logger=logger,
    )

    final_train_metrics = evaluate_multiclass_mlp(model, X_train, y_train)
    final_val_metrics = evaluate_multiclass_mlp(model, X_val, y_val)
    final_test_metrics = evaluate_multiclass_mlp(model, X_test, y_test)
    logger.info(
        "Final train CE: %.6f, accuracy: %.6f",
        final_train_metrics["cross_entropy"],
        final_train_metrics["accuracy"],
    )
    logger.info(
        "Final validation CE: %.6f, accuracy: %.6f",
        final_val_metrics["cross_entropy"],
        final_val_metrics["accuracy"],
    )
    logger.info(
        "Final test CE: %.6f, accuracy: %.6f",
        final_test_metrics["cross_entropy"],
        final_test_metrics["accuracy"],
    )
    logger.info("Number of parameter updates: %s", history["update_count"])

    metadata = build_multiclass_mlp_metadata(
        model=model,
        input_scaling=checkpoint_config["input_scaling"],
        class_names=checkpoint_config["class_names"],
        extra_metadata={
            "seed": seed,
            "training_epochs": training_config["num_epochs"],
            "batch_size": training_config["batch_size"],
            "optimizer": optimizer_config,
            "final_train_metrics": final_train_metrics,
            "final_validation_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
        },
    )

    checkpoint_path = PROJECT_ROOT / checkpoint_config["path"]
    save_multiclass_mlp_checkpoint(
        model,
        checkpoint_path,
        metadata,
    )
    loaded_model, loaded_metadata = load_multiclass_mlp_checkpoint(checkpoint_path)

    assert loaded_metadata["class_names"] == checkpoint_config["class_names"]
    assert loaded_metadata["n_features"] == model.n_features
    assert loaded_metadata["hidden_dim"] == model.hidden_dim
    assert loaded_metadata["num_classes"] == model.num_classes

    original_probabilities = model.predict_proba(X_test)
    loaded_probabilities = loaded_model.predict_proba(X_test)
    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
        rtol=1e-12,
        atol=1e-12,
    )

    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)
    np.testing.assert_array_equal(original_predictions, loaded_predictions)

    logger.info("Saved checkpoint to: %s", checkpoint_path)
    logger.info("Loaded checkpoint model class: %s", loaded_metadata["model_class"])
    logger.info("Loaded checkpoint class names: %s", loaded_metadata["class_names"])
    logger.info("Loaded-model probabilities match original model probabilities.")
    logger.info("Loaded-model predictions match original model predictions.")

    n_demo_examples = inference_config["n_demo_examples"]
    demo_results = predict_digit_batch(
        loaded_model,
        X_test[:n_demo_examples],
        top_k=inference_config["top_k"],
    )
    for index in range(n_demo_examples):
        logger.info(
            (
                "Demo %s - true=%s prediction=%s confidence=%.6f "
                "top_k_indices=%s top_k_probabilities=%s"
            ),
            index,
            int(y_test[index]),
            int(demo_results["predictions"][index]),
            float(demo_results["confidences"][index]),
            demo_results["top_k_indices"][index].tolist(),
            np.round(demo_results["top_k_probabilities"][index], 6).tolist(),
        )

    logger.info(
        (
            "This checkpoint stores model parameters and metadata for reproducible "
            "inference. It does not yet include GUI input preprocessing, calibration, "
            "or distribution-shift analysis."
        )
    )


if __name__ == "__main__":
    main()
