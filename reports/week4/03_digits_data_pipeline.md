# Digits Data Pipeline

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why move from synthetic XOR to real digits

XOR validated that a scratch MLP can learn a nonlinear binary decision boundary. The handwritten digits task is a larger and more realistic capstone step: inputs are real images, labels span ten classes, handwriting can be ambiguous, and model quality must be monitored with train, validation, and test splits.

This is the first real-data stage for the multiclass MLP. It checks whether the same explicit forward pass, softmax output, cross-entropy loss, backpropagation, and optimizer interface can support a practical image classification baseline.

## 2. Dataset structure

The dataset is loaded with `sklearn.datasets.load_digits()`. Each example is an `8 x 8` grayscale image. The model receives a flattened feature vector with 64 input features, one per pixel.

Labels are integer class IDs from 0 to 9. Original pixel intensities are in the range 0 to 16, and the baseline scales them to the range 0 to 1 before training.

## 3. Why scale pixels to 0-1

Large input magnitudes can create large hidden activations and logits, which can make optimization less stable. Scaling pixels to 0-1 keeps feature values small and helps Adam make smoother early updates.

For this first baseline, 0-1 scaling is also more interpretable than arbitrary train-set standardization. A value still means image intensity: 0 is background-like and 1 is the brightest pixel value in the original digits data.

## 4. Why use stratified train/validation/test split

Multiclass evaluation requires every class to be represented in every split. Without stratification, a small validation or test split can accidentally underrepresent a digit, making aggregate metrics less meaningful.

The training split is used to fit parameters. The validation split is used for development monitoring during the baseline run. The test split is reserved for final baseline reporting after training.

## 5. Training pipeline

The baseline trains `MulticlassMLPScratch`, a single-hidden-layer neural network with a ReLU hidden layer and a softmax output layer. Cross entropy measures the probability assigned to the correct digit class.

Mini-batches are shuffled reproducibly each epoch. Gradients are computed by the scratch model, mapped to parameter names, and passed to the generic Adam optimizer. Metrics are recorded after each full epoch on the complete training and validation sets.

## 6. Interpretation boundaries

This baseline does not include checkpointing. It does not include a confusion matrix or per-class error analysis yet. It also does not evaluate local handwritten inputs or distribution shift from the scikit-learn digits dataset.

High aggregate accuracy will not imply reliability. Detailed evaluation follows in later tasks, including confusion analysis, per-class accuracy, confidence checks, and local handwritten-input diagnostics.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
