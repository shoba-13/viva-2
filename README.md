# Cat vs Dog Image Classifier (CNN)

A convolutional neural network built with **TensorFlow 2 and Keras** that classifies images as a cat or a dog. Completed as part of the freeCodeCamp *Machine Learning with Python* certification.

**Goal:** at least 63% accuracy on the test set (stretch goal: 70%)
**Result:** `[your final test accuracy]%`

## Dataset

```
cats_and_dogs
├── train        (1,000 cats + 1,000 dogs)
├── validation   (500 cats + 500 dogs)
└── test         (50 unlabeled images)
```

The test images are loaded with `shuffle=False` so predictions stay in the order the challenge test expects.

## Approach

**1. Preprocessing** – `ImageDataGenerator` reads and decodes images into floating-point tensors, rescaled from [0, 255] to [0, 1] and resized to 150 × 150.

**2. Data augmentation** – the training set is small, so random transformations reduce overfitting:

| Transformation | Value |
|---|---|
| Rotation | 40° |
| Width / height shift | 0.2 |
| Shear | 0.2 |
| Zoom | 0.2 |
| Horizontal flip | True |

**3. Model** – a `Sequential` CNN:

```
Conv2D(32)  → MaxPooling2D
Conv2D(64)  → MaxPooling2D
Conv2D(128) → MaxPooling2D
Conv2D(128) → MaxPooling2D
Flatten → Dropout(0.5) → Dense(512, ReLU) → Dense(1, Sigmoid)
```

Optimizer: Adam · Loss: binary cross-entropy · Metric: accuracy

**4. Training** – `[20]` epochs, batch size 128, with validation on the 1,000 validation images.

**5. Prediction** – the model outputs a probability per test image (closer to 0 = cat, closer to 1 = dog). Predictions are rounded and checked by the final test cell.

## Results

| Metric | Value |
|---|---|
| Training accuracy | `[xx]%` |
| Validation accuracy | `[xx]%` |
| Test accuracy | `[xx]%` |

## Tech Stack

Python 3 · TensorFlow / Keras · NumPy · Matplotlib · Google Colaboratory

## How to Run

1. Open the notebook in Google Colab and save a copy.
2. Run the cells in order (the dataset downloads in cell 2).
3. Train the model, then view the plots and predictions.
4. Run the last cell to check whether the challenge passes.
5. Turn on link sharing ("anyone with the link") before submitting.

## Possible Improvements

- Train for more epochs
- Use transfer learning (MobileNetV2, VGG16)
- Tune learning rate, batch size and dropout
- Add more training data
