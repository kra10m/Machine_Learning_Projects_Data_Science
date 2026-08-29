# Fully Connected Neural Networks

A three-part deep learning project exploring Fully Connected Neural Networks (FCNNs) from implementation fundamentals to learned representations, spatial robustness, gradient flow, and training sensitivity.

The project combines a NumPy implementation built from scratch with PyTorch experiments to investigate not only how FCNNs work, but also how input representation, network depth, activation functions, normalization, regularization, learning rate, and optimizer choice affect their behavior.

---

## Project Overview

The project is organized as a progression of experiments:

| Part | Focus | Dataset | Main Question |
|---|---|---|---|
| 1 | FCNN From Scratch | UCI Adult Census Income | How does an FCNN work internally, and how does feature scaling affect optimization? |
| 2 | Learned Representations & Spatial Permutation | MNIST | What does an FCNN learn, and does it depend on the original spatial arrangement of pixels? |
| 3 | Deep FCNN & Training Robustness | Tiny ImageNet (10 classes) | What happens when an FCNN becomes deep, and which training choices matter? |

---

# Part 1 — FCNN From Scratch

### Dataset

**UCI Adult Census Income**

The first part implements a three-layer fully connected neural network entirely with NumPy for binary income classification.

The network architecture is:

105 → 64 → 32 → 1

### Implementation

The neural network is implemented manually without using a deep-learning framework.

The implementation includes:

- He initialization
- Zero bias initialization
- ReLU activation
- Sigmoid output activation
- Binary Cross-Entropy loss
- Forward propagation
- Manual backpropagation
- Vanilla Stochastic Gradient Descent
- Gradient-norm tracking

### Preprocessing

The original Adult dataset contains numerical and categorical features.

The preprocessing pipeline includes:

- Handling missing values represented by `?`
- Replacing missing categorical values using the training-set mode
- One-hot encoding categorical variables
- Converting the resulting features to NumPy arrays
- Preserving the original Adult training/test split

After preprocessing, the network receives **105 input features**.

### Feature Scaling Experiment

Two versions of the same FCNN architecture are trained:

1. Raw preprocessed features
2. MinMax-scaled features

The purpose is not simply to compare accuracy, but to investigate how feature magnitude affects gradient-based optimization.

### Results

| Metric | Raw Features | MinMax Scaled |
|---|---:|---:|
| Final training loss | 0.565254 | 0.445844 |
| Training accuracy | 76.09% | 75.91% |
| Test accuracy | 76.61% | 76.37% |
| Initial `||dW1||` | 126,771.934 | 0.304967 |
| Final `||dW1||` | 0.0 | 0.075506 |

### Key Finding

Feature scaling dramatically changed the magnitude of the gradients.

The raw-feature model began with extremely large first-layer gradients, while MinMax scaling reduced the initial gradient norm by several orders of magnitude. The scaled model also reached a lower final training loss.

However, test accuracy changed very little.

This demonstrates an important distinction between **optimization behavior** and **final predictive accuracy**: feature scaling can substantially change the numerical behavior of gradient descent without necessarily improving test accuracy under the same training setup.

### Important Interpretation

The Adult target is imbalanced. The test-set majority-class baseline is approximately **76.38%**, compared with **76.61%** for the raw FCNN.

Therefore, the reported accuracy should not be interpreted as strong predictive performance. The primary purpose of this experiment is understanding FCNN mechanics and the effect of feature scale on optimization.

---

# Part 2 — MNIST: Learned Representations and Spatial Permutation

### Dataset

**MNIST**

The second part uses a fully connected neural network to investigate what the network learns from flattened handwritten-digit images and whether it depends on the original spatial arrangement of pixels.

Each MNIST image is:

28 × 28 = 784 pixels

The images are flattened into 784-dimensional vectors before being passed to the FCNN.

### Architecture

784 → 128 → 64 → 10

The network uses:

- Two hidden layers
- ReLU activations
- Cross-Entropy loss
- Adam optimizer
- Learning rate = `0.001`
- Five training epochs

### Learned Weight Visualization

The weights of neurons in the first hidden layer are extracted and reshaped from:

784

into:

28 × 28

This allows the learned weights to be visualized as heatmaps.

The visualization provides a way to inspect which pixel coordinates contribute positively or negatively to individual hidden neurons.

---

## Fixed Pixel Permutation Experiment

To investigate whether the FCNN depends on the original spatial arrangement of the image pixels, a single random permutation of the 784 pixel coordinates is generated.

The **same permutation is applied to every training and test image**.

This means that:

- The pixel layout is visually scrambled.
- The same original pixel always maps to the same new coordinate.
- The underlying pixel information is preserved.
- The task remains learnable by a sufficiently flexible fully connected network.

A separate FCNN is trained on the scrambled representation using the same architecture and training procedure.

### Results

| Dataset | Test Accuracy |
|---|---:|
| Normal MNIST | 97.27% |
| Scrambled MNIST | 97.37% |
| Difference | -0.10 percentage points |

### Key Finding

The two models achieve essentially identical test accuracy.

This demonstrates that a standard FCNN does not have an inherent spatial inductive bias. Because the same permutation is applied consistently to every image, the network can learn a new mapping from the rearranged input coordinates to the digit classes.

The result does **not** imply that spatial relationships are unimportant for image recognition.

Instead, it demonstrates that spatial locality is not explicitly encoded in the architecture of a standard FCNN.

This is one of the key differences between fully connected networks and architectures such as convolutional neural networks, which explicitly incorporate spatial locality and weight sharing.

> **Note:** The model used for first-layer weight visualization is a separate training run from the models used in the controlled normal-versus-scrambled comparison.

---

# Part 3 — Deep FCNN: Gradient Flow and Training Robustness

### Dataset

**Tiny ImageNet — 10-class subset**

The third part investigates what happens when the fully connected architecture becomes substantially deeper.

The selected subset contains:

- 10 classes
- 500 training images per class
- 5,000 training images total
- 500 validation images

Each image is an RGB `64 × 64` image.

After flattening:

64 × 64 × 3 = 12,288 input features

### Architecture

The deep FCNN uses eight hidden layers:

12,288
   ↓
1,024
   ↓
512
   ↓
256
   ↓
128
   ↓
128
   ↓
64
   ↓
64
   ↓
32
   ↓
10

The purpose of using this deliberately deep architecture is to make gradient-flow and optimization behavior visible.

---

## Part 3.1 — Vanishing Gradients

Two deep FCNN variants are compared.

### Model A — Deep Sigmoid

Sigmoid activations are used throughout the hidden layers.

### Model B — ReLU + Batch Normalization

Each hidden layer uses:

Linear → BatchNorm → ReLU

Both models use the same architecture, training data, optimizer, learning rate, and five-epoch training budget.

The L2 norm of the gradient of the first fully connected layer is tracked after each epoch.

### Results

| Epoch | Sigmoid `||dW1||` | ReLU + BatchNorm `||dW1||` |
|---|---:|---:|
| 1 | 4.078e-7 | 1.1384 |
| 2 | 7.790e-8 | 0.1354 |
| 3 | 6.924e-8 | 0.1966 |
| 4 | 8.319e-8 | 0.3142 |
| 5 | 1.157e-7 | 0.3016 |

Training loss:

| Epoch | Sigmoid | ReLU + BatchNorm |
|---|---:|---:|
| 1 | 2.3134 | 2.1820 |
| 2 | 2.3061 | 1.9812 |
| 3 | 2.3058 | 1.8906 |
| 4 | 2.3055 | 1.8142 |
| 5 | 2.3054 | 1.7474 |

### Key Finding

The Sigmoid network produces extremely small first-layer gradients, remaining around `10⁻⁷` to `10⁻⁸` across the five epochs.

Its training loss also remains close to:

ln(10) ≈ 2.303

which corresponds to the random-guessing region for a ten-class classification problem.

In contrast, the ReLU + BatchNorm network maintains substantially larger first-layer gradients and continues reducing its training loss.

This provides evidence of severe gradient attenuation in the deep Sigmoid network and demonstrates why activation and normalization choices become increasingly important as network depth increases.

> The experiment specifically measures the gradient reaching the **first layer**. It does not directly measure gradient norms at every hidden layer.

---

# Part 3.2 — Training Ablation Study

The second experiment investigates the sensitivity of the deep FCNN to several training choices.

### Baseline Configuration

- ReLU + BatchNorm
- Dropout `p = 0.2`
- Adam optimizer
- Learning rate = `0.001`
- Five training epochs

One factor is changed at a time while keeping the other main settings fixed.

### Results

| Configuration | Validation Accuracy | Change vs Baseline |
|---|---:|---:|
| Baseline | 29.20% | 0.00 pp |
| No Dropout | 31.60% | +2.40 pp |
| Learning Rate ×10 | 28.20% | −1.00 pp |
| Learning Rate ÷10 | 18.60% | −10.60 pp |
| Adam → SGD | 9.80% | −19.40 pp |

### Interpretation

Removing Dropout improved validation accuracy by **2.40 percentage points**. Within this short five-epoch training budget, the regularization introduced by Dropout did not appear beneficial.

Increasing the learning rate by 10× produced only a small decrease in validation accuracy, while reducing it by 10× caused a much larger decrease. This suggests that the smaller learning rate did not allow the network to optimize sufficiently within the available five epochs.

Replacing Adam with vanilla SGD produced the largest performance drop, from **29.20% to 9.80%**.

Therefore, **among the tested configurations**, optimizer choice had the largest measured effect on validation performance under the five-epoch training budget.

This result should not be interpreted as evidence that Adam is universally superior to SGD. It describes the behavior observed under this specific architecture, learning rate, dataset, and training budget.

---

# Overall Findings

The three parts investigate different aspects of fully connected neural networks while building on the same underlying architecture.

### 1. Feature scale affects optimization

Part 1 showed that raw feature magnitudes can produce dramatically different gradient scales.

MinMax scaling reduced the initial first-layer gradient norm from approximately:

126,772 → 0.305

while producing similar test accuracy under the tested training setup.

### 2. FCNNs do not have an inherent spatial inductive bias

Part 2 showed that a fixed permutation of MNIST pixel coordinates caused essentially no change in classification accuracy:

Normal:    97.27%
Scrambled: 97.37%

Because the permutation was consistent across the dataset, the FCNN could learn the new coordinate mapping.

### 3. Deep networks are sensitive to gradient-flow choices

Part 3 showed severe first-layer gradient attenuation when Sigmoid activations were repeatedly used in a deep FCNN.

Replacing them with ReLU and Batch Normalization resulted in substantially larger gradients and continued loss reduction.

### 4. Optimization choices matter

The ablation study showed that optimizer choice produced the largest performance change among the tested configurations, while learning rate and Dropout also affected performance under the limited five-epoch training budget.

### 5. Accuracy must be interpreted in context

The experiments are primarily designed to investigate neural-network mechanics, representation, gradient flow, and optimization behavior.

The results should therefore not be interpreted solely through final accuracy.

---

# Reproducibility

The notebooks contain the complete implementation and executed experiments.

To reproduce the experiments:

1. Clone or download this repository.
2. Install the required Python dependencies.
3. Download the required datasets.
4. Place the datasets in the expected locations under `data/`.
5. Open the corresponding notebook.
6. Run the notebook cells in order.

The datasets themselves are **not included in this repository**.

This avoids committing large dataset files to GitHub.

> The exact training results may vary across runs because the PyTorch experiments do not use a fixed global PyTorch random seed for model initialization. The fixed pixel permutation in Part 2 is reproducible.

---

# Dataset Setup

The following datasets are required:

### Part 1 — UCI Adult Census Income

Used for binary income classification.

Place the required Adult dataset files in:

data/Adult/

The Part 1 notebook expects the dataset files relative to the notebook's location.

### Part 2 — MNIST

Used for handwritten-digit classification.

Place the MNIST dataset files in the location expected by the Part 2 notebook.

### Part 3 — Tiny ImageNet

Used for the deep FCNN experiments.

Place the Tiny ImageNet directory under:

data/TinyImageNet/

The notebook expects the standard Tiny ImageNet directory structure containing:

- `wnids.txt`
- training images
- validation images
- `val_annotations.txt`

The selected 10-class subset is created by the notebook.

> **Do not upload the datasets to GitHub.** The `data/` directory is included to document and preserve the expected project structure.

---

# Running the Notebooks

Each notebook corresponds to one part of the project.

### Part 1

Open:

notebooks/Part1_FCNN_From_Scratch.ipynb

Ensure the Adult dataset is available in the expected `data/` location, then run the notebook from top to bottom.

### Part 2

Open:

notebooks/Part2_MNIST_FCNN.ipynb

Ensure the MNIST dataset is available in the expected local dataset location, then run the notebook from top to bottom.

### Part 3

Open:

notebooks/Part3_Deep_FCNN_Robustness.ipynb

Ensure Tiny ImageNet is available in the expected `data/TinyImageNet/` location, then run the notebook from top to bottom.

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Jupyter Notebook

---

# Key Concepts Demonstrated

This project covers:

- Fully Connected Neural Networks
- Neural-network initialization
- He initialization
- Forward propagation
- Backpropagation
- Activation functions
- ReLU
- Sigmoid
- Binary Cross-Entropy
- Cross-Entropy Loss
- Vanilla SGD
- Adam
- Feature scaling
- Gradient norms
- Gradient attenuation
- Batch Normalization
- Dropout
- Learning-rate sensitivity
- Spatial inductive bias
- Learned feature representations
- Neural-network optimization
- Controlled ablation experiments

---

# Project Report

A detailed technical report containing the methodology, architectures, experiments, results, visualizations, interpretations, and limitations is included in the repository.

**[Read the Technical Report](./FCNN_Technical_Report_Portfolio_Final.pdf)**

---

# Limitations

Several results are intentionally interpreted within the limits of the experimental setup.

### Adult Census Income

The target is imbalanced, making accuracy close to the majority-class baseline. The experiment therefore focuses more on optimization behavior than on demonstrating a high-performing income classifier.

### MNIST

The normal and scrambled models are trained independently. The experiment demonstrates the effect of a fixed coordinate permutation, but it is not intended as a comparison against convolutional architectures.

### Tiny ImageNet

The deep FCNN experiments use only a 10-class subset and a five-epoch training budget. The ablation results therefore describe sensitivity under this specific experimental setup rather than providing universal conclusions about optimizers, learning rates, or regularization.

---

# Conclusion

The project demonstrates how a fully connected neural network behaves across increasing levels of complexity:

From-scratch implementation
        ↓
Learned representations
        ↓
Spatial permutation
        ↓
Deep network gradient flow
        ↓
Training robustness and ablation

Rather than focusing only on final accuracy, the experiments examine the mechanisms that determine how FCNNs learn and how their behavior changes with feature scale, input representation, depth, activation functions, normalization, regularization, learning rate, and optimizer choice.
