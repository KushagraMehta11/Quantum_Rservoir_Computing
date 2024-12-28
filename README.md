# Quantum_Reservoir_Computing

This repository contains solutions to two quantum machine learning assignments focusing on credit risk modeling using the **Statlog (German Credit) Dataset**. Each assignment involves different approaches and implementations of quantum-based techniques for classifying imbalanced data.

---

## Assignment 1: Quantum Reservoir Computing with Qiskit

### Overview
This assignment implements a **Quantum Reservoir Computing (QRC)** model combined with **Logistic Regression** to classify "good" and "bad" credit risks. Key highlights include:
- Preprocessing the Statlog dataset to handle categorical and numerical data.
- Applying **SMOTE** (Synthetic Minority Over-sampling Technique) and **RandomUnderSampler** to address class imbalance.
- Using **Qiskit** to create quantum circuits that transform input data into quantum feature representations.

### Steps
1. **Data Preprocessing**:
   - Split features into categorical and numerical groups.
   - Apply **OneHotEncoder** and **StandardScaler** for feature transformation.
   - Reduce dimensionality to 8 principal components using PCA.
   - Address class imbalance using a pipeline combining SMOTE and RandomUnderSampler.

2. **Quantum Reservoir Computing**:
   - Construct quantum circuits using Qiskit's `GenericBackendV2`.
   - Embed features using parameterized `RY` and `RZ` rotations.
   - Introduce entanglement using `CNOT` gates in a ring topology.
   - Extract quantum feature vectors from measurement results.

3. **Hybrid Classifier**:
   - Combine the quantum feature transformations with **Logistic Regression**.
   - Tune the decision threshold to optimize recall for the minority class.

### Results
- **Minority Class Recall (tuned):** 0.517
- The model successfully balances recall for "bad credit risk" cases while mitigating class imbalance.

---

## Assignment 2: Quantum Reservoir Computing with PennyLane

### Overview
This assignment re-implements credit risk modeling using **PennyLane**, introducing an end-to-end differentiable quantum-classical model. The focus is on leveraging PennyLane’s automatic differentiation for training quantum circuits.

### Steps
1. **Data Preprocessing**:
   - Similar to Assignment 1, data is preprocessed using **OneHotEncoder**, **StandardScaler**, PCA (4 components), and class balancing (SMOTE + RandomUnderSampler).

2. **PennyLane Quantum Circuit**:
   - Define a QNode with 4 qubits and initialize using **Hadamard gates**.
   - Encode features using `RY` rotations.
   - Introduce reservoir layers using `CNOT` gates for entanglement and parameterized `RZ`/`RX` rotations for complexity.
   - Output the expectation value of the **Pauli-Z operator**, mapped to a class-2 probability.

3. **Training**:
   - Use **Binary Cross-Entropy Loss** for optimization.
   - Train embedding and reservoir parameters with mini-batch gradient descent over 30 epochs.

4. **Evaluation**:
   - Classify test samples based on a threshold (default: 0.3).
   - Evaluate the model using precision, recall, and F1-score metrics.

### Results and Comparison
- **Minority Class Recall:** 0.967
- The model biases predictions toward the minority class, significantly reducing false negatives. This high recall improves detection of "bad credit risks" but sacrifices precision and overall accuracy.

---

## Comparison of Assignments
| Metric                       | Assignment 1 (Qiskit) | Assignment 2 (PennyLane) |
|------------------------------|-----------------------|--------------------------|
| Minority Class Recall        | 0.517                | 0.967                   |
| Training Framework           | Qiskit               | PennyLane               |
| Dimensionality Reduction     | PCA (8 components)   | PCA (4 components)      |
| Class Balancing              | SMOTE + Undersample  | SMOTE + Undersample     |
| Quantum Circuit Parameters   | Fixed (GenericBackendV2) | Trainable (differentiable) |
| Threshold for Classification | Tuned (to achieve ~0.5 recall) | Default (0.3)          |

### Key Takeaways
- **Assignment 1:** Balances recall and precision better but achieves lower recall for the minority class.
- **Assignment 2:** Maximizes minority recall, prioritizing detection of risky borrowers at the expense of precision and accuracy.

---

## Requirements
- Python 3.8+
- Libraries:
  - `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`
  - `qiskit` (Assignment 1)
  - `pennylane` (Assignment 2)

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/bloq-quantum-assignments.git
   cd bloq-quantum-assignments
