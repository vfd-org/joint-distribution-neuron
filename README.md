# Joint-Distribution Neuron (HCRNN Prototype)

A prototype implementation of **joint-distribution neural units** and a small multi-layer network inspired by Jarek Duda’s Hierarchical Correlation Reconstruction (HCR) model.

Unlike conventional neural networks built around scalar activations and one-way information flow, these units learn **local joint probability densities** and support **bidirectional inference**, enabling:

- **X → Y** forward inference  
- **Y → X** reverse inference  
- **conditional sampling**  
- **uncertainty propagation**  
- **reversible regression**  
- **multi-layer density transformations**

This repo provides the first working, test-verified prototype of a **multi-layer HCR neural network (HCRNN)**.

---

## ✨ Features

### ✔ Joint-Distribution Neuron
Each neuron represents a probability density over its inputs using an orthonormal polynomial basis.

Capabilities:
- Learnable joint density `ρ(x)`
- Conditional inference `p(y|x)` and `p(x|y)`
- Density evaluation and sampling
- Support for 2D and 3D demos

### ✔ Multi-Layer HCR Network
Stack and train multiple joint-distribution units into a reversible network.

- Forward pass: `X → Hidden → Y`
- Reverse pass: `Y → Hidden → X`
- Alternating, CMA-ES, and coordinate descent training
- Resonance-based regularization to favor coherent, low-frequency components
- Uncertainty propagation across layers

### ✔ Extensive Tests
78 tests validate:
- basis orthonormality
- joint density estimation
- conditional inference correctness
- multi-layer forward/reverse reconstruction
- regularization stability

---

## 📂 Repository Structure

