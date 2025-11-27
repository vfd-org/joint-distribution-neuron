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

joint-distribution-neuron/
├── hcrnn/
│ ├── init.py
│ ├── basis.py
│ ├── conditionals.py
│ ├── joint_density.py
│ └── network.py
├── examples/
│ ├── demo_2d_correlated.py
│ ├── hcrnn_multilayer_demo.py
│ └── *.png
├── tests/
│ ├── test_basis.py
│ ├── test_conditionals.py
│ ├── test_joint_density.py
│ └── test_network.py
├── pyproject.toml
├── requirements.txt
└── README.md

yaml
Copy code

---

## 📊 Quick Example

A 2D joint-distribution neuron learning a correlated density and performing forward/reverse inference:

```bash
python examples/demo_2d_correlated.py
A full multi-layer reversible network:

bash
Copy code
python examples/hcrnn_multilayer_demo.py
Example output (condensed):

yaml
Copy code
HCRNetwork([2→4 → 4→2], fitted)

Forward pass: X → Y
Reverse pass: Y → X

forward_mse: 0.6262
reverse_mse: 0.9478
🔧 Installation
bash
Copy code
pip install -r requirements.txt
or with pyproject.toml:

bash
Copy code
pip install .
📚 Background
This project is inspired by:

Jarek Duda — “Biology-inspired joint distribution neurons based on HCR allowing for multidirectional neural networks”
arXiv:2405.05097

The goal is experimental:
to explore whether joint-density units can serve as flexible, biologically plausible building blocks for inference-driven neural architectures.
