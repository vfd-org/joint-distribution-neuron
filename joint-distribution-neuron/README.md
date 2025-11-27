# HCRNN: Multi-Layer HCR-Based Joint Distribution Network

A Python implementation of multi-layer Hierarchical Correlation Reconstruction (HCR) networks, based on Jarek Duda's joint distribution neuron concept from [arXiv:2405.05097](https://arxiv.org/abs/2405.05097).

**Now featuring multi-layer networks with bidirectional inference, resonance regularization, and uncertainty propagation.**

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HCR Network (HCRNN)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input X ──┬──► [W₁] ──► JointDensity₁ ──┬──► Hidden H₁           │
│             │         p(X, H₁)            │                        │
│             │                              │                        │
│             │    ◄── Reverse ◄────────────┘                        │
│             │                                                       │
│             └──────────────────────────────────────────────────────┐│
│                                                                     │
│   H₁ ───────┬──► [W₂] ──► JointDensity₂ ──┬──► Hidden H₂           │
│             │         p(H₁, H₂)           │                        │
│             │                              │                        │
│             │    ◄── Reverse ◄────────────┘                        │
│             │                                                       │
│             └──────────────────────────────────────────────────────┐│
│                                                                     │
│   H₂ ───────┬──► [W₃] ──► JointDensity₃ ──┬──► Output Y            │
│             │         p(H₂, Y)            │                        │
│             │                              │                        │
│             │    ◄── Reverse ◄────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Key: Each layer learns a JOINT distribution, enabling bidirectional flow.
     Forward:  X → H₁ → H₂ → Y   (standard inference)
     Reverse:  Y → H₂ → H₁ → X   (inverse inference)
```

## What Makes HCRNN Different?

| Feature | Traditional NN | HCRNN |
|---------|---------------|-------|
| Distribution | p(Y\|X) only | Full p(X,Y) at each layer |
| Inference | Forward only | Bidirectional |
| Inverse | Requires separate model | Built-in reverse pass |
| Uncertainty | Typically unavailable | Variance at each layer |
| Constraints | Input → Output | Can propagate both ways |

## Key Features

### 1. **Bidirectional Inference**
```python
# Forward: X → Y
Y_pred = net.forward(X)

# Reverse: Y → X (from the SAME model!)
X_recon = net.reverse(Y_pred)
```

### 2. **Uncertainty Quantification**
```python
# Get output variance at any layer
variance = net.compute_uncertainty(X, layer_idx=1)
```

### 3. **Resonance-Based Regularization**
Penalizes high-frequency polynomial coefficients, encouraging smooth, stable, coherent representations—implementing VFD's "coherence filter" principle.

### 4. **Gradient-Free Training**
Three training methods:
- `alternating`: Alternating least squares (fast, default)
- `cmaes`: Powell optimization
- `coordinate`: Coordinate descent

## Installation

```bash
cd joint-distribution-neuron
pip install -r requirements.txt
pip install -e .
```

Requirements: Python 3.8+, numpy, scipy, matplotlib, pytest

## Quick Start

### Single Neuron (Basic)

```python
from hcrnn import build_tensor_basis, JointDensity, conditional_expectation

# Fit joint density on 2D data
basis = build_tensor_basis(dim=2, degrees_per_dim=3)
joint = JointDensity(basis)
joint.fit(X)  # X in [0,1]^2

# Bidirectional inference
E_y_given_x = conditional_expectation(joint, target_index=1,
                                       given_indices=[0], given_values=[0.5])
E_x_given_y = conditional_expectation(joint, target_index=0,
                                       given_indices=[1], given_values=[0.7])
```

### Multi-Layer Network

```python
from hcrnn import HCRNetwork, LayerSpec

# Define architecture: 2 → 4 → 2
specs = [
    LayerSpec(input_dim=2, output_dim=4, basis_degree=3, resonance_decay=0.1),
    LayerSpec(input_dim=4, output_dim=2, basis_degree=3, resonance_decay=0.1),
]

# Create and train
net = HCRNetwork(specs)
net.fit(X_train, Y_train, method="alternating", max_iter=10)

# Forward inference
Y_pred = net.forward(X_test)

# Reverse inference (from output back to input!)
X_reconstructed = net.reverse(Y_pred)

# Get intermediate activations
Y_pred, intermediates = net.forward(X_test, return_intermediates=True)

# Measure reconstruction quality
errors = net.reconstruction_error(X_test, Y_test)
print(f"Forward RMSE: {errors['forward_rmse']:.4f}")
print(f"Reverse RMSE: {errors['reverse_rmse']:.4f}")
```

## Running the Demos

```bash
# Single neuron demo
python examples/demo_2d_correlated.py

# Multi-layer network demo
python examples/hcrnn_multilayer_demo.py
```

## Running Tests

```bash
pytest -v                    # All tests
pytest tests/test_network.py # Network tests only
```

## Project Structure

```
hcrnn/
├── __init__.py           # Package exports
├── basis.py              # Orthonormal polynomial basis
├── joint_density.py      # Single JointDensity neuron
├── conditionals.py       # Conditional inference utilities
└── network.py            # Multi-layer HCRNetwork

examples/
├── demo_2d_correlated.py      # Single neuron demo
└── hcrnn_multilayer_demo.py   # Multi-layer demo

tests/
├── test_basis.py
├── test_joint_density.py
├── test_conditionals.py
└── test_network.py
```

## API Reference

### `LayerSpec`

```python
LayerSpec(
    input_dim: int,        # Input dimensionality
    output_dim: int,       # Output dimensionality
    basis_degree: int = 3, # Polynomial degree for joint density
    resonance_decay: float = 0.1,  # Regularization strength
)
```

### `HCRNetwork`

```python
net = HCRNetwork(layer_specs: List[LayerSpec], seed: int = 42)

# Training
net.fit(X, Y, method="alternating", max_iter=10, verbose=True)

# Inference
Y = net.forward(X)                    # Forward pass
X = net.reverse(Y)                    # Reverse pass
Y, acts = net.forward(X, return_intermediates=True)

# Analysis
variance = net.compute_uncertainty(X, layer_idx=-1)
errors = net.reconstruction_error(X, Y)
```

### `JointDensity`

```python
joint = JointDensity(basis)
joint.fit(X)              # X in [0,1]^d
rho = joint.density(x)    # Evaluate density
samples = joint.sample(n) # Generate samples
```

### Conditional Inference

```python
from hcrnn import (
    conditional_density,      # p(target | given)
    conditional_expectation,  # E[target | given]
    conditional_variance,     # Var[target | given]
    conditional_mode,         # Mode of conditional
    sample_conditional,       # Sample from conditional
)
```

## How It Works

### Layer-wise Joint Distribution

Each layer maintains:
1. **Projection matrix W**: Linear transform from input to output
2. **Joint density p(input, output)**: Polynomial expansion over concatenated space

The joint density enables:
- Forward: `E[output | input]` via conditional expectation
- Reverse: `E[input | output]` via reverse conditional

### Resonance Regularization

High-order polynomial coefficients are penalized:
```
penalty = Σⱼ exp(decay × degree(j)) × aⱼ²
```

This encourages:
- Smooth, low-frequency representations
- Stable correlation patterns
- Coherent information propagation

### Training

**Alternating Least Squares** (default):
1. Initialize intermediate targets via interpolation
2. Fit projection W via least squares
3. Fit joint density on (input, projection output)
4. Use joint density to refine targets
5. Repeat

## Limitations

1. **Scalability**: Joint density over (input, output) grows with `(degree+1)^(dim_in + dim_out)`
2. **Polynomial negativity**: Density can go negative; clamping is used
3. **Numerical integration**: Grid-based, limits practical joint dimensionality to ~6-8
4. **Bounded domain**: Data must be normalized to [0,1]

## Theoretical Background

The HCRNN implements key concepts from HCR/VFD theory:

- **Joint Distribution Neuron**: Each neuron models full p(X,Y), not just p(Y|X)
- **Bidirectional Inference**: Constraints propagate in both directions
- **Resonance Selection**: High-frequency (noisy) correlations are filtered out
- **Coherence**: Stable, low-order moments dominate

This creates a network where information flows like waves in a field, with coherent patterns reinforcing and noise dissipating.

## References

- Duda, J. (2024). "Hierarchical Correlation Reconstruction with Joint Distribution Neuron" [arXiv:2405.05097](https://arxiv.org/abs/2405.05097)
- Duda, J. - Various papers on HCR, VFD, and field-based computation

## License

MIT License

## Future Directions

1. **Continuous time dynamics**: Temporal evolution of joint distributions
2. **Attractor formation**: Stable states as coherent field configurations
3. **Neural basis**: Replace polynomials with learned neural features
4. **GPU acceleration**: Batch operations for larger networks
5. **Hierarchical attention**: Selective constraint propagation
