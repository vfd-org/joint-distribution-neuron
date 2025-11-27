"""
HCRNN: HCR-based Joint Distribution Network

A minimal prototype implementing Jarek Duda's joint distribution neuron
concept from arXiv:2405.05097, using polynomial basis representations
for bidirectional inference.

Now extended to multi-layer networks with:
- Stacked bidirectional inference
- Learnable projections between layers
- Resonance-based regularization for coherent representations
"""

from hcrnn.basis import (
    orthonormal_poly_1d,
    build_tensor_basis,
    TensorBasis,
)
from hcrnn.joint_density import JointDensity
from hcrnn.conditionals import (
    conditional_density,
    conditional_expectation,
    conditional_variance,
    conditional_mode,
    marginal_density,
    sample_conditional,
)
from hcrnn.network import (
    HCRNetwork,
    LayerSpec,
    LayerState,
)

__version__ = "0.2.0"
__all__ = [
    # Basis
    "orthonormal_poly_1d",
    "build_tensor_basis",
    "TensorBasis",
    # Single neuron
    "JointDensity",
    # Conditionals
    "conditional_density",
    "conditional_expectation",
    "conditional_variance",
    "conditional_mode",
    "marginal_density",
    "sample_conditional",
    # Multi-layer network
    "HCRNetwork",
    "LayerSpec",
    "LayerState",
]
