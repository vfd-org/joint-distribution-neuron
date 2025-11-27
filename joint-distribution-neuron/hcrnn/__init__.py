"""
HCRNN: HCR-based Joint Distribution Network

A minimal prototype implementing Jarek Duda's joint distribution neuron
concept from arXiv:2405.05097, using polynomial basis representations
for bidirectional inference.

v0.4 Features:
- PyTorch integration wrappers for inference
- Model save/load utilities
- N-dimensional support with total-degree basis constraint
- Marginalization and conditioning operations
- Multi-layer networks with stacked bidirectional inference
- Learnable projections between layers
- Resonance-based regularization for coherent representations
"""

from hcrnn.basis import (
    orthonormal_poly_1d,
    build_tensor_basis,
    build_total_degree_basis,
    TensorBasis,
    TotalDegreeBasis,
    enumerate_multi_indices,
    count_total_degree_terms,
    BasisType,
)
from hcrnn.joint_density import JointDensity, ConditionalDensity
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
    build_default_network,
    build_network_from_topology,
)
from hcrnn.pipeline import Pipeline
from hcrnn.io import (
    save_density,
    load_density,
    save_network,
    load_network,
)

# Conditionally import torch integration
# Classes are always importable but raise at instantiation if torch is missing
from hcrnn.torch_integration import (
    HCRLayer,
    HCRNetworkModule,
    TORCH_AVAILABLE,
)

__version__ = "0.4.0"
__all__ = [
    # Basis
    "orthonormal_poly_1d",
    "build_tensor_basis",
    "build_total_degree_basis",
    "TensorBasis",
    "TotalDegreeBasis",
    "enumerate_multi_indices",
    "count_total_degree_terms",
    "BasisType",
    # Single neuron
    "JointDensity",
    "ConditionalDensity",
    # Conditionals (legacy API)
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
    "build_default_network",
    "build_network_from_topology",
    # Pipeline
    "Pipeline",
    # I/O utilities
    "save_density",
    "load_density",
    "save_network",
    "load_network",
    # PyTorch integration
    "HCRLayer",
    "HCRNetworkModule",
    "TORCH_AVAILABLE",
]
