"""
Model I/O utilities for saving and loading HCRNN models.

This module provides serialization utilities for JointDensity
and HCRNetwork objects. Since the basis functions contain closures
that cannot be directly pickled, we serialize the essential state
(coefficients, parameters) and reconstruct objects on load.

Usage:
    from hcrnn import save_network, load_network, save_density, load_density

    # Save and load a network
    save_network(net, "model.pkl")
    net2 = load_network("model.pkl")

    # Save and load a density
    save_density(density, "density.pkl")
    density2 = load_density("density.pkl")

Note:
    Uses pickle for serialization. Files contain Python objects and should
    only be loaded from trusted sources.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union, Dict, Any

import numpy as np

from hcrnn.basis import build_tensor_basis, build_total_degree_basis, TensorBasis, TotalDegreeBasis
from hcrnn.joint_density import JointDensity
from hcrnn.network import HCRNetwork, LayerSpec


PathLike = Union[str, Path]


def _density_to_state(density: JointDensity) -> Dict[str, Any]:
    """Extract serializable state from JointDensity."""
    basis = density.basis

    # Determine basis type and parameters
    if isinstance(basis, TotalDegreeBasis):
        basis_type = "total_degree"
        basis_params = {
            "dim": basis.dim,
            "total_degree": basis.total_degree,
        }
    else:
        # TensorBasis
        basis_type = "tensor"
        basis_params = {
            "dim": basis.dim,
            "degrees": basis.degrees,
        }

    return {
        "basis_type": basis_type,
        "basis_params": basis_params,
        "coeffs": density.coeffs.copy() if density.coeffs is not None else None,
        "_fit_data": density._fit_data.copy() if density._fit_data is not None else None,
    }


def _state_to_density(state: Dict[str, Any]) -> JointDensity:
    """Reconstruct JointDensity from serialized state."""
    basis_type = state["basis_type"]
    basis_params = state["basis_params"]

    if basis_type == "total_degree":
        basis = build_total_degree_basis(
            dim=basis_params["dim"],
            total_degree=basis_params["total_degree"],
        )
    else:
        # TensorBasis
        basis = build_tensor_basis(
            dim=basis_params["dim"],
            degrees_per_dim=list(basis_params["degrees"]),
        )

    density = JointDensity(basis)

    if state["coeffs"] is not None:
        density.coeffs = state["coeffs"].copy()

    if state["_fit_data"] is not None:
        density._fit_data = state["_fit_data"].copy()

    return density


def _network_to_state(network: HCRNetwork) -> Dict[str, Any]:
    """Extract serializable state from HCRNetwork."""
    # Save specs
    specs_data = []
    for spec in network.specs:
        specs_data.append({
            "input_dim": spec.input_dim,
            "output_dim": spec.output_dim,
            "basis_degree": spec.basis_degree,
            "resonance_decay": spec.resonance_decay,
        })

    # Save layer states
    layers_data = []
    for layer in network.layers:
        layer_state = {
            "W": layer.W.copy(),
            "bias": layer.bias.copy(),
            "input_range": (
                layer.input_range[0].copy() if layer.input_range[0] is not None else None,
                layer.input_range[1].copy() if layer.input_range[1] is not None else None,
            ),
            "output_range": (
                layer.output_range[0].copy() if layer.output_range[0] is not None else None,
                layer.output_range[1].copy() if layer.output_range[1] is not None else None,
            ),
            "joint_coeffs": layer.joint.coeffs.copy() if layer.joint.coeffs is not None else None,
            "joint_fit_data": layer.joint._fit_data.copy() if layer.joint._fit_data is not None else None,
        }
        layers_data.append(layer_state)

    return {
        "specs": specs_data,
        "layers": layers_data,
        "_is_fitted": network._is_fitted,
        "seed": 42,  # Default seed for reconstruction
    }


def _state_to_network(state: Dict[str, Any]) -> HCRNetwork:
    """Reconstruct HCRNetwork from serialized state."""
    # Reconstruct specs
    specs = []
    for spec_data in state["specs"]:
        specs.append(LayerSpec(
            input_dim=spec_data["input_dim"],
            output_dim=spec_data["output_dim"],
            basis_degree=spec_data["basis_degree"],
            resonance_decay=spec_data["resonance_decay"],
        ))

    # Create network (this initializes layers with random weights)
    network = HCRNetwork(specs, seed=state.get("seed", 42))

    # Restore layer states
    for layer, layer_data in zip(network.layers, state["layers"]):
        layer.W = layer_data["W"].copy()
        layer.bias = layer_data["bias"].copy()

        if layer_data["input_range"][0] is not None:
            layer.input_range = (
                layer_data["input_range"][0].copy(),
                layer_data["input_range"][1].copy(),
            )

        if layer_data["output_range"][0] is not None:
            layer.output_range = (
                layer_data["output_range"][0].copy(),
                layer_data["output_range"][1].copy(),
            )

        if layer_data["joint_coeffs"] is not None:
            layer.joint.coeffs = layer_data["joint_coeffs"].copy()

        if layer_data["joint_fit_data"] is not None:
            layer.joint._fit_data = layer_data["joint_fit_data"].copy()

    network._is_fitted = state["_is_fitted"]

    return network


def save_density(density: JointDensity, path: PathLike) -> None:
    """
    Save a JointDensity to a file.

    Args:
        density: JointDensity instance to save
        path: File path (will be created/overwritten)

    Raises:
        TypeError: If density is not a JointDensity instance

    Example:
        >>> from hcrnn import JointDensity, build_tensor_basis
        >>> from hcrnn import save_density, load_density
        >>>
        >>> basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        >>> density = JointDensity(basis)
        >>> density.fit(X)
        >>>
        >>> save_density(density, "my_density.pkl")
    """
    if not isinstance(density, JointDensity):
        raise TypeError(
            f"Expected JointDensity instance, got {type(density).__name__}"
        )

    state = _density_to_state(density)

    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_density(path: PathLike) -> JointDensity:
    """
    Load a JointDensity from a file.

    Args:
        path: Path to the saved density file

    Returns:
        Loaded JointDensity instance

    Raises:
        TypeError: If loaded object is not a valid density state
        FileNotFoundError: If path does not exist

    Example:
        >>> from hcrnn import load_density
        >>>
        >>> density = load_density("my_density.pkl")
        >>> rho = density.density(x)
    """
    path = Path(path)
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if not isinstance(state, dict) or "basis_type" not in state:
        raise TypeError(
            "Invalid density file format"
        )

    return _state_to_density(state)


def save_network(network: HCRNetwork, path: PathLike) -> None:
    """
    Save an HCRNetwork to a file.

    Args:
        network: HCRNetwork instance to save
        path: File path (will be created/overwritten)

    Raises:
        TypeError: If network is not an HCRNetwork instance

    Example:
        >>> from hcrnn import HCRNetwork, LayerSpec, save_network
        >>>
        >>> specs = [LayerSpec(input_dim=2, output_dim=2)]
        >>> net = HCRNetwork(specs)
        >>> net.fit(X, Y)
        >>>
        >>> save_network(net, "my_network.pkl")
    """
    if not isinstance(network, HCRNetwork):
        raise TypeError(
            f"Expected HCRNetwork instance, got {type(network).__name__}"
        )

    state = _network_to_state(network)

    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_network(path: PathLike) -> HCRNetwork:
    """
    Load an HCRNetwork from a file.

    Args:
        path: Path to the saved network file

    Returns:
        Loaded HCRNetwork instance

    Raises:
        TypeError: If loaded object is not a valid network state
        FileNotFoundError: If path does not exist

    Example:
        >>> from hcrnn import load_network
        >>>
        >>> net = load_network("my_network.pkl")
        >>> Y_pred = net.forward(X)
    """
    path = Path(path)
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if not isinstance(state, dict) or "specs" not in state:
        raise TypeError(
            "Invalid network file format"
        )

    return _state_to_network(state)


__all__ = ["save_density", "load_density", "save_network", "load_network"]
