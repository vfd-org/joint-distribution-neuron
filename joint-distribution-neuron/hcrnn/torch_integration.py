"""
PyTorch integration wrappers for HCRNN models.

This module provides torch.nn.Module wrappers for JointDensity and HCRNetwork,
enabling integration with PyTorch pipelines for inference. These are inference-only
wrappers - gradients are not computed since the underlying HCR models use
non-differentiable polynomial basis functions.

Usage:
    from hcrnn import HCRNetwork
    from hcrnn.torch_integration import HCRNetworkModule

    net = HCRNetwork(specs)
    net.fit(X, Y)

    module = HCRNetworkModule(net)
    y_pred = module(x_tensor)  # Returns torch.Tensor

Note:
    Requires PyTorch to be installed: pip install torch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

# Lazy import of torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from hcrnn.joint_density import JointDensity
    from hcrnn.network import HCRNetwork


def _check_torch_available() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for torch integration. "
            "Please install it with: pip install torch"
        )


class HCRLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch Module wrapper for a single JointDensity.

    This wrapper enables using a fitted JointDensity within a PyTorch
    pipeline for inference. It handles conversion between torch.Tensor
    and numpy arrays automatically.

    Note:
        This is an inference-only wrapper. Gradients are not computed
        because JointDensity uses non-differentiable polynomial operations.

    Args:
        density: A fitted JointDensity instance
        device: Target device for output tensors (default: same as input)
        dtype: Target dtype for output tensors (default: same as input)

    Example:
        >>> from hcrnn import build_tensor_basis, JointDensity
        >>> from hcrnn.torch_integration import HCRLayer
        >>> import torch
        >>>
        >>> basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        >>> density = JointDensity(basis)
        >>> density.fit(X_train)
        >>>
        >>> layer = HCRLayer(density)
        >>> x = torch.rand(10, 2)
        >>> rho = layer(x)  # Returns density values as torch.Tensor
    """

    def __init__(
        self,
        density: JointDensity,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        _check_torch_available()
        super().__init__()

        from hcrnn.joint_density import JointDensity as JD
        if not isinstance(density, JD):
            raise TypeError(
                f"Expected JointDensity instance, got {type(density).__name__}"
            )
        if density.coeffs is None:
            raise ValueError("JointDensity must be fitted before wrapping")

        self._density = density
        self._device = device
        self._dtype = dtype

    @property
    def density(self) -> JointDensity:
        """Access the underlying JointDensity."""
        return self._density

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate density at input points.

        Args:
            x: Input tensor of shape (N, dim) with values in [0, 1]

        Returns:
            Density values as tensor of shape (N,)
        """
        # Store original device and dtype for output
        input_device = x.device
        input_dtype = x.dtype

        # Convert to numpy (move to CPU if needed)
        x_np = x.detach().cpu().numpy()

        # Evaluate density
        rho_np = self._density.density(x_np, clamp=True)

        # Convert back to torch
        rho = torch.from_numpy(rho_np.astype(np.float32))

        # Apply device/dtype (use input's if not specified)
        target_device = self._device if self._device is not None else input_device
        target_dtype = self._dtype if self._dtype is not None else input_dtype

        return rho.to(device=target_device, dtype=target_dtype)

    def extra_repr(self) -> str:
        """Extra representation for print."""
        return f"dim={self._density.dim}, num_basis={self._density.basis.num_basis}"


class HCRNetworkModule(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch Module wrapper for HCRNetwork.

    This wrapper enables using a fitted HCRNetwork within a PyTorch
    pipeline for inference. It handles conversion between torch.Tensor
    and numpy arrays automatically.

    Note:
        This is an inference-only wrapper. Gradients are not computed
        because HCRNetwork uses non-differentiable polynomial operations.

    Args:
        network: A fitted HCRNetwork instance
        device: Target device for output tensors (default: same as input)
        dtype: Target dtype for output tensors (default: same as input)

    Example:
        >>> from hcrnn import HCRNetwork, LayerSpec
        >>> from hcrnn.torch_integration import HCRNetworkModule
        >>> import torch
        >>>
        >>> specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        >>> net = HCRNetwork(specs)
        >>> net.fit(X_train, Y_train)
        >>>
        >>> module = HCRNetworkModule(net)
        >>> x = torch.rand(10, 2)
        >>> y = module(x)  # Returns torch.Tensor
    """

    def __init__(
        self,
        network: HCRNetwork,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        _check_torch_available()
        super().__init__()

        from hcrnn.network import HCRNetwork as Net
        if not isinstance(network, Net):
            raise TypeError(
                f"Expected HCRNetwork instance, got {type(network).__name__}"
            )

        self._network = network
        self._device = device
        self._dtype = dtype

    @property
    def network(self) -> HCRNetwork:
        """Access the underlying HCRNetwork."""
        return self._network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, input_dim)

        Returns:
            Output tensor of shape (N, output_dim)
        """
        # Store original device and dtype for output
        input_device = x.device
        input_dtype = x.dtype

        # Convert to numpy (move to CPU if needed)
        x_np = x.detach().cpu().numpy()

        # Forward through network
        y_np = self._network.forward(x_np)

        # Convert back to torch
        y = torch.from_numpy(y_np.astype(np.float32))

        # Apply device/dtype (use input's if not specified)
        target_device = self._device if self._device is not None else input_device
        target_dtype = self._dtype if self._dtype is not None else input_dtype

        return y.to(device=target_device, dtype=target_dtype)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reverse pass through the network.

        Args:
            y: Output tensor of shape (N, output_dim)

        Returns:
            Reconstructed input tensor of shape (N, input_dim)
        """
        # Store original device and dtype for output
        input_device = y.device
        input_dtype = y.dtype

        # Convert to numpy (move to CPU if needed)
        y_np = y.detach().cpu().numpy()

        # Reverse through network
        x_np = self._network.reverse(y_np)

        # Convert back to torch
        x = torch.from_numpy(x_np.astype(np.float32))

        # Apply device/dtype (use input's if not specified)
        target_device = self._device if self._device is not None else input_device
        target_dtype = self._dtype if self._dtype is not None else input_dtype

        return x.to(device=target_device, dtype=target_dtype)

    def extra_repr(self) -> str:
        """Extra representation for print."""
        layers_str = " -> ".join(
            f"{s.input_dim}x{s.output_dim}" for s in self._network.specs
        )
        return f"layers=[{layers_str}], fitted={self._network._is_fitted}"


__all__ = ["HCRLayer", "HCRNetworkModule", "TORCH_AVAILABLE"]
