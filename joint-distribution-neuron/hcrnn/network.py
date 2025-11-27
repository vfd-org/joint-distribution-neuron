"""
Multi-layer HCR Network (HCRNN).

This module implements a hierarchical network of joint-density neurons,
enabling:
- Stacked bidirectional inference across multiple layers
- Learned linear projections between layers
- Resonance-based regularization for coherent representations
- Forward and reverse passes through the network

The key insight: each layer maintains a joint distribution over its
input-output space. This allows uncertainty to propagate bidirectionally,
and constraints at any layer can influence all other layers.

Architecture:
    Input → [W₁] → JointDensity₁ → [W₂] → JointDensity₂ → ... → Output

Each layer:
    1. Projects input via learnable matrix W
    2. Normalizes to [0,1] range
    3. Models joint distribution p(input, output) at that layer
    4. Can infer in either direction

Reference: Duda, J. (2024). arXiv:2405.05097
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import optimize

from hcrnn.basis import build_tensor_basis, TensorBasis
from hcrnn.joint_density import JointDensity
from hcrnn.conditionals import (
    conditional_expectation,
    conditional_density,
    conditional_variance,
)


@dataclass
class LayerSpec:
    """Specification for a single HCRNN layer.

    Attributes:
        input_dim: Dimensionality of input to this layer
        output_dim: Dimensionality of output from this layer
        basis_degree: Polynomial degree for the joint density basis
        resonance_decay: Decay rate for high-frequency basis regularization
    """
    input_dim: int
    output_dim: int
    basis_degree: int = 3
    resonance_decay: float = 0.1


@dataclass
class LayerState:
    """Runtime state for a single HCRNN layer.

    Attributes:
        W: Projection matrix (output_dim, input_dim)
        bias: Bias vector (output_dim,)
        joint: JointDensity model for this layer
        input_range: (min, max) for input normalization
        output_range: (min, max) for output normalization
    """
    W: np.ndarray
    bias: np.ndarray
    joint: JointDensity
    basis: TensorBasis
    input_range: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (None, None))
    output_range: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (None, None))


class HCRNetwork:
    """
    Multi-layer Hierarchical Correlation Reconstruction Network.

    A stack of joint-density neurons with learnable projections,
    supporting bidirectional inference and resonance-based regularization.

    The network learns to model p(x₀, x₁, ..., xₙ) implicitly through
    factorized layers: p(x₀, x₁) × p(x₁, x₂) × ... × p(xₙ₋₁, xₙ)

    This factorization allows efficient inference in both directions
    while capturing complex nonlinear relationships.

    Example:
        >>> specs = [
        ...     LayerSpec(input_dim=2, output_dim=4, basis_degree=3),
        ...     LayerSpec(input_dim=4, output_dim=2, basis_degree=3),
        ... ]
        >>> net = HCRNetwork(specs)
        >>> net.fit(X_train, Y_train)
        >>> Y_pred = net.forward(X_test)
        >>> X_reconstructed = net.reverse(Y_pred)
    """

    def __init__(
        self,
        layer_specs: Sequence[LayerSpec],
        seed: int = 42,
    ):
        """
        Initialize HCR Network.

        Args:
            layer_specs: List of LayerSpec defining each layer
            seed: Random seed for reproducibility
        """
        self.specs = list(layer_specs)
        self.num_layers = len(self.specs)
        self._rng = np.random.default_rng(seed)

        # Initialize layers
        self.layers: list[LayerState] = []
        self._init_layers()

        # Training state
        self._is_fitted = False

    def _init_layers(self) -> None:
        """Initialize layer parameters with small random values."""
        for spec in self.specs:
            # Xavier-like initialization for projection matrix
            scale = np.sqrt(2.0 / (spec.input_dim + spec.output_dim))
            W = self._rng.normal(0, scale, size=(spec.output_dim, spec.input_dim))
            bias = np.zeros(spec.output_dim)

            # Build basis for joint density over (input, output) space
            joint_dim = spec.input_dim + spec.output_dim
            basis = build_tensor_basis(dim=joint_dim, degrees_per_dim=spec.basis_degree)

            # Create unfitted JointDensity
            joint = JointDensity(basis)

            layer = LayerState(
                W=W,
                bias=bias,
                joint=joint,
                basis=basis,
            )
            self.layers.append(layer)

    def _normalize_to_unit(
        self,
        X: np.ndarray,
        range_min: np.ndarray | None,
        range_max: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize data to [0, 1] range.

        Returns normalized data and the (min, max) used.
        """
        if range_min is None or range_max is None:
            range_min = X.min(axis=0) - 1e-6
            range_max = X.max(axis=0) + 1e-6

        # Avoid division by zero
        span = range_max - range_min
        span = np.maximum(span, 1e-8)

        X_norm = (X - range_min) / span
        X_norm = np.clip(X_norm, 0.001, 0.999)

        return X_norm, range_min, range_max

    def _denormalize_from_unit(
        self,
        X_norm: np.ndarray,
        range_min: np.ndarray,
        range_max: np.ndarray,
    ) -> np.ndarray:
        """Convert from [0, 1] back to original range."""
        span = range_max - range_min
        return X_norm * span + range_min

    def _forward_layer(
        self,
        layer: LayerState,
        X: np.ndarray,
        use_expectation: bool = True,
    ) -> np.ndarray:
        """
        Forward pass through a single layer.

        Args:
            layer: LayerState to use
            X: Input array of shape (N, input_dim)
            use_expectation: If True, use E[output | input]. If False, use mode.

        Returns:
            Output array of shape (N, output_dim)
        """
        N = X.shape[0]
        spec = self.specs[self.layers.index(layer)]

        # Apply linear projection
        Z = X @ layer.W.T + layer.bias

        # If joint is not fitted, just return the linear transform
        if layer.joint.coeffs is None:
            return Z

        # Normalize input and linear output for joint density
        X_norm, _, _ = self._normalize_to_unit(X, *layer.input_range)
        Z_norm, z_min, z_max = self._normalize_to_unit(Z, *layer.output_range)

        # Use joint density for refined output
        input_indices = list(range(spec.input_dim))
        output_indices = list(range(spec.input_dim, spec.input_dim + spec.output_dim))

        Y_norm = np.zeros((N, spec.output_dim))

        for i in range(N):
            for j, out_idx in enumerate(output_indices):
                if use_expectation:
                    Y_norm[i, j] = conditional_expectation(
                        layer.joint,
                        target_index=out_idx,
                        given_indices=input_indices,
                        given_values=X_norm[i].tolist(),
                        grid_size=50,
                    )
                else:
                    # Use mode (MAP estimate)
                    from hcrnn.conditionals import conditional_mode
                    Y_norm[i, j] = conditional_mode(
                        layer.joint,
                        target_index=out_idx,
                        given_indices=input_indices,
                        given_values=X_norm[i].tolist(),
                        grid_size=50,
                    )

        # Denormalize output
        Y = self._denormalize_from_unit(Y_norm, z_min, z_max)

        return Y

    def _reverse_layer(
        self,
        layer: LayerState,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Reverse pass through a single layer: infer input from output.

        Args:
            layer: LayerState to use
            Y: Output array of shape (N, output_dim)

        Returns:
            Reconstructed input array of shape (N, input_dim)
        """
        N = Y.shape[0]
        spec = self.specs[self.layers.index(layer)]

        # If joint is not fitted, use pseudo-inverse
        if layer.joint.coeffs is None:
            W_pinv = np.linalg.pinv(layer.W)
            return (Y - layer.bias) @ W_pinv.T

        # Normalize output
        Y_norm, _, _ = self._normalize_to_unit(Y, *layer.output_range)

        input_indices = list(range(spec.input_dim))
        output_indices = list(range(spec.input_dim, spec.input_dim + spec.output_dim))

        X_norm = np.zeros((N, spec.input_dim))

        for i in range(N):
            for j, in_idx in enumerate(input_indices):
                X_norm[i, j] = conditional_expectation(
                    layer.joint,
                    target_index=in_idx,
                    given_indices=output_indices,
                    given_values=Y_norm[i].tolist(),
                    grid_size=50,
                )

        # Denormalize input
        X = self._denormalize_from_unit(X_norm, *layer.input_range)

        return X

    def forward(
        self,
        X: np.ndarray,
        return_intermediates: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Forward pass through entire network.

        Args:
            X: Input array of shape (N, input_dim)
            return_intermediates: If True, also return activations at each layer

        Returns:
            Output array, or (output, [intermediate activations])
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        intermediates = [X]
        current = X

        for layer in self.layers:
            current = self._forward_layer(layer, current)
            intermediates.append(current)

        if return_intermediates:
            return current, intermediates
        return current

    def reverse(
        self,
        Y: np.ndarray,
        return_intermediates: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Reverse pass through entire network: infer input from output.

        Args:
            Y: Output array of shape (N, output_dim)
            return_intermediates: If True, also return activations at each layer

        Returns:
            Reconstructed input array, or (input, [intermediate activations])
        """
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        intermediates = [Y]
        current = Y

        # Go through layers in reverse
        for layer in reversed(self.layers):
            current = self._reverse_layer(layer, current)
            intermediates.append(current)

        if return_intermediates:
            return current, intermediates
        return current

    def compute_uncertainty(
        self,
        X: np.ndarray,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Compute output uncertainty (variance) at a given layer.

        Args:
            X: Input array of shape (N, input_dim)
            layer_idx: Which layer to compute uncertainty at (-1 for final)

        Returns:
            Variance array of shape (N, output_dim_at_layer)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Forward to the target layer
        current = X
        target_idx = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx

        for i, layer in enumerate(self.layers):
            if i < target_idx:
                current = self._forward_layer(layer, current)
            elif i == target_idx:
                layer = self.layers[i]
                spec = self.specs[i]

                if layer.joint.coeffs is None:
                    # No joint fitted, return zeros
                    return np.zeros((X.shape[0], spec.output_dim))

                N = current.shape[0]
                X_norm, _, _ = self._normalize_to_unit(current, *layer.input_range)

                input_indices = list(range(spec.input_dim))
                output_indices = list(range(spec.input_dim, spec.input_dim + spec.output_dim))

                variances = np.zeros((N, spec.output_dim))

                for i in range(N):
                    for j, out_idx in enumerate(output_indices):
                        variances[i, j] = conditional_variance(
                            layer.joint,
                            target_index=out_idx,
                            given_indices=input_indices,
                            given_values=X_norm[i].tolist(),
                            grid_size=50,
                        )

                return variances

        return np.zeros((X.shape[0], self.specs[-1].output_dim))

    def _resonance_penalty(self, layer: LayerState, spec: LayerSpec) -> float:
        """
        Compute resonance-based regularization penalty.

        Penalizes high-frequency (high-order) polynomial coefficients
        to encourage smooth, coherent representations.

        This implements the "coherence filter" from VFD theory:
        higher-order moments should decay, favoring stable low-frequency
        patterns that represent robust correlations.
        """
        if layer.joint.coeffs is None:
            return 0.0

        coeffs = layer.joint.coeffs
        multi_indices = layer.basis.multi_indices

        penalty = 0.0
        for j, idx in enumerate(multi_indices):
            # Total degree of this basis function
            total_degree = sum(idx)
            if total_degree > 0:
                # Exponential decay penalty for higher orders
                weight = np.exp(spec.resonance_decay * total_degree)
                penalty += weight * coeffs[j] ** 2

        return penalty

    def _fit_layer_joint(
        self,
        layer: LayerState,
        spec: LayerSpec,
        X_input: np.ndarray,
        X_output: np.ndarray,
    ) -> None:
        """Fit the joint density for a single layer."""
        # Normalize both input and output
        X_in_norm, in_min, in_max = self._normalize_to_unit(X_input, None, None)
        X_out_norm, out_min, out_max = self._normalize_to_unit(X_output, None, None)

        # Store ranges for inference
        layer.input_range = (in_min, in_max)
        layer.output_range = (out_min, out_max)

        # Concatenate for joint density
        X_joint = np.hstack([X_in_norm, X_out_norm])

        # Fit joint density
        layer.joint.fit(X_joint)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        method: str = "alternating",
        max_iter: int = 10,
        verbose: bool = True,
    ) -> HCRNetwork:
        """
        Train the network on input-output pairs.

        Uses a layerwise approach:
        1. Initialize with linear regression targets
        2. Alternately optimize projections W and fit joint densities

        Args:
            X: Input data of shape (N, input_dim)
            Y: Target output of shape (N, output_dim)
            method: Training method ("alternating", "cmaes", or "coordinate")
            max_iter: Maximum iterations for optimization
            verbose: Print progress

        Returns:
            self, for method chaining
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if verbose:
            print(f"Training HCRNN with {self.num_layers} layers")
            print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

        if method == "alternating":
            self._fit_alternating(X, Y, max_iter, verbose)
        elif method == "cmaes":
            self._fit_cmaes(X, Y, max_iter, verbose)
        elif method == "coordinate":
            self._fit_coordinate(X, Y, max_iter, verbose)
        else:
            raise ValueError(f"Unknown training method: {method}")

        self._is_fitted = True
        return self

    def _fit_alternating(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iter: int,
        verbose: bool,
    ) -> None:
        """
        Alternating least squares training.

        Strategy:
        1. Initialize intermediate targets via linear interpolation
        2. Fit each layer's projection W via least squares
        3. Fit each layer's joint density
        4. Repeat, using joint density outputs as new targets
        """
        N = X.shape[0]

        # Initialize intermediate targets via linear interpolation
        targets = [X]
        for i, spec in enumerate(self.specs):
            if i == len(self.specs) - 1:
                targets.append(Y)
            else:
                # Interpolate between input and output
                alpha = (i + 1) / len(self.specs)
                # Simple dimension matching via padding/truncation
                target_dim = spec.output_dim
                if X.shape[1] >= target_dim and Y.shape[1] >= target_dim:
                    interp = (1 - alpha) * X[:, :target_dim] + alpha * Y[:, :target_dim]
                else:
                    interp = self._rng.standard_normal((N, target_dim)) * 0.1
                targets.append(interp)

        for iteration in range(max_iter):
            total_loss = 0.0

            for i, (layer, spec) in enumerate(zip(self.layers, self.specs)):
                X_layer = targets[i]
                Y_layer = targets[i + 1]

                # Fit projection W via least squares
                # Y = X @ W.T + b  =>  W.T = pinv(X) @ Y
                X_aug = np.hstack([X_layer, np.ones((N, 1))])
                W_aug, residuals, rank, s = np.linalg.lstsq(X_aug, Y_layer, rcond=None)

                layer.W = W_aug[:-1, :].T
                layer.bias = W_aug[-1, :]

                # Compute linear output
                Z = X_layer @ layer.W.T + layer.bias

                # Fit joint density
                self._fit_layer_joint(layer, spec, X_layer, Z)

                # Compute loss
                layer_loss = np.mean((Z - Y_layer) ** 2)
                total_loss += layer_loss

                # Add resonance penalty
                total_loss += 0.01 * self._resonance_penalty(layer, spec)

            if verbose:
                print(f"  Iteration {iteration + 1}/{max_iter}, Loss: {total_loss:.6f}")

            # Update intermediate targets using joint density expectations
            if iteration < max_iter - 1:
                current = X
                for i, layer in enumerate(self.layers[:-1]):
                    current = self._forward_layer(layer, current, use_expectation=True)
                    # Blend with original targets
                    alpha = 0.5
                    targets[i + 1] = alpha * current + (1 - alpha) * targets[i + 1]

    def _fit_cmaes(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iter: int,
        verbose: bool,
    ) -> None:
        """
        CMA-ES based training for projection matrices.

        Uses covariance matrix adaptation evolution strategy
        to optimize the projection parameters.
        """
        # Collect all parameters into a single vector
        def pack_params() -> np.ndarray:
            params = []
            for layer in self.layers:
                params.extend(layer.W.ravel())
                params.extend(layer.bias)
            return np.array(params)

        def unpack_params(params: np.ndarray) -> None:
            idx = 0
            for layer, spec in zip(self.layers, self.specs):
                w_size = spec.output_dim * spec.input_dim
                layer.W = params[idx:idx + w_size].reshape(spec.output_dim, spec.input_dim)
                idx += w_size
                layer.bias = params[idx:idx + spec.output_dim]
                idx += spec.output_dim

        def objective(params: np.ndarray) -> float:
            unpack_params(params)

            # Fit joint densities with current W
            current = X
            for layer, spec in zip(self.layers, self.specs):
                Z = current @ layer.W.T + layer.bias
                self._fit_layer_joint(layer, spec, current, Z)
                current = Z

            # Compute reconstruction loss
            Y_pred = self.forward(X)
            mse = np.mean((Y_pred - Y) ** 2)

            # Add resonance penalty
            penalty = sum(
                self._resonance_penalty(layer, spec)
                for layer, spec in zip(self.layers, self.specs)
            )

            return mse + 0.01 * penalty

        # Initial parameters
        x0 = pack_params()
        sigma0 = 0.5

        if verbose:
            print("  Running CMA-ES optimization...")

        # Use scipy's minimize with bounds as a simpler alternative
        # (Full CMA-ES would require additional library)
        result = optimize.minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': max_iter * 10, 'disp': verbose},
        )

        unpack_params(result.x)

        # Final joint density fit
        current = X
        for layer, spec in zip(self.layers, self.specs):
            Z = current @ layer.W.T + layer.bias
            self._fit_layer_joint(layer, spec, current, Z)
            current = Z

    def _fit_coordinate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iter: int,
        verbose: bool,
    ) -> None:
        """
        Coordinate descent training.

        Optimizes one parameter at a time while holding others fixed.
        Simple but can be slow for many parameters.
        """
        def compute_loss() -> float:
            # Forward pass
            current = X
            for layer, spec in zip(self.layers, self.specs):
                Z = current @ layer.W.T + layer.bias
                self._fit_layer_joint(layer, spec, current, Z)
                current = Z

            Y_pred = current
            mse = np.mean((Y_pred - Y) ** 2)

            penalty = sum(
                self._resonance_penalty(layer, spec)
                for layer, spec in zip(self.layers, self.specs)
            )

            return mse + 0.01 * penalty

        best_loss = compute_loss()
        step_size = 0.1

        for iteration in range(max_iter):
            improved = False

            for layer_idx, (layer, spec) in enumerate(zip(self.layers, self.specs)):
                # Optimize W elements
                for i in range(spec.output_dim):
                    for j in range(spec.input_dim):
                        original = layer.W[i, j]

                        # Try positive step
                        layer.W[i, j] = original + step_size
                        loss_plus = compute_loss()

                        # Try negative step
                        layer.W[i, j] = original - step_size
                        loss_minus = compute_loss()

                        # Keep best
                        if loss_plus < best_loss and loss_plus <= loss_minus:
                            layer.W[i, j] = original + step_size
                            best_loss = loss_plus
                            improved = True
                        elif loss_minus < best_loss:
                            layer.W[i, j] = original - step_size
                            best_loss = loss_minus
                            improved = True
                        else:
                            layer.W[i, j] = original

                # Optimize bias
                for i in range(spec.output_dim):
                    original = layer.bias[i]

                    layer.bias[i] = original + step_size
                    loss_plus = compute_loss()

                    layer.bias[i] = original - step_size
                    loss_minus = compute_loss()

                    if loss_plus < best_loss and loss_plus <= loss_minus:
                        layer.bias[i] = original + step_size
                        best_loss = loss_plus
                        improved = True
                    elif loss_minus < best_loss:
                        layer.bias[i] = original - step_size
                        best_loss = loss_minus
                        improved = True
                    else:
                        layer.bias[i] = original

            if verbose:
                print(f"  Iteration {iteration + 1}/{max_iter}, Loss: {best_loss:.6f}")

            # Reduce step size if no improvement
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break

    def reconstruction_error(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        Compute various reconstruction error metrics.

        Args:
            X: Input data
            Y: Target output data

        Returns:
            Dictionary with error metrics
        """
        Y_pred = self.forward(X)
        X_recon = self.reverse(Y_pred)

        forward_mse = np.mean((Y_pred - Y) ** 2)
        reverse_mse = np.mean((X_recon - X) ** 2)
        cycle_mse = np.mean((self.forward(X_recon) - Y_pred) ** 2)

        return {
            "forward_mse": forward_mse,
            "reverse_mse": reverse_mse,
            "cycle_mse": cycle_mse,
            "forward_rmse": np.sqrt(forward_mse),
            "reverse_rmse": np.sqrt(reverse_mse),
        }

    def __repr__(self) -> str:
        layers_str = " → ".join(
            f"{s.input_dim}→{s.output_dim}" for s in self.specs
        )
        status = "fitted" if self._is_fitted else "not fitted"
        return f"HCRNetwork([{layers_str}], {status})"
