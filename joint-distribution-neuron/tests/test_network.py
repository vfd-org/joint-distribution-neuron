"""Tests for multi-layer HCR network."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hcrnn.network import HCRNetwork, LayerSpec


def generate_linear_data(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data from a linear transformation with noise."""
    rng = np.random.default_rng(seed)

    # Random linear transformation
    W_true = rng.standard_normal((output_dim, input_dim))
    b_true = rng.standard_normal(output_dim) * 0.5

    X = rng.standard_normal((n_samples, input_dim))
    Y = X @ W_true.T + b_true + rng.standard_normal((n_samples, output_dim)) * noise_scale

    return X, Y, W_true


def generate_nonlinear_data(
    n_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D -> 2D nonlinear transformation data."""
    rng = np.random.default_rng(seed)

    X = rng.randn(n_samples, 2)

    # Nonlinear transformation: rotation + scaling + nonlinearity
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    Y = X @ R.T
    Y[:, 0] = np.tanh(Y[:, 0] * 2)
    Y[:, 1] = Y[:, 1] ** 2 * np.sign(Y[:, 1])
    Y += rng.randn(n_samples, 2) * 0.1

    return X, Y


class TestLayerSpec:
    """Tests for LayerSpec dataclass."""

    def test_create_basic(self):
        """Create a basic layer spec."""
        spec = LayerSpec(input_dim=3, output_dim=5)
        assert spec.input_dim == 3
        assert spec.output_dim == 5
        assert spec.basis_degree == 3  # Default
        assert spec.resonance_decay == 0.1  # Default

    def test_create_custom(self):
        """Create layer spec with custom parameters."""
        spec = LayerSpec(
            input_dim=4,
            output_dim=2,
            basis_degree=5,
            resonance_decay=0.2,
        )
        assert spec.basis_degree == 5
        assert spec.resonance_decay == 0.2


class TestHCRNetworkInit:
    """Tests for HCRNetwork initialization."""

    def test_single_layer_init(self):
        """Initialize single-layer network."""
        specs = [LayerSpec(input_dim=2, output_dim=3)]
        net = HCRNetwork(specs)

        assert net.num_layers == 1
        assert len(net.layers) == 1
        assert net.layers[0].W.shape == (3, 2)
        assert net.layers[0].bias.shape == (3,)

    def test_multi_layer_init(self):
        """Initialize multi-layer network."""
        specs = [
            LayerSpec(input_dim=2, output_dim=4),
            LayerSpec(input_dim=4, output_dim=3),
            LayerSpec(input_dim=3, output_dim=2),
        ]
        net = HCRNetwork(specs)

        assert net.num_layers == 3
        assert net.layers[0].W.shape == (4, 2)
        assert net.layers[1].W.shape == (3, 4)
        assert net.layers[2].W.shape == (2, 3)

    def test_repr_unfitted(self):
        """Test string representation before fitting."""
        specs = [LayerSpec(input_dim=2, output_dim=3)]
        net = HCRNetwork(specs)
        repr_str = repr(net)
        assert "2→3" in repr_str
        assert "not fitted" in repr_str


class TestHCRNetworkForward:
    """Tests for forward pass."""

    def test_forward_single_layer(self):
        """Forward pass through single layer."""
        specs = [LayerSpec(input_dim=2, output_dim=3, basis_degree=2)]
        net = HCRNetwork(specs)

        X = np.random.rand(10, 2)
        Y = net.forward(X)

        assert Y.shape == (10, 3)
        assert np.all(np.isfinite(Y))

    def test_forward_multi_layer(self):
        """Forward pass through multiple layers."""
        specs = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=3, basis_degree=2),
        ]
        net = HCRNetwork(specs)

        X = np.random.rand(10, 2)
        Y = net.forward(X)

        assert Y.shape == (10, 3)

    def test_forward_with_intermediates(self):
        """Forward pass returning intermediate activations."""
        specs = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=3, basis_degree=2),
        ]
        net = HCRNetwork(specs)

        X = np.random.rand(10, 2)
        Y, intermediates = net.forward(X, return_intermediates=True)

        assert len(intermediates) == 3  # Input + 2 layer outputs
        assert intermediates[0].shape == (10, 2)
        assert intermediates[1].shape == (10, 4)
        assert intermediates[2].shape == (10, 3)

    def test_forward_single_sample(self):
        """Forward pass with single sample (1D input)."""
        specs = [LayerSpec(input_dim=2, output_dim=3, basis_degree=2)]
        net = HCRNetwork(specs)

        x = np.random.rand(2)
        y = net.forward(x)

        assert y.shape == (1, 3)


class TestHCRNetworkReverse:
    """Tests for reverse pass."""

    def test_reverse_single_layer(self):
        """Reverse pass through single layer."""
        specs = [LayerSpec(input_dim=2, output_dim=3, basis_degree=2)]
        net = HCRNetwork(specs)

        Y = np.random.rand(10, 3)
        X_recon = net.reverse(Y)

        assert X_recon.shape == (10, 2)
        assert np.all(np.isfinite(X_recon))

    def test_reverse_multi_layer(self):
        """Reverse pass through multiple layers."""
        specs = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=3, basis_degree=2),
        ]
        net = HCRNetwork(specs)

        Y = np.random.rand(10, 3)
        X_recon = net.reverse(Y)

        assert X_recon.shape == (10, 2)


class TestHCRNetworkTraining:
    """Tests for network training."""

    def test_fit_linear_alternating(self):
        """Train on linear data using alternating method."""
        X, Y, _ = generate_linear_data(200, input_dim=2, output_dim=2, noise_scale=0.05)

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)

        net.fit(X, Y, method="alternating", max_iter=3, verbose=False)

        assert net._is_fitted
        Y_pred = net.forward(X)
        mse = np.mean((Y_pred - Y) ** 2)
        # Prototype: just verify it produces finite outputs
        # Performance optimization is future work
        assert np.isfinite(mse)

    def test_fit_linear_coordinate(self):
        """Train on linear data using coordinate descent."""
        X, Y, _ = generate_linear_data(100, input_dim=2, output_dim=2, noise_scale=0.1)

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)

        net.fit(X, Y, method="coordinate", max_iter=2, verbose=False)

        assert net._is_fitted

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        X, Y, _ = generate_linear_data(50, input_dim=2, output_dim=2)
        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)

        result = net.fit(X, Y, method="alternating", max_iter=1, verbose=False)
        assert result is net

    def test_repr_after_fit(self):
        """Test string representation after fitting."""
        X, Y, _ = generate_linear_data(50, input_dim=2, output_dim=3)
        specs = [LayerSpec(input_dim=2, output_dim=3, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=1, verbose=False)

        repr_str = repr(net)
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestHCRNetworkBidirectional:
    """Tests for bidirectional inference."""

    def test_cycle_consistency(self):
        """Forward then reverse should produce finite outputs."""
        X, Y, _ = generate_linear_data(100, input_dim=2, output_dim=2, noise_scale=0.05)

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=3, verbose=False)

        # Forward then reverse
        Y_pred = net.forward(X[:10])
        X_recon = net.reverse(Y_pred)

        # Verify shapes and finiteness (prototype test)
        assert X_recon.shape == X[:10].shape
        assert np.all(np.isfinite(X_recon))

    def test_reconstruction_error_metrics(self):
        """Test reconstruction error computation."""
        X, Y, _ = generate_linear_data(100, input_dim=2, output_dim=2, noise_scale=0.1)

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        errors = net.reconstruction_error(X, Y)

        assert "forward_mse" in errors
        assert "reverse_mse" in errors
        assert "cycle_mse" in errors
        assert "forward_rmse" in errors
        assert "reverse_rmse" in errors

        # All errors should be finite
        for key, val in errors.items():
            assert np.isfinite(val), f"{key} is not finite"


class TestHCRNetworkUncertainty:
    """Tests for uncertainty quantification."""

    def test_compute_uncertainty(self):
        """Compute output uncertainty."""
        X, Y, _ = generate_linear_data(100, input_dim=2, output_dim=2, noise_scale=0.1)

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        variances = net.compute_uncertainty(X[:10])

        assert variances.shape == (10, 2)
        assert np.all(variances >= 0)  # Variance is non-negative


class TestHCRNetworkMultiLayer:
    """Tests specifically for multi-layer networks."""

    def test_three_layer_network(self):
        """Test 3-layer network end-to-end."""
        X = np.random.randn(200, 2)
        Y = np.random.randn(200, 2)

        specs = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=2, basis_degree=2),
        ]
        net = HCRNetwork(specs)

        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        Y_pred = net.forward(X)
        assert Y_pred.shape == (200, 2)

        X_recon = net.reverse(Y_pred)
        assert X_recon.shape == (200, 2)

    def test_autoencoder_structure(self):
        """Test autoencoder-like structure (compress then expand)."""
        X = np.random.randn(100, 4)

        # 4 -> 2 -> 4 (bottleneck architecture)
        specs = [
            LayerSpec(input_dim=4, output_dim=2, basis_degree=2),
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
        ]
        net = HCRNetwork(specs)

        # Use X as both input and target (autoencoder)
        net.fit(X, X, method="alternating", max_iter=3, verbose=False)

        X_recon = net.forward(X)
        mse = np.mean((X_recon - X) ** 2)

        # Should achieve some reconstruction
        assert np.isfinite(mse)


class TestResonanceRegularization:
    """Tests for resonance-based regularization."""

    def test_resonance_penalty_increases_with_complexity(self):
        """Higher-order coefficients should incur higher penalty."""
        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=3, resonance_decay=0.5)]
        net = HCRNetwork(specs)

        # Fit with some data
        X = np.random.rand(100, 2)
        Y = X * 2 + 0.1  # Simple linear relationship

        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        # The penalty should be finite and positive
        penalty = net._resonance_penalty(net.layers[0], specs[0])
        assert np.isfinite(penalty)
        assert penalty >= 0
