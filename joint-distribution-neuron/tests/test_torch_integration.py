"""Tests for PyTorch integration wrappers."""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from hcrnn import (
    JointDensity,
    HCRNetwork,
    LayerSpec,
    build_tensor_basis,
)
from hcrnn.torch_integration import (
    HCRLayer,
    HCRNetworkModule,
    TORCH_AVAILABLE,
)


class TestTorchAvailable:
    """Test torch availability detection."""

    def test_torch_is_available(self):
        """Torch should be available since we imported it."""
        assert TORCH_AVAILABLE is True


class TestHCRLayer:
    """Tests for HCRLayer wrapper."""

    @pytest.fixture
    def fitted_density(self):
        """Create a small fitted JointDensity."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(200, 2))

        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        density.fit(X)
        return density

    def test_init_requires_fitted_density(self):
        """Should raise if density is not fitted."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)  # Not fitted

        with pytest.raises(ValueError, match="must be fitted"):
            HCRLayer(density)

    def test_init_requires_density_type(self):
        """Should raise TypeError for wrong type."""
        with pytest.raises(TypeError, match="Expected JointDensity"):
            HCRLayer("not a density")

    def test_forward_shape(self, fitted_density):
        """Forward should return correct shape."""
        layer = HCRLayer(fitted_density)

        x = torch.rand(10, 2)
        rho = layer(x)

        assert rho.shape == (10,)
        assert rho.dtype == x.dtype

    def test_forward_returns_positive(self, fitted_density):
        """Forward should return positive density values."""
        layer = HCRLayer(fitted_density)

        x = torch.rand(50, 2)
        rho = layer(x)

        assert torch.all(rho > 0)

    def test_forward_preserves_device(self, fitted_density):
        """Output should be on same device as input."""
        layer = HCRLayer(fitted_density)

        x = torch.rand(10, 2)
        rho = layer(x)

        assert rho.device == x.device

    def test_forward_matches_numpy(self, fitted_density):
        """Torch output should match numpy output."""
        layer = HCRLayer(fitted_density)

        x_np = np.random.rand(10, 2).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        rho_torch = layer(x_torch).numpy()
        rho_np = fitted_density.density(x_np, clamp=True)

        np.testing.assert_allclose(rho_torch, rho_np, rtol=1e-5)

    def test_extra_repr(self, fitted_density):
        """extra_repr should return info string."""
        layer = HCRLayer(fitted_density)
        repr_str = layer.extra_repr()

        assert "dim=" in repr_str
        assert "num_basis=" in repr_str


class TestHCRNetworkModule:
    """Tests for HCRNetworkModule wrapper."""

    @pytest.fixture
    def fitted_network(self):
        """Create a small fitted HCRNetwork."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(100, 2))
        Y = rng.uniform(0, 1, size=(100, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)
        return net

    def test_init_requires_network_type(self):
        """Should raise TypeError for wrong type."""
        with pytest.raises(TypeError, match="Expected HCRNetwork"):
            HCRNetworkModule("not a network")

    def test_forward_shape(self, fitted_network):
        """Forward should return correct shape."""
        module = HCRNetworkModule(fitted_network)

        x = torch.rand(10, 2)
        y = module(x)

        assert y.shape == (10, 2)

    def test_forward_preserves_device(self, fitted_network):
        """Output should be on same device as input."""
        module = HCRNetworkModule(fitted_network)

        x = torch.rand(10, 2)
        y = module(x)

        assert y.device == x.device

    def test_forward_matches_numpy(self, fitted_network):
        """Torch output should match numpy output."""
        module = HCRNetworkModule(fitted_network)

        x_np = np.random.rand(10, 2).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        y_torch = module(x_torch).numpy()
        y_np = fitted_network.forward(x_np)

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-4, atol=1e-4)

    def test_reverse_shape(self, fitted_network):
        """Reverse should return correct shape."""
        module = HCRNetworkModule(fitted_network)

        y = torch.rand(10, 2)
        x = module.reverse(y)

        assert x.shape == (10, 2)

    def test_reverse_matches_numpy(self, fitted_network):
        """Reverse should match numpy output."""
        module = HCRNetworkModule(fitted_network)

        y_np = np.random.rand(10, 2).astype(np.float32)
        y_torch = torch.from_numpy(y_np)

        x_torch = module.reverse(y_torch).numpy()
        x_np = fitted_network.reverse(y_np)

        np.testing.assert_allclose(x_torch, x_np, rtol=1e-4, atol=1e-4)

    def test_extra_repr(self, fitted_network):
        """extra_repr should return info string."""
        module = HCRNetworkModule(fitted_network)
        repr_str = module.extra_repr()

        assert "layers=" in repr_str
        assert "fitted=" in repr_str

    def test_network_property(self, fitted_network):
        """network property should return underlying network."""
        module = HCRNetworkModule(fitted_network)
        assert module.network is fitted_network


class TestRoundTrip:
    """Test round-trip consistency between numpy and torch."""

    def test_network_roundtrip_consistency(self):
        """Network forward through numpy and torch should match."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        module = HCRNetworkModule(net)

        # Test data
        x_np = rng.uniform(0, 1, size=(20, 2)).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Forward through both
        y_np = net.forward(x_np)
        y_torch = module(x_torch).numpy()

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-4, atol=1e-4)

        # Reverse through both
        x_recon_np = net.reverse(y_np)
        x_recon_torch = module.reverse(torch.from_numpy(y_np.astype(np.float32))).numpy()

        np.testing.assert_allclose(x_recon_torch, x_recon_np, rtol=1e-4, atol=1e-4)
