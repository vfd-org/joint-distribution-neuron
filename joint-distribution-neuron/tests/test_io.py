"""Tests for model I/O utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hcrnn import (
    JointDensity,
    HCRNetwork,
    LayerSpec,
    build_tensor_basis,
    save_density,
    load_density,
    save_network,
    load_network,
)


class TestSaveDensity:
    """Tests for save_density function."""

    def test_save_density_creates_file(self, tmp_path):
        """save_density should create a file."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        density.fit(rng.uniform(0, 1, size=(100, 2)))

        filepath = tmp_path / "density.pkl"
        save_density(density, filepath)

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_save_density_rejects_wrong_type(self, tmp_path):
        """save_density should reject non-JointDensity objects."""
        filepath = tmp_path / "bad.pkl"

        with pytest.raises(TypeError, match="Expected JointDensity"):
            save_density("not a density", filepath)

        with pytest.raises(TypeError, match="Expected JointDensity"):
            save_density({"a": 1}, filepath)

    def test_save_density_accepts_string_path(self, tmp_path):
        """save_density should accept string paths."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        density.fit(rng.uniform(0, 1, size=(100, 2)))

        filepath = str(tmp_path / "density.pkl")
        save_density(density, filepath)

        assert Path(filepath).exists()


class TestLoadDensity:
    """Tests for load_density function."""

    def test_load_density_returns_correct_type(self, tmp_path):
        """load_density should return JointDensity."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        density.fit(rng.uniform(0, 1, size=(100, 2)))

        filepath = tmp_path / "density.pkl"
        save_density(density, filepath)

        loaded = load_density(filepath)
        assert isinstance(loaded, JointDensity)

    def test_load_density_preserves_coefficients(self, tmp_path):
        """Loaded density should have same coefficients."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        density.fit(rng.uniform(0, 1, size=(100, 2)))

        filepath = tmp_path / "density.pkl"
        save_density(density, filepath)

        loaded = load_density(filepath)
        np.testing.assert_array_equal(loaded.coeffs, density.coeffs)

    def test_load_density_preserves_predictions(self, tmp_path):
        """Loaded density should give same predictions."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        density.fit(rng.uniform(0, 1, size=(200, 2)))

        filepath = tmp_path / "density.pkl"
        save_density(density, filepath)

        loaded = load_density(filepath)

        # Test predictions
        X_test = rng.uniform(0, 1, size=(20, 2))
        rho_original = density.density(X_test)
        rho_loaded = loaded.density(X_test)

        np.testing.assert_allclose(rho_loaded, rho_original, rtol=1e-10)

    def test_load_density_file_not_found(self, tmp_path):
        """load_density should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_density(tmp_path / "nonexistent.pkl")


class TestSaveNetwork:
    """Tests for save_network function."""

    def test_save_network_creates_file(self, tmp_path):
        """save_network should create a file."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        filepath = tmp_path / "network.pkl"
        save_network(net, filepath)

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_save_network_rejects_wrong_type(self, tmp_path):
        """save_network should reject non-HCRNetwork objects."""
        filepath = tmp_path / "bad.pkl"

        with pytest.raises(TypeError, match="Expected HCRNetwork"):
            save_network("not a network", filepath)

        with pytest.raises(TypeError, match="Expected HCRNetwork"):
            save_network({"a": 1}, filepath)


class TestLoadNetwork:
    """Tests for load_network function."""

    def test_load_network_returns_correct_type(self, tmp_path):
        """load_network should return HCRNetwork."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        filepath = tmp_path / "network.pkl"
        save_network(net, filepath)

        loaded = load_network(filepath)
        assert isinstance(loaded, HCRNetwork)

    def test_load_network_preserves_structure(self, tmp_path):
        """Loaded network should have same structure."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        filepath = tmp_path / "network.pkl"
        save_network(net, filepath)

        loaded = load_network(filepath)

        assert loaded.num_layers == net.num_layers
        assert loaded._is_fitted == net._is_fitted

    def test_load_network_preserves_predictions(self, tmp_path):
        """Loaded network should give same predictions."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        filepath = tmp_path / "network.pkl"
        save_network(net, filepath)

        loaded = load_network(filepath)

        # Test predictions
        X_test = rng.uniform(0, 1, size=(10, 2))
        Y_original = net.forward(X_test)
        Y_loaded = loaded.forward(X_test)

        np.testing.assert_allclose(Y_loaded, Y_original, rtol=1e-10)

    def test_load_network_file_not_found(self, tmp_path):
        """load_network should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_network(tmp_path / "nonexistent.pkl")


class TestRoundTrip:
    """Round-trip save/load tests."""

    def test_density_roundtrip(self, tmp_path):
        """Save and load density should be equivalent."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(300, 2))
        density.fit(X)

        filepath = tmp_path / "density.pkl"
        save_density(density, filepath)
        loaded = load_density(filepath)

        # Check structure
        assert loaded.dim == density.dim
        assert loaded.basis.num_basis == density.basis.num_basis

        # Check coefficients
        np.testing.assert_array_equal(loaded.coeffs, density.coeffs)

        # Check predictions at multiple points
        X_test = rng.uniform(0, 1, size=(50, 2))
        np.testing.assert_allclose(
            loaded.density(X_test),
            density.density(X_test),
            rtol=1e-10
        )

    def test_network_roundtrip(self, tmp_path):
        """Save and load network should be equivalent."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(100, 2))
        Y = rng.uniform(0, 1, size=(100, 3))

        specs = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
            LayerSpec(input_dim=4, output_dim=3, basis_degree=2),
        ]
        net = HCRNetwork(specs, seed=42)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        filepath = tmp_path / "network.pkl"
        save_network(net, filepath)
        loaded = load_network(filepath)

        # Check structure
        assert loaded.num_layers == net.num_layers
        assert loaded._is_fitted == net._is_fitted

        # Check forward predictions
        X_test = rng.uniform(0, 1, size=(20, 2))
        np.testing.assert_allclose(
            loaded.forward(X_test),
            net.forward(X_test),
            rtol=1e-10
        )

    def test_unfitted_network_roundtrip(self, tmp_path):
        """Unfitted network should also round-trip correctly."""
        specs = [LayerSpec(input_dim=2, output_dim=2, basis_degree=2)]
        net = HCRNetwork(specs, seed=42)

        filepath = tmp_path / "unfitted.pkl"
        save_network(net, filepath)
        loaded = load_network(filepath)

        assert loaded.num_layers == net.num_layers
        assert loaded._is_fitted == net._is_fitted
