"""Tests for JointDensity class."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hcrnn.basis import build_tensor_basis, build_total_degree_basis
from hcrnn.joint_density import JointDensity, ConditionalDensity


def generate_correlated_gaussian_data(
    n_samples: int,
    rho: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D correlated Gaussian data mapped to [0,1]^2.

    Args:
        n_samples: Number of samples
        rho: Correlation coefficient
        seed: Random seed

    Returns:
        Tuple of (X_uniform, X_original) where X_uniform is in [0,1]^2
        and X_original is the raw Gaussian samples.
    """
    rng = np.random.default_rng(seed)

    # Generate correlated Gaussian
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    X_gaussian = rng.multivariate_normal(mean, cov, size=n_samples)

    # Map to [0,1] via CDF
    X_uniform = np.zeros_like(X_gaussian)
    X_uniform[:, 0] = stats.norm.cdf(X_gaussian[:, 0])
    X_uniform[:, 1] = stats.norm.cdf(X_gaussian[:, 1])

    return X_uniform, X_gaussian


class TestJointDensityBasic:
    """Basic functionality tests."""

    def test_init(self):
        """Test initialization."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        assert density.dim == 2
        assert density.coeffs is None
        assert density.basis is basis

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        X = np.random.rand(100, 2)
        result = density.fit(X)
        assert result is density

    def test_fit_sets_coefficients(self):
        """fit() should set coefficients array."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        X = np.random.rand(100, 2)
        density.fit(X)
        assert density.coeffs is not None
        assert density.coeffs.shape == (basis.num_basis,)

    def test_density_requires_fit(self):
        """density() should raise if not fitted."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        with pytest.raises(RuntimeError):
            density.density(np.array([0.5, 0.5]))

    def test_fit_validates_shape(self):
        """fit() should reject wrong-shaped data."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        with pytest.raises(ValueError):
            density.fit(np.random.rand(100, 3))  # Wrong dim

    def test_fit_validates_range(self):
        """fit() should reject data outside [0,1]."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        X = np.random.rand(100, 2) * 2  # Values up to 2
        with pytest.raises(ValueError):
            density.fit(X)


class TestJointDensityEvaluation:
    """Tests for density evaluation."""

    def test_density_single_point(self):
        """Evaluate density at a single point."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        x = np.array([0.5, 0.5])
        rho = density.density(x)
        assert isinstance(rho, float)
        assert np.isfinite(rho)

    def test_density_batch(self):
        """Evaluate density at multiple points."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        X_test = np.random.rand(100, 2)
        rho = density.density(X_test)
        assert rho.shape == (100,)
        assert np.all(np.isfinite(rho))

    def test_density_clamp(self):
        """Clamped density should be positive."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        X_test = np.random.rand(100, 2)
        rho = density.density(X_test, clamp=True)
        assert np.all(rho > 0)

    def test_log_density(self):
        """Log density should be finite."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        x = np.array([0.5, 0.5])
        log_rho = density.log_density(x)
        assert isinstance(log_rho, (float, np.floating))
        assert np.isfinite(log_rho)


class TestJointDensityIntegration:
    """Tests for density integration and normalization."""

    def test_integrate_uniform_data(self):
        """Density fitted on uniform data should integrate close to 1."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)

        # Uniform data in [0,1]^2
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(5000, 2))
        density.fit(X)

        integral = density.integrate(n_samples=50_000)
        # For uniform data, integral should be close to 1
        assert abs(integral - 1.0) < 0.1

    def test_normalize(self):
        """normalize() should make integral closer to 1."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(2000, 2))
        density.fit(X)
        density.normalize(n_samples=50_000)

        integral = density.integrate(n_samples=50_000)
        assert abs(integral - 1.0) < 0.1


class TestJointDensitySampling:
    """Tests for sampling from density."""

    def test_sample_shape(self):
        """sample() should return correct shape."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        samples = density.sample(100, method="rejection")
        assert samples.shape == (100, 2)

    def test_sample_in_range(self):
        """Samples should be in [0,1]^d."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X)

        samples = density.sample(100)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_sample_data_method(self):
        """Data-based sampling should work."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        density = JointDensity(basis)
        X = np.random.rand(500, 2)
        density.fit(X, store_data=True)

        samples = density.sample(100, method="data")
        assert samples.shape == (100, 2)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)


class TestJointDensityCorrelatedData:
    """Tests with correlated Gaussian data to verify learning."""

    def test_density_higher_at_data_regions(self):
        """Density should be higher where data is concentrated."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=4)
        density = JointDensity(basis)

        # Generate correlated data (positive correlation)
        X, _ = generate_correlated_gaussian_data(2000, rho=0.8)
        density.fit(X)

        # Points along diagonal (high data density for positive correlation)
        diagonal_points = np.array([
            [0.3, 0.3],
            [0.5, 0.5],
            [0.7, 0.7],
        ])

        # Points off diagonal (low data density)
        off_diagonal_points = np.array([
            [0.2, 0.8],
            [0.8, 0.2],
        ])

        rho_diagonal = density.density(diagonal_points, clamp=True)
        rho_off_diagonal = density.density(off_diagonal_points, clamp=True)

        # Diagonal should have higher average density
        assert np.mean(rho_diagonal) > np.mean(rho_off_diagonal)

    def test_3d_density_fitting(self):
        """Test fitting on 3D data."""
        basis = build_tensor_basis(dim=3, degrees_per_dim=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        x = np.array([0.5, 0.5, 0.5])
        rho = density.density(x)
        assert np.isfinite(rho)


class TestJointDensityCoefficients:
    """Tests for coefficient manipulation."""

    def test_get_coefficients(self):
        """get_coefficients() should return copy."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        X = np.random.rand(100, 2)
        density.fit(X)

        coeffs = density.get_coefficients()
        assert coeffs.shape == (basis.num_basis,)

        # Modifying returned array shouldn't affect internal state
        coeffs[0] = 999
        assert density.coeffs[0] != 999

    def test_set_coefficients(self):
        """set_coefficients() should work."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)

        coeffs = np.ones(basis.num_basis)
        density.set_coefficients(coeffs)

        assert density.coeffs is not None
        np.testing.assert_array_equal(density.coeffs, coeffs)

    def test_set_coefficients_validates_shape(self):
        """set_coefficients() should reject wrong shape."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)

        with pytest.raises(ValueError):
            density.set_coefficients(np.ones(5))  # Wrong size


class TestJointDensityRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Repr should indicate unfitted status."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        repr_str = repr(density)
        assert "not fitted" in repr_str
        assert "dim=2" in repr_str

    def test_repr_fitted(self):
        """Repr should indicate fitted status."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        density = JointDensity(basis)
        X = np.random.rand(100, 2)
        density.fit(X)
        repr_str = repr(density)
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


# =============================================================================
# v0.3 Tests: Total Degree Basis
# =============================================================================

class TestTotalDegreeBasis:
    """Tests for total-degree basis support."""

    def test_init_total_degree(self):
        """Test initialization with total-degree basis."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)
        assert density.dim == 3
        # C(3+3, 3) = 20 terms
        assert basis.num_basis == 20

    def test_fit_3d_total_degree(self):
        """Fit on 3D data with total-degree basis."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 3))
        density.fit(X)

        assert density.coeffs is not None
        assert density.coeffs.shape == (20,)

    def test_evaluate_3d_total_degree(self):
        """Evaluate 3D density with total-degree basis."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 3))
        density.fit(X)

        # Evaluate at a point
        x = np.array([0.5, 0.5, 0.5])
        rho = density.density(x)
        assert np.isfinite(rho)

        # Batch evaluation
        X_test = rng.uniform(0, 1, size=(50, 3))
        rho_batch = density.density(X_test)
        assert rho_batch.shape == (50,)
        assert np.all(np.isfinite(rho_batch))

    def test_total_degree_fewer_terms_than_tensor(self):
        """Total-degree basis should have fewer terms than tensor product."""
        from hcrnn.basis import count_total_degree_terms

        for dim in [3, 4, 5]:
            for degree in [2, 3]:
                n_total = count_total_degree_terms(dim, degree)
                n_tensor = (degree + 1) ** dim
                assert n_total < n_tensor


# =============================================================================
# v0.3 Tests: Marginalization
# =============================================================================

class TestMarginalization:
    """Tests for marginalize() method."""

    def test_marginalize_3d_to_2d(self):
        """Marginalize 3D density to 2D."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        # Marginalize to keep dims 0 and 1
        marginal = density.marginalize([0, 1])

        assert marginal.dim == 2
        assert marginal.coeffs is not None

    def test_marginalize_3d_to_1d(self):
        """Marginalize 3D density to 1D."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        # Marginalize to keep only dim 0
        marginal = density.marginalize([0])

        assert marginal.dim == 1
        assert marginal.coeffs is not None

        # Should integrate to approximately 1
        integral = marginal.integrate(n_samples=10000)
        assert abs(integral - 1.0) < 0.2

    def test_marginalize_preserves_integration(self):
        """Marginal should integrate to approximately 1."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(2000, 3))
        density.fit(X)

        marginal_01 = density.marginalize([0, 1])
        marginal_2 = density.marginalize([2])

        # Both should integrate to ~1
        int_01 = marginal_01.integrate(n_samples=10000)
        int_2 = marginal_2.integrate(n_samples=10000)

        assert abs(int_01 - 1.0) < 0.2
        assert abs(int_2 - 1.0) < 0.2

    def test_marginalize_requires_fit(self):
        """marginalize() should raise if not fitted."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        with pytest.raises(RuntimeError):
            density.marginalize([0, 1])

    def test_marginalize_validates_dims(self):
        """marginalize() should validate dimension indices."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 3))
        density.fit(X)

        with pytest.raises(ValueError):
            density.marginalize([])  # Empty

        with pytest.raises(ValueError):
            density.marginalize([0, 5])  # Invalid dim


# =============================================================================
# v0.3 Tests: Conditioning
# =============================================================================

class TestConditioning:
    """Tests for condition_on() method."""

    def test_condition_on_single_dim(self):
        """Condition on a single dimension."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        # Condition on x2 = 0.5
        cond = density.condition_on({2: 0.5})

        assert isinstance(cond, ConditionalDensity)
        assert cond.free_dim == 2
        assert cond.free_dims == [0, 1]

    def test_condition_on_multiple_dims(self):
        """Condition on multiple dimensions."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        # Condition on x0 = 0.3, x1 = 0.7
        cond = density.condition_on({0: 0.3, 1: 0.7})

        assert cond.free_dim == 1
        assert cond.free_dims == [2]

    def test_conditional_sample(self):
        """Sample from conditional distribution."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        cond = density.condition_on({2: 0.5})
        samples = cond.sample(100)

        assert samples.shape == (100, 2)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_conditional_expected_value(self):
        """Compute expected value from conditional."""
        basis = build_total_degree_basis(dim=3, total_degree=3)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(1000, 3))
        density.fit(X)

        cond = density.condition_on({2: 0.5})
        expected = cond.expected_value(n_samples=2000)

        assert expected.shape == (2,)
        # For uniform data, expected values should be near 0.5
        assert np.all(np.abs(expected - 0.5) < 0.3)

    def test_condition_on_requires_fit(self):
        """condition_on() should raise if not fitted."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        with pytest.raises(RuntimeError):
            density.condition_on({0: 0.5})

    def test_condition_on_validates_dims(self):
        """condition_on() should validate fixed dimensions."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 3))
        density.fit(X)

        with pytest.raises(ValueError):
            density.condition_on({5: 0.5})  # Invalid dim

        with pytest.raises(ValueError):
            density.condition_on({0: -0.1})  # Value out of range

    def test_condition_on_all_dims_raises(self):
        """Conditioning on all dims should raise."""
        basis = build_total_degree_basis(dim=2, total_degree=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 2))
        density.fit(X)

        with pytest.raises(ValueError):
            density.condition_on({0: 0.5, 1: 0.5})  # No free dims

    def test_conditional_repr(self):
        """Test ConditionalDensity string representation."""
        basis = build_total_degree_basis(dim=3, total_degree=2)
        density = JointDensity(basis)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(500, 3))
        density.fit(X)

        cond = density.condition_on({1: 0.5})
        repr_str = repr(cond)

        assert "free_dims" in repr_str
        assert "x1=0.500" in repr_str
