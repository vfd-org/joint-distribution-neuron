"""Tests for conditional inference utilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hcrnn.basis import build_tensor_basis
from hcrnn.joint_density import JointDensity
from hcrnn.conditionals import (
    conditional_density,
    conditional_expectation,
    conditional_variance,
    conditional_mode,
    marginal_density,
    sample_conditional,
)


def generate_correlated_gaussian_data(
    n_samples: int,
    rho: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D correlated Gaussian data mapped to [0,1]^2.

    Returns:
        Tuple of (X_uniform, mean, cov) where X_uniform is in [0,1]^2
    """
    rng = np.random.default_rng(seed)

    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    X_gaussian = rng.multivariate_normal(mean, cov, size=n_samples)

    X_uniform = np.zeros_like(X_gaussian)
    X_uniform[:, 0] = stats.norm.cdf(X_gaussian[:, 0])
    X_uniform[:, 1] = stats.norm.cdf(X_gaussian[:, 1])

    return X_uniform, mean, cov


def true_conditional_expectation_gaussian(
    x_given: float,
    rho: float,
    given_idx: int = 0,
) -> float:
    """
    True conditional expectation for standard bivariate Gaussian.

    For (X, Y) ~ N(0, [[1, rho], [rho, 1]]):
    E[Y | X = x] = rho * x

    After mapping through CDF:
    E[CDF(Y) | CDF(X) = u] is more complex, but for moderate correlation
    it should show positive relationship when rho > 0.
    """
    # This is the true conditional expectation in the original Gaussian space
    # E[Y | X = x] = rho * x for standard bivariate normal
    x_original = stats.norm.ppf(x_given)
    y_expected_original = rho * x_original
    return float(stats.norm.cdf(y_expected_original))


class TestConditionalDensity:
    """Tests for conditional_density function."""

    def test_returns_grid_and_density(self):
        """Should return grid points and density values."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        grid, density = conditional_density(
            joint,
            given_indices=[0],
            given_values=[0.5],
            target_indices=[1],
            grid_size=50,
        )

        assert grid.shape == (50,)
        assert density.shape == (50,)

    def test_density_integrates_to_one(self):
        """Conditional density should integrate to approximately 1."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(1000, 2)
        joint.fit(X)

        grid, density = conditional_density(
            joint,
            given_indices=[0],
            given_values=[0.5],
            target_indices=[1],
            grid_size=100,
        )

        # Numerical integration
        integral = np.trapz(density, grid)
        assert abs(integral - 1.0) < 0.2

    def test_density_nonnegative(self):
        """Conditional density should be non-negative (clamped)."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        grid, density = conditional_density(
            joint,
            given_indices=[0],
            given_values=[0.5],
            target_indices=[1],
        )

        assert np.all(density >= 0)

    def test_validates_indices(self):
        """Should raise on invalid indices."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        # Overlapping indices
        with pytest.raises(ValueError):
            conditional_density(
                joint,
                given_indices=[0],
                given_values=[0.5],
                target_indices=[0],  # Same as given
            )

        # Out of range index
        with pytest.raises(ValueError):
            conditional_density(
                joint,
                given_indices=[5],  # Invalid for 2D
                given_values=[0.5],
            )

    def test_bidirectional_inference(self):
        """Can condition on either variable (x|y or y|x)."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X, _, _ = generate_correlated_gaussian_data(2000, rho=0.7)
        joint.fit(X)

        # p(y | x=0.5)
        grid_y, density_y = conditional_density(
            joint,
            given_indices=[0],
            given_values=[0.5],
            target_indices=[1],
        )

        # p(x | y=0.5)
        grid_x, density_x = conditional_density(
            joint,
            given_indices=[1],
            given_values=[0.5],
            target_indices=[0],
        )

        # Both should be valid densities
        assert np.all(density_y >= 0)
        assert np.all(density_x >= 0)
        assert abs(np.trapz(density_y, grid_y) - 1.0) < 0.2
        assert abs(np.trapz(density_x, grid_x) - 1.0) < 0.2


class TestConditionalExpectation:
    """Tests for conditional_expectation function."""

    def test_returns_scalar(self):
        """Should return a scalar value."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        exp = conditional_expectation(
            joint,
            target_index=1,
            given_indices=[0],
            given_values=[0.5],
        )

        assert isinstance(exp, float)
        assert 0 <= exp <= 1

    def test_expectation_in_range(self):
        """Expectation should be in [0, 1]."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(1000, 2)
        joint.fit(X)

        for x_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            exp = conditional_expectation(
                joint,
                target_index=1,
                given_indices=[0],
                given_values=[x_val],
            )
            assert 0 <= exp <= 1

    def test_positive_correlation_effect(self):
        """With positive correlation, E[y|x] should increase with x."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=4)
        joint = JointDensity(basis)
        X, _, _ = generate_correlated_gaussian_data(3000, rho=0.8)
        joint.fit(X)

        # E[y | x=0.2] should be less than E[y | x=0.8]
        exp_low = conditional_expectation(
            joint, target_index=1, given_indices=[0], given_values=[0.2]
        )
        exp_high = conditional_expectation(
            joint, target_index=1, given_indices=[0], given_values=[0.8]
        )

        assert exp_high > exp_low

    def test_bidirectional_expectation(self):
        """Can compute E[x|y] and E[y|x]."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=4)
        joint = JointDensity(basis)
        X, _, _ = generate_correlated_gaussian_data(2000, rho=0.7)
        joint.fit(X)

        # E[y | x=0.5]
        exp_y_given_x = conditional_expectation(
            joint, target_index=1, given_indices=[0], given_values=[0.5]
        )

        # E[x | y=0.5]
        exp_x_given_y = conditional_expectation(
            joint, target_index=0, given_indices=[1], given_values=[0.5]
        )

        # Both should be valid and close to 0.5 for symmetric data
        assert 0.3 <= exp_y_given_x <= 0.7
        assert 0.3 <= exp_x_given_y <= 0.7


class TestConditionalVariance:
    """Tests for conditional_variance function."""

    def test_returns_nonnegative(self):
        """Variance should be non-negative."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        var = conditional_variance(
            joint,
            target_index=1,
            given_indices=[0],
            given_values=[0.5],
        )

        assert var >= 0


class TestConditionalMode:
    """Tests for conditional_mode function."""

    def test_mode_in_range(self):
        """Mode should be in [0, 1]."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        mode = conditional_mode(
            joint,
            target_index=1,
            given_indices=[0],
            given_values=[0.5],
        )

        assert 0 <= mode <= 1


class TestMarginalDensity:
    """Tests for marginal_density function."""

    def test_marginal_1d(self):
        """Compute 1D marginal from 2D joint."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(1000, 2)
        joint.fit(X)

        grid, density = marginal_density(joint, target_indices=[0])

        assert grid.shape[0] == density.shape[0]
        assert np.all(density >= 0)

    def test_marginal_uniform_data(self):
        """Marginal of uniform data should be roughly uniform."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(2000, 2))
        joint.fit(X)

        grid, density = marginal_density(joint, target_indices=[0], grid_size=50)

        # For uniform, density should be relatively flat
        # Check coefficient of variation is small
        cv = np.std(density) / np.mean(density)
        assert cv < 0.5  # Not too variable


class TestSampleConditional:
    """Tests for sample_conditional function."""

    def test_sample_shape(self):
        """Should return correct shape."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        samples = sample_conditional(
            joint,
            given_indices=[0],
            given_values=[0.5],
            target_indices=[1],
            n_samples=50,
        )

        assert samples.shape == (50, 1)

    def test_samples_in_range(self):
        """Samples should be in [0, 1]."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        joint = JointDensity(basis)
        X = np.random.rand(500, 2)
        joint.fit(X)

        samples = sample_conditional(
            joint,
            given_indices=[0],
            given_values=[0.5],
            n_samples=100,
        )

        assert np.all(samples >= 0)
        assert np.all(samples <= 1)


class TestThreeDimensional:
    """Tests with 3D data."""

    def test_3d_conditional(self):
        """Conditional density works with 3D joint."""
        basis = build_tensor_basis(dim=3, degrees_per_dim=2)
        joint = JointDensity(basis)
        X = np.random.rand(1000, 3)
        joint.fit(X)

        # p(x3 | x1=0.5, x2=0.5)
        grid, density = conditional_density(
            joint,
            given_indices=[0, 1],
            given_values=[0.5, 0.5],
            target_indices=[2],
        )

        assert grid.shape[0] == density.shape[0]
        assert np.all(density >= 0)

    def test_3d_expectation(self):
        """Conditional expectation works with 3D joint."""
        basis = build_tensor_basis(dim=3, degrees_per_dim=2)
        joint = JointDensity(basis)
        X = np.random.rand(1000, 3)
        joint.fit(X)

        exp = conditional_expectation(
            joint,
            target_index=2,
            given_indices=[0, 1],
            given_values=[0.5, 0.5],
        )

        assert 0 <= exp <= 1
