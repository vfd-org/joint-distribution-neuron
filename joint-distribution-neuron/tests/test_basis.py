"""Tests for polynomial basis module."""

import numpy as np
import pytest

from hcrnn.basis import (
    orthonormal_poly_1d,
    build_tensor_basis,
    verify_orthonormality,
    TensorBasis,
)


class TestOrthonormalPoly1D:
    """Tests for 1D orthonormal polynomial basis."""

    def test_correct_number_of_basis_functions(self):
        """Basis up to degree n should have n+1 functions."""
        for degree in [0, 1, 2, 5, 10]:
            basis = orthonormal_poly_1d(degree)
            assert len(basis) == degree + 1

    def test_constant_basis_is_one(self):
        """The zeroth basis function (degree 0) should be constant 1."""
        basis = orthonormal_poly_1d(0)
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(basis[0](x), np.ones(5), rtol=1e-10)

    def test_orthonormality_1d(self):
        """1D basis should be orthonormal on [0,1] via Monte Carlo."""
        degree = 5
        basis = orthonormal_poly_1d(degree)
        n_samples = 100_000
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, n_samples)

        # Compute Gram matrix
        n_basis = len(basis)
        gram = np.zeros((n_basis, n_basis))
        vals = np.zeros((n_samples, n_basis))
        for i, phi in enumerate(basis):
            vals[:, i] = phi(x)

        gram = vals.T @ vals / n_samples

        # Should be close to identity
        expected = np.eye(n_basis)
        np.testing.assert_allclose(gram, expected, atol=0.02)

    def test_basis_values_at_endpoints(self):
        """Test that basis functions are well-defined at [0,1] boundaries."""
        basis = orthonormal_poly_1d(3)
        x = np.array([0.0, 1.0])
        for phi in basis:
            vals = phi(x)
            assert np.all(np.isfinite(vals))

    def test_vectorized_evaluation(self):
        """Basis functions should handle array inputs."""
        basis = orthonormal_poly_1d(2)
        x = np.linspace(0, 1, 100)
        for phi in basis:
            vals = phi(x)
            assert vals.shape == x.shape


class TestTensorBasis:
    """Tests for tensor-product basis."""

    def test_correct_number_of_basis_functions_uniform_degree(self):
        """Tensor basis size should be product of (degree+1) per dimension."""
        # 2D, degree 2 in each: (2+1)*(2+1) = 9
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        assert basis.num_basis == 9
        assert len(basis) == 9

        # 3D, degree 1 in each: 2*2*2 = 8
        basis = build_tensor_basis(dim=3, degrees_per_dim=1)
        assert basis.num_basis == 8

    def test_correct_number_of_basis_functions_mixed_degree(self):
        """Test with different degrees per dimension."""
        # degrees [2, 3]: (2+1)*(3+1) = 12
        basis = build_tensor_basis(dim=2, degrees_per_dim=[2, 3])
        assert basis.num_basis == 12
        assert basis.degrees == (2, 3)

    def test_multi_indices_correct(self):
        """Multi-indices should cover all combinations."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=1)
        expected_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert sorted(basis.multi_indices) == sorted(expected_indices)

    def test_evaluate_single_point(self):
        """Evaluate basis at a single point."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        x = np.array([0.5, 0.5])
        vals = basis.evaluate(x)
        assert vals.shape == (basis.num_basis,)
        assert np.all(np.isfinite(vals))

    def test_evaluate_multiple_points(self):
        """Evaluate basis at multiple points."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        n_points = 50
        X = np.random.rand(n_points, 2)
        vals = basis.evaluate(X)
        assert vals.shape == (n_points, basis.num_basis)
        assert np.all(np.isfinite(vals))

    def test_evaluate_single_matches_batch(self):
        """Single-point evaluation should match batch evaluation."""
        basis = build_tensor_basis(dim=3, degrees_per_dim=2)
        x = np.array([0.3, 0.5, 0.7])

        # Single point via evaluate
        vals_batch = basis.evaluate(x)

        # Single points via evaluate_single
        vals_single = np.array([basis.evaluate_single(x, j) for j in range(basis.num_basis)])

        np.testing.assert_allclose(vals_batch, vals_single, rtol=1e-10)

    def test_orthonormality_2d(self):
        """2D tensor basis should be orthonormal."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        is_ortho, gram = verify_orthonormality(basis, n_samples=100_000, tol=0.05)
        assert is_ortho, f"Max error from identity: {np.max(np.abs(gram - np.eye(basis.num_basis)))}"

    def test_orthonormality_3d(self):
        """3D tensor basis should be orthonormal."""
        basis = build_tensor_basis(dim=3, degrees_per_dim=2)
        is_ortho, gram = verify_orthonormality(basis, n_samples=100_000, tol=0.05)
        assert is_ortho, f"Max error from identity: {np.max(np.abs(gram - np.eye(basis.num_basis)))}"

    def test_get_basis_funcs(self):
        """get_basis_funcs should return callable functions."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        funcs = basis.get_basis_funcs()
        assert len(funcs) == basis.num_basis

        x = np.array([0.4, 0.6])
        for j, f in enumerate(funcs):
            val = f(x)
            expected = basis.evaluate_single(x, j)
            np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_dimension_mismatch_raises(self):
        """Wrong dimension input should raise error."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0.5, 0.5, 0.5]))  # 3D point for 2D basis

    def test_degrees_per_dim_length_mismatch_raises(self):
        """Mismatched degrees_per_dim length should raise error."""
        with pytest.raises(ValueError):
            build_tensor_basis(dim=3, degrees_per_dim=[2, 2])  # Only 2 degrees for 3D


class TestVerifyOrthonormality:
    """Tests for orthonormality verification utility."""

    def test_passes_for_orthonormal_basis(self):
        """Should return True for properly constructed basis."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        is_ortho, _ = verify_orthonormality(basis, n_samples=50_000, tol=0.1)
        assert is_ortho

    def test_returns_gram_matrix(self):
        """Should return the computed Gram matrix."""
        basis = build_tensor_basis(dim=2, degrees_per_dim=1)
        _, gram = verify_orthonormality(basis, n_samples=50_000)
        assert gram.shape == (basis.num_basis, basis.num_basis)
        # Diagonal should be close to 1
        np.testing.assert_allclose(np.diag(gram), np.ones(basis.num_basis), atol=0.1)
