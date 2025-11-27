"""
Polynomial basis functions for joint density representation.

This module provides:
- Orthonormal 1D polynomial basis on [0,1] using shifted Legendre polynomials
- Tensor-product basis for multi-dimensional domains [0,1]^d
- Total-degree basis for N-dimensional domains (sum of degrees ≤ max_degree)

The orthonormal basis allows efficient least-squares fitting of densities
and enables the HCR neuron's bidirectional inference capability.

v0.3: Added N-dimensional support with total-degree constraint for more
efficient representation in higher dimensions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np
from numpy.polynomial import legendre as leg


def orthonormal_poly_1d(degree: int) -> list[Callable[[np.ndarray], np.ndarray]]:
    """
    Create orthonormal polynomial basis functions on [0,1] up to given degree.

    Uses shifted Legendre polynomials normalized so that:
        ∫_0^1 φ_n(x) φ_m(x) dx = δ_{nm}

    The standard Legendre polynomials P_n are orthogonal on [-1,1] with
    ∫_{-1}^{1} P_n(x) P_m(x) dx = 2/(2n+1) δ_{nm}

    Shifting to [0,1] via x' = 2x-1 and normalizing gives:
        φ_n(x) = √(2n+1) P_n(2x - 1)

    Args:
        degree: Maximum polynomial degree (inclusive). Creates degree+1 basis functions.

    Returns:
        List of callables, each taking array of shape (...,) and returning same shape.

    Example:
        >>> basis = orthonormal_poly_1d(2)
        >>> len(basis)
        3
        >>> x = np.array([0.0, 0.5, 1.0])
        >>> basis[0](x)  # Constant function φ_0 = 1
        array([1., 1., 1.])
    """
    basis_funcs = []

    for n in range(degree + 1):
        # Create coefficient array for P_n (all zeros except 1 at position n)
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        # Normalization factor: sqrt(2n+1) for orthonormality on [0,1]
        norm = np.sqrt(2 * n + 1)

        # Capture n and coeffs in closure
        def make_basis_func(c: np.ndarray, normalization: float) -> Callable:
            def phi(x: np.ndarray) -> np.ndarray:
                """Evaluate orthonormal basis function at points x in [0,1]."""
                x = np.asarray(x)
                # Shift [0,1] -> [-1,1] for Legendre evaluation
                x_shifted = 2.0 * x - 1.0
                return normalization * leg.legval(x_shifted, c)
            return phi

        basis_funcs.append(make_basis_func(coeffs.copy(), norm))

    return basis_funcs


@dataclass
class TensorBasis:
    """
    Tensor-product polynomial basis for [0,1]^d.

    For d-dimensional input, creates basis functions as products of 1D
    orthonormal polynomials:
        f_{(k1,...,kd)}(x) = φ_{k1}(x_1) × φ_{k2}(x_2) × ... × φ_{kd}(x_d)

    where each φ_k is an orthonormal polynomial on [0,1].

    Attributes:
        dim: Number of dimensions
        degrees: Maximum degree for each dimension
        multi_indices: List of tuples (k1, k2, ..., kd) for each basis function
        num_basis: Total number of basis functions
    """
    dim: int
    degrees: tuple[int, ...]
    multi_indices: list[tuple[int, ...]]
    _basis_1d: list[list[Callable[[np.ndarray], np.ndarray]]]

    @property
    def num_basis(self) -> int:
        """Total number of basis functions."""
        return len(self.multi_indices)

    def __len__(self) -> int:
        return self.num_basis

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at given points.

        Args:
            x: Points in [0,1]^d. Shape (N, d) for N points, or (d,) for single point.

        Returns:
            Array of shape (N, num_basis) with f_j(x_i) for each point and basis.
            If x has shape (d,), returns shape (num_basis,).
        """
        x = np.asarray(x)
        single_point = (x.ndim == 1)
        if single_point:
            x = x.reshape(1, -1)

        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {x.shape[1]}")

        n_points = x.shape[0]

        # Pre-compute all 1D basis evaluations for efficiency
        # shape: basis_1d_vals[dim_idx][degree] = array of shape (N,)
        basis_1d_vals: list[list[np.ndarray]] = []
        for d in range(self.dim):
            dim_vals = []
            for k in range(self.degrees[d] + 1):
                dim_vals.append(self._basis_1d[d][k](x[:, d]))
            basis_1d_vals.append(dim_vals)

        # Compute tensor products for each multi-index
        result = np.zeros((n_points, self.num_basis))
        for j, multi_idx in enumerate(self.multi_indices):
            # Product of 1D basis values across all dimensions
            product = np.ones(n_points)
            for d, k in enumerate(multi_idx):
                product *= basis_1d_vals[d][k]
            result[:, j] = product

        if single_point:
            return result[0]
        return result

    def evaluate_single(self, x: np.ndarray, basis_idx: int) -> float:
        """
        Evaluate a single basis function at a single point.

        Args:
            x: Point in [0,1]^d, shape (d,)
            basis_idx: Index of basis function to evaluate

        Returns:
            Scalar value of basis function at x
        """
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {x.shape}")

        multi_idx = self.multi_indices[basis_idx]
        result = 1.0
        for d, k in enumerate(multi_idx):
            result *= self._basis_1d[d][k](x[d])
        return float(result)

    def get_basis_funcs(self) -> list[Callable[[np.ndarray], float]]:
        """
        Return list of individual basis functions as callables.

        Each callable takes x of shape (d,) and returns a float.
        Useful for integration routines that expect simple function signatures.

        Returns:
            List of callables f_j(x) -> float
        """
        funcs = []
        for j in range(self.num_basis):
            def make_func(idx: int) -> Callable[[np.ndarray], float]:
                def f(x: np.ndarray) -> float:
                    return self.evaluate_single(x, idx)
                return f
            funcs.append(make_func(j))
        return funcs


def build_tensor_basis(
    dim: int,
    degrees_per_dim: int | Sequence[int],
) -> TensorBasis:
    """
    Build a tensor-product polynomial basis for [0,1]^d.

    Creates all combinations of 1D orthonormal polynomials up to the
    specified degree in each dimension.

    Args:
        dim: Number of dimensions
        degrees_per_dim: Either a single int (same degree for all dims)
                        or sequence of ints specifying max degree per dimension

    Returns:
        TensorBasis object for evaluating basis functions

    Example:
        >>> basis = build_tensor_basis(dim=2, degrees_per_dim=2)
        >>> basis.num_basis  # (2+1) * (2+1) = 9
        9
        >>> basis.multi_indices[0]
        (0, 0)
        >>> basis.multi_indices[-1]
        (2, 2)
    """
    # Normalize degrees to tuple
    if isinstance(degrees_per_dim, int):
        degrees = tuple([degrees_per_dim] * dim)
    else:
        degrees = tuple(degrees_per_dim)
        if len(degrees) != dim:
            raise ValueError(f"degrees_per_dim length {len(degrees)} != dim {dim}")

    # Build 1D basis for each dimension
    basis_1d = [orthonormal_poly_1d(d) for d in degrees]

    # Generate all multi-indices via Cartesian product
    ranges = [range(d + 1) for d in degrees]
    multi_indices = list(itertools.product(*ranges))

    return TensorBasis(
        dim=dim,
        degrees=degrees,
        multi_indices=multi_indices,
        _basis_1d=basis_1d,
    )


def verify_orthonormality(
    basis: TensorBasis,
    n_samples: int = 100_000,
    tol: float = 0.05,
) -> tuple[bool, np.ndarray]:
    """
    Verify orthonormality of basis via Monte Carlo integration.

    Computes the Gram matrix G_ij = ∫ f_i(x) f_j(x) dx ≈ (1/N) Σ f_i(x_k) f_j(x_k)
    and checks that it is close to the identity matrix.

    Args:
        basis: TensorBasis to verify
        n_samples: Number of Monte Carlo samples
        tol: Tolerance for max deviation from identity

    Returns:
        Tuple of (is_orthonormal, gram_matrix)

    Note:
        For uniform sampling on [0,1]^d, the Monte Carlo estimate directly
        gives the integral since the volume is 1.
    """
    # Sample uniformly on [0,1]^d
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(n_samples, basis.dim))

    # Evaluate all basis functions at all points
    Phi = basis.evaluate(X)  # shape (N, num_basis)

    # Gram matrix via Monte Carlo: G_ij = <f_i, f_j> ≈ mean(f_i * f_j)
    gram = (Phi.T @ Phi) / n_samples

    # Compare to identity
    identity = np.eye(basis.num_basis)
    max_error = np.max(np.abs(gram - identity))

    is_orthonormal = max_error < tol
    return is_orthonormal, gram


# =============================================================================
# N-dimensional Total-Degree Basis (v0.3)
# =============================================================================

def enumerate_multi_indices(dim: int, total_degree: int) -> list[tuple[int, ...]]:
    """
    Enumerate all multi-indices (k1, ..., kd) where sum(ki) ≤ total_degree.

    This gives fewer terms than full tensor product, scaling as
    O(total_degree^dim / dim!) instead of O(total_degree^dim).

    Args:
        dim: Number of dimensions
        total_degree: Maximum total degree (sum of individual degrees)

    Returns:
        List of multi-indices sorted by total degree then lexicographically.

    Example:
        >>> enumerate_multi_indices(2, 2)
        [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
        >>> len(enumerate_multi_indices(3, 3))  # C(3+3, 3) = 20
        20
    """
    indices = []

    def generate(current: list[int], remaining_degree: int, start_dim: int):
        if start_dim == dim:
            indices.append(tuple(current))
            return
        for k in range(remaining_degree + 1):
            current.append(k)
            generate(current, remaining_degree - k, start_dim + 1)
            current.pop()

    generate([], total_degree, 0)

    # Sort by total degree, then lexicographically
    indices.sort(key=lambda idx: (sum(idx), idx))
    return indices


def count_total_degree_terms(dim: int, total_degree: int) -> int:
    """
    Count the number of basis terms for given dimension and total degree.

    Formula: C(dim + total_degree, dim) = (dim + total_degree)! / (dim! * total_degree!)

    Args:
        dim: Number of dimensions
        total_degree: Maximum total degree

    Returns:
        Number of basis functions
    """
    # Use multiplicative formula to avoid overflow
    result = 1
    for i in range(1, dim + 1):
        result = result * (total_degree + i) // i
    return result


def multi_index_to_flat(
    multi_idx: tuple[int, ...],
    dim: int,
    total_degree: int,
) -> int:
    """
    Convert a multi-index to its flat index in the enumeration.

    Args:
        multi_idx: Tuple (k1, ..., kd)
        dim: Number of dimensions
        total_degree: Maximum total degree

    Returns:
        Integer index in the sorted list of multi-indices

    Raises:
        ValueError: If multi_idx is invalid
    """
    if len(multi_idx) != dim:
        raise ValueError(f"multi_idx must have {dim} elements")
    if sum(multi_idx) > total_degree:
        raise ValueError(f"sum({multi_idx}) = {sum(multi_idx)} > total_degree = {total_degree}")

    # Find index by enumeration (could be optimized with combinatorial formula)
    indices = enumerate_multi_indices(dim, total_degree)
    return indices.index(tuple(multi_idx))


def flat_to_multi_index(
    flat_idx: int,
    dim: int,
    total_degree: int,
) -> tuple[int, ...]:
    """
    Convert a flat index back to its multi-index.

    Args:
        flat_idx: Integer index
        dim: Number of dimensions
        total_degree: Maximum total degree

    Returns:
        Multi-index tuple (k1, ..., kd)

    Raises:
        IndexError: If flat_idx is out of range
    """
    indices = enumerate_multi_indices(dim, total_degree)
    if flat_idx < 0 or flat_idx >= len(indices):
        raise IndexError(f"flat_idx {flat_idx} out of range [0, {len(indices)})")
    return indices[flat_idx]


@dataclass
class TotalDegreeBasis:
    """
    Polynomial basis for [0,1]^d with total degree constraint.

    Unlike TensorBasis which uses per-dimension degree limits, this basis
    constrains sum(k1 + k2 + ... + kd) ≤ total_degree. This is more efficient
    for higher dimensions.

    For d-dimensional input, creates basis functions as products of 1D
    orthonormal polynomials:
        f_{(k1,...,kd)}(x) = φ_{k1}(x_1) × φ_{k2}(x_2) × ... × φ_{kd}(x_d)

    where sum(ki) ≤ total_degree.

    Attributes:
        dim: Number of dimensions
        total_degree: Maximum sum of individual degrees
        multi_indices: List of tuples (k1, k2, ..., kd) for each basis function
        num_basis: Total number of basis functions
    """
    dim: int
    total_degree: int
    multi_indices: list[tuple[int, ...]]
    _basis_1d: list[Callable[[np.ndarray], np.ndarray]]

    @property
    def num_basis(self) -> int:
        """Total number of basis functions."""
        return len(self.multi_indices)

    def __len__(self) -> int:
        return self.num_basis

    def total_degree_of(self, basis_idx: int) -> int:
        """Get the total degree (sum of indices) for a basis function."""
        return sum(self.multi_indices[basis_idx])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at given points.

        Args:
            x: Points in [0,1]^d. Shape (N, d) for N points, or (d,) for single point.

        Returns:
            Array of shape (N, num_basis) with f_j(x_i) for each point and basis.
            If x has shape (d,), returns shape (num_basis,).
        """
        x = np.asarray(x)
        single_point = (x.ndim == 1)
        if single_point:
            x = x.reshape(1, -1)

        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {x.shape[1]}")

        n_points = x.shape[0]

        # Pre-compute all needed 1D basis evaluations
        # We need degrees 0 to total_degree for each dimension
        basis_1d_vals: list[list[np.ndarray]] = []
        for d in range(self.dim):
            dim_vals = []
            for k in range(self.total_degree + 1):
                dim_vals.append(self._basis_1d[k](x[:, d]))
            basis_1d_vals.append(dim_vals)

        # Compute tensor products for each multi-index
        result = np.zeros((n_points, self.num_basis))
        for j, multi_idx in enumerate(self.multi_indices):
            product = np.ones(n_points)
            for d, k in enumerate(multi_idx):
                product *= basis_1d_vals[d][k]
            result[:, j] = product

        if single_point:
            return result[0]
        return result

    def evaluate_single(self, x: np.ndarray, basis_idx: int) -> float:
        """
        Evaluate a single basis function at a single point.

        Args:
            x: Point in [0,1]^d, shape (d,)
            basis_idx: Index of basis function to evaluate

        Returns:
            Scalar value of basis function at x
        """
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {x.shape}")

        multi_idx = self.multi_indices[basis_idx]
        result = 1.0
        for d, k in enumerate(multi_idx):
            result *= self._basis_1d[k](x[d])
        return float(result)

    def get_basis_funcs(self) -> list[Callable[[np.ndarray], float]]:
        """Return list of individual basis functions as callables."""
        funcs = []
        for j in range(self.num_basis):
            def make_func(idx: int) -> Callable[[np.ndarray], float]:
                def f(x: np.ndarray) -> float:
                    return self.evaluate_single(x, idx)
                return f
            funcs.append(make_func(j))
        return funcs

    # Compatibility property to match TensorBasis interface
    @property
    def degrees(self) -> tuple[int, ...]:
        """Per-dimension max degree (for compatibility with TensorBasis)."""
        return tuple([self.total_degree] * self.dim)


def build_total_degree_basis(dim: int, total_degree: int) -> TotalDegreeBasis:
    """
    Build a total-degree polynomial basis for [0,1]^d.

    Creates basis functions where the sum of individual polynomial degrees
    is at most total_degree. This is more efficient than full tensor product
    for higher dimensions.

    Args:
        dim: Number of dimensions
        total_degree: Maximum sum of polynomial degrees

    Returns:
        TotalDegreeBasis object for evaluating basis functions

    Example:
        >>> basis = build_total_degree_basis(dim=3, total_degree=2)
        >>> basis.num_basis  # C(3+2, 3) = 10
        10
        >>> basis.multi_indices[0]
        (0, 0, 0)
    """
    # Get all multi-indices
    multi_indices = enumerate_multi_indices(dim, total_degree)

    # Build 1D basis up to total_degree
    basis_1d = orthonormal_poly_1d(total_degree)

    return TotalDegreeBasis(
        dim=dim,
        total_degree=total_degree,
        multi_indices=multi_indices,
        _basis_1d=basis_1d,
    )


# Type alias for any basis type
BasisType = Union[TensorBasis, TotalDegreeBasis]
