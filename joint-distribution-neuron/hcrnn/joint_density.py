"""
Joint density representation using polynomial basis (HCR Neuron).

This module implements the core JointDensity class that:
- Represents a probability density on [0,1]^d using polynomial basis expansion
- Learns coefficients from data via moment matching
- Evaluates density and log-density at arbitrary points
- Samples from the learned density via rejection sampling

This is the "neuron" in the HCR (Hierarchical Correlation Reconstruction)
framework, capable of bidirectional inference through conditioning.

Reference: Duda, J. (2024). "Joint distribution neuron" arXiv:2405.05097
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hcrnn.basis import TensorBasis


class JointDensity:
    """
    Joint probability density on [0,1]^d using polynomial basis expansion.

    The density is represented as:
        ρ(x) = Σ_j a_j f_j(x)

    where f_j are orthonormal polynomial basis functions and a_j are learned
    coefficients. For an orthonormal basis, coefficient estimation reduces
    to moment matching: a_j ≈ E[f_j(X)] estimated from data.

    Note: The polynomial representation can produce negative values in some
    regions. This is a known limitation of HCR-style density estimation.
    Methods clamp negative values when needed (e.g., for log_density, sampling).

    Attributes:
        dim: Number of dimensions
        basis: TensorBasis object for polynomial evaluation
        coeffs: Learned coefficients, shape (num_basis,)
        _fit_data: Stored training data for importance sampling (optional)

    Example:
        >>> from hcrnn.basis import build_tensor_basis
        >>> basis = build_tensor_basis(dim=2, degrees_per_dim=3)
        >>> density = JointDensity(basis)
        >>> X = np.random.rand(1000, 2)  # Training data in [0,1]^2
        >>> density.fit(X)
        >>> density.density(np.array([0.5, 0.5]))  # Evaluate at a point
    """

    def __init__(self, basis: TensorBasis):
        """
        Initialize JointDensity with a polynomial basis.

        Args:
            basis: TensorBasis object defining the polynomial basis
        """
        self.basis = basis
        self.dim = basis.dim
        self.coeffs: np.ndarray | None = None
        self._fit_data: np.ndarray | None = None
        self._rng = np.random.default_rng()

    def fit(
        self,
        X: np.ndarray,
        store_data: bool = True,
    ) -> JointDensity:
        """
        Fit density to data using moment matching.

        For orthonormal basis, the optimal coefficients under L2 loss are:
            a_j = E_data[f_j(x)] ≈ (1/N) Σ_i f_j(X_i)

        This is equivalent to projecting the empirical density onto the
        polynomial subspace.

        Args:
            X: Training data of shape (N, dim), with values in [0,1]^dim.
               Data should be pre-normalized to [0,1] range.
            store_data: If True, store data for use in importance sampling

        Returns:
            self, for method chaining

        Raises:
            ValueError: If X has wrong shape or values outside [0,1]
        """
        X = np.asarray(X)

        # Validate input
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"X must have shape (N, {self.dim}), got {X.shape}")

        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Data must be in [0,1]^d. Apply normalization first.")

        n_samples = X.shape[0]

        # Evaluate all basis functions at all data points
        # Shape: (N, num_basis)
        Phi = self.basis.evaluate(X)

        # Moment matching: a_j = mean(f_j(X))
        self.coeffs = np.mean(Phi, axis=0)

        # Store data for rejection sampling baseline
        if store_data:
            self._fit_data = X.copy()

        return self

    def density(self, x: np.ndarray, clamp: bool = False) -> np.ndarray | float:
        """
        Evaluate the density at given point(s).

        Computes ρ(x) = Σ_j a_j f_j(x)

        Args:
            x: Points in [0,1]^d. Shape (d,) for single point or (N, d) for batch.
            clamp: If True, clamp negative values to small positive epsilon

        Returns:
            Density value(s). Scalar for single point, array of shape (N,) for batch.

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before evaluating density")

        x = np.asarray(x)
        single_point = (x.ndim == 1)

        # Evaluate basis functions
        Phi = self.basis.evaluate(x)  # (num_basis,) or (N, num_basis)

        # Compute density as linear combination
        if single_point:
            rho = np.dot(Phi, self.coeffs)
        else:
            rho = Phi @ self.coeffs

        if clamp:
            eps = 1e-10
            rho = np.maximum(rho, eps)

        return float(rho) if single_point else rho

    def log_density(self, x: np.ndarray, eps: float = 1e-10) -> np.ndarray | float:
        """
        Evaluate log density at given point(s).

        Computes log(max(ρ(x), eps)) to handle potential negative regions.

        Args:
            x: Points in [0,1]^d. Shape (d,) for single point or (N, d) for batch.
            eps: Small value to avoid log(0) and handle negative density regions

        Returns:
            Log density value(s). Same shape rules as density().
        """
        rho = self.density(x, clamp=False)
        rho_clamped = np.maximum(rho, eps)
        return np.log(rho_clamped)

    def sample(
        self,
        n_samples: int,
        method: str = "rejection",
        max_iterations: int = 100,
    ) -> np.ndarray:
        """
        Sample from the learned density.

        Uses rejection sampling with uniform proposal on [0,1]^d.
        Falls back to perturbing training data if rejection sampling
        is too slow (density has regions much higher than 1).

        Args:
            n_samples: Number of samples to generate
            method: Sampling method, one of:
                - "rejection": Standard rejection sampling (default)
                - "data": Sample from training data with small noise
            max_iterations: Max iterations for rejection sampling before fallback

        Returns:
            Array of shape (n_samples, dim) with samples from density

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before sampling")

        if method == "data":
            return self._sample_from_data(n_samples)
        elif method == "rejection":
            return self._sample_rejection(n_samples, max_iterations)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_rejection(
        self,
        n_samples: int,
        max_iterations: int,
    ) -> np.ndarray:
        """
        Rejection sampling with uniform proposal on [0,1]^d.

        Finds approximate max of density via grid search, then uses
        rejection sampling. If acceptance rate is too low, falls back
        to data-based sampling.
        """
        # Estimate max density for rejection sampling bound
        # Use a coarse grid plus random samples
        n_grid = 1000
        X_grid = self._rng.uniform(0, 1, size=(n_grid, self.dim))
        rho_grid = self.density(X_grid, clamp=True)
        rho_max = np.max(rho_grid) * 1.2  # Safety margin

        # Rejection sampling
        samples = []
        total_proposed = 0

        for _ in range(max_iterations):
            # Propose batch of candidates
            batch_size = max(n_samples * 10, 1000)
            candidates = self._rng.uniform(0, 1, size=(batch_size, self.dim))
            rho_candidates = self.density(candidates, clamp=True)

            # Accept/reject
            u = self._rng.uniform(0, rho_max, size=batch_size)
            accepted = candidates[u < rho_candidates]
            samples.append(accepted)
            total_proposed += batch_size

            # Check if we have enough
            n_collected = sum(len(s) for s in samples)
            if n_collected >= n_samples:
                break

        all_samples = np.vstack(samples)

        if len(all_samples) >= n_samples:
            return all_samples[:n_samples]
        else:
            # Fallback to data-based sampling
            n_remaining = n_samples - len(all_samples)
            data_samples = self._sample_from_data(n_remaining)
            return np.vstack([all_samples, data_samples])

    def _sample_from_data(self, n_samples: int) -> np.ndarray:
        """
        Sample by bootstrapping from training data with small noise.

        Adds small Gaussian noise to training samples to avoid
        exact replication.
        """
        if self._fit_data is None:
            # If no stored data, sample uniformly
            return self._rng.uniform(0, 1, size=(n_samples, self.dim))

        # Bootstrap from training data
        indices = self._rng.integers(0, len(self._fit_data), size=n_samples)
        samples = self._fit_data[indices].copy()

        # Add small noise
        noise_scale = 0.01
        noise = self._rng.normal(0, noise_scale, size=samples.shape)
        samples = samples + noise

        # Clip to [0,1]
        samples = np.clip(samples, 0, 1)

        return samples

    def integrate(self, n_samples: int = 100_000) -> float:
        """
        Estimate integral of density over [0,1]^d via Monte Carlo.

        For a properly normalized density, this should return ≈1.

        Args:
            n_samples: Number of Monte Carlo samples

        Returns:
            Estimated integral value
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before integrating")

        # Monte Carlo integration with uniform sampling
        X = self._rng.uniform(0, 1, size=(n_samples, self.dim))
        rho = self.density(X, clamp=False)
        return float(np.mean(rho))

    def normalize(self, n_samples: int = 100_000) -> JointDensity:
        """
        Normalize coefficients so density integrates to 1.

        Estimates current integral and scales coefficients accordingly.

        Args:
            n_samples: Number of Monte Carlo samples for integral estimation

        Returns:
            self, for method chaining
        """
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before normalizing")

        current_integral = self.integrate(n_samples)
        if current_integral > 0:
            self.coeffs = self.coeffs / current_integral
        return self

    def get_coefficients(self) -> np.ndarray:
        """Return copy of learned coefficients."""
        if self.coeffs is None:
            raise RuntimeError("Must call fit() before getting coefficients")
        return self.coeffs.copy()

    def set_coefficients(self, coeffs: np.ndarray) -> JointDensity:
        """
        Set coefficients directly.

        Useful for loading saved models or experimenting with manual coefficients.

        Args:
            coeffs: Coefficient array of shape (num_basis,)

        Returns:
            self, for method chaining
        """
        coeffs = np.asarray(coeffs)
        if coeffs.shape != (self.basis.num_basis,):
            raise ValueError(
                f"coeffs must have shape ({self.basis.num_basis},), got {coeffs.shape}"
            )
        self.coeffs = coeffs.copy()
        return self

    def __repr__(self) -> str:
        fitted = "fitted" if self.coeffs is not None else "not fitted"
        return (
            f"JointDensity(dim={self.dim}, "
            f"num_basis={self.basis.num_basis}, "
            f"status={fitted})"
        )
