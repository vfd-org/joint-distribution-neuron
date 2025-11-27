"""
Conditional density and expectation utilities for bidirectional inference.

This module enables the HCR neuron's key capability: computing conditionals
in any direction. Given a joint density p(x1, x2, ..., xd), we can compute:
- p(xi | xj, xk, ...) for any subset of conditioning variables
- E[xi | xj, xk, ...] conditional expectations

The approach uses numerical integration over the target variables while
fixing the conditioning variables.

This is what makes the HCR neuron "bidirectional" - unlike traditional
neural networks that only compute p(y|x), we can equally well compute
p(x|y), p(x1|x2,x3), etc.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import integrate

from hcrnn.joint_density import JointDensity


def conditional_density(
    joint: JointDensity,
    given_indices: Sequence[int],
    given_values: Sequence[float],
    target_indices: Sequence[int] | None = None,
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional density over target variables given fixed values.

    For joint density p(x), computes p(x_target | x_given = given_values)
    by evaluating the joint on a grid and normalizing.

    Args:
        joint: Fitted JointDensity object
        given_indices: Indices of variables to condition on (fix)
        given_values: Values for the conditioning variables
        target_indices: Indices of variables to compute density over.
                       If None, uses all non-given indices.
        grid_size: Number of grid points per target dimension

    Returns:
        Tuple of (grid_points, density_values) where:
        - grid_points: Array of shape (grid_size^n_target, n_target) or (grid_size,) for 1D
        - density_values: Normalized conditional density at each grid point

    Example:
        >>> # For 2D joint p(x, y), compute p(y | x=0.3)
        >>> grid, density = conditional_density(
        ...     joint, given_indices=[0], given_values=[0.3], target_indices=[1]
        ... )
    """
    given_indices = list(given_indices)
    given_values = list(given_values)

    if len(given_indices) != len(given_values):
        raise ValueError("given_indices and given_values must have same length")

    # Determine target indices
    if target_indices is None:
        target_indices = [i for i in range(joint.dim) if i not in given_indices]
    else:
        target_indices = list(target_indices)

    # Validate indices
    all_indices = set(given_indices) | set(target_indices)
    if len(all_indices) != len(given_indices) + len(target_indices):
        raise ValueError("given_indices and target_indices must not overlap")
    if max(all_indices) >= joint.dim or min(all_indices) < 0:
        raise ValueError(f"Indices must be in [0, {joint.dim})")

    n_target = len(target_indices)

    if n_target == 0:
        raise ValueError("Must have at least one target variable")

    # Create grid over target variables
    grids_1d = [np.linspace(0.001, 0.999, grid_size) for _ in range(n_target)]

    if n_target == 1:
        # 1D case: simple grid
        target_grid = grids_1d[0]
        n_points = grid_size
    else:
        # Multi-D case: meshgrid
        mesh = np.meshgrid(*grids_1d, indexing='ij')
        target_grid = np.stack([m.ravel() for m in mesh], axis=1)
        n_points = target_grid.shape[0]

    # Build full points array by inserting given values
    full_points = np.zeros((n_points, joint.dim))

    # Fill in given values
    for idx, val in zip(given_indices, given_values):
        full_points[:, idx] = val

    # Fill in target grid values
    if n_target == 1:
        full_points[:, target_indices[0]] = target_grid
    else:
        for i, idx in enumerate(target_indices):
            full_points[:, idx] = target_grid[:, i]

    # Evaluate joint density at all points
    density_values = joint.density(full_points, clamp=True)

    # Normalize to get conditional density
    # Use trapezoidal approximation for normalization
    if n_target == 1:
        dx = target_grid[1] - target_grid[0]
        normalizer = np.trapz(density_values, dx=dx)
    else:
        # Multi-D: approximate integral via mean * volume
        volume = 1.0  # (0.999 - 0.001)^n_target ≈ 1
        normalizer = np.mean(density_values) * volume

    if normalizer > 0:
        density_values = density_values / normalizer

    return target_grid, density_values


def conditional_expectation(
    joint: JointDensity,
    target_index: int,
    given_indices: Sequence[int],
    given_values: Sequence[float],
    grid_size: int = 100,
) -> float:
    """
    Compute conditional expectation E[x_target | x_given = given_values].

    Uses numerical integration: E[x_t | x_g] = ∫ x_t p(x_t | x_g) dx_t

    Args:
        joint: Fitted JointDensity object
        target_index: Index of variable to compute expectation of
        given_indices: Indices of conditioning variables
        given_values: Values for conditioning variables
        grid_size: Number of grid points for integration

    Returns:
        Conditional expectation value

    Example:
        >>> # Compute E[y | x=0.3] for 2D joint p(x, y)
        >>> expected_y = conditional_expectation(
        ...     joint, target_index=1, given_indices=[0], given_values=[0.3]
        ... )
    """
    given_indices = list(given_indices)
    given_values = list(given_values)

    # Get conditional density
    grid, density = conditional_density(
        joint,
        given_indices=given_indices,
        given_values=given_values,
        target_indices=[target_index],
        grid_size=grid_size,
    )

    # E[x] = ∫ x p(x) dx ≈ Σ x_i p(x_i) Δx / Σ p(x_i) Δx
    # Since density is already normalized, just compute weighted mean
    expectation = np.trapz(grid * density, grid) / np.trapz(density, grid)

    return float(expectation)


def conditional_variance(
    joint: JointDensity,
    target_index: int,
    given_indices: Sequence[int],
    given_values: Sequence[float],
    grid_size: int = 100,
) -> float:
    """
    Compute conditional variance Var[x_target | x_given = given_values].

    Uses Var[X] = E[X^2] - E[X]^2

    Args:
        joint: Fitted JointDensity object
        target_index: Index of variable
        given_indices: Indices of conditioning variables
        given_values: Values for conditioning variables
        grid_size: Number of grid points for integration

    Returns:
        Conditional variance value
    """
    given_indices = list(given_indices)
    given_values = list(given_values)

    grid, density = conditional_density(
        joint,
        given_indices=given_indices,
        given_values=given_values,
        target_indices=[target_index],
        grid_size=grid_size,
    )

    # E[X] and E[X^2]
    normalizer = np.trapz(density, grid)
    ex = np.trapz(grid * density, grid) / normalizer
    ex2 = np.trapz(grid**2 * density, grid) / normalizer

    variance = ex2 - ex**2
    return float(max(0, variance))  # Clamp numerical errors


def conditional_mode(
    joint: JointDensity,
    target_index: int,
    given_indices: Sequence[int],
    given_values: Sequence[float],
    grid_size: int = 100,
) -> float:
    """
    Compute conditional mode (most likely value) of target given conditions.

    Args:
        joint: Fitted JointDensity object
        target_index: Index of variable
        given_indices: Indices of conditioning variables
        given_values: Values for conditioning variables
        grid_size: Number of grid points

    Returns:
        Value of target variable with highest conditional density
    """
    grid, density = conditional_density(
        joint,
        given_indices=given_indices,
        given_values=given_values,
        target_indices=[target_index],
        grid_size=grid_size,
    )

    mode_idx = np.argmax(density)
    return float(grid[mode_idx])


def marginal_density(
    joint: JointDensity,
    target_indices: Sequence[int],
    grid_size: int = 50,
    n_mc_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal density over a subset of variables.

    Marginalizes out non-target variables via Monte Carlo integration.

    Args:
        joint: Fitted JointDensity object
        target_indices: Indices of variables to keep
        grid_size: Number of grid points per target dimension
        n_mc_samples: Number of MC samples for marginalization

    Returns:
        Tuple of (grid_points, density_values)
    """
    target_indices = list(target_indices)
    other_indices = [i for i in range(joint.dim) if i not in target_indices]

    n_target = len(target_indices)

    # Create grid over target variables
    grids_1d = [np.linspace(0.001, 0.999, grid_size) for _ in range(n_target)]

    if n_target == 1:
        target_grid = grids_1d[0]
        n_grid_points = grid_size
    else:
        mesh = np.meshgrid(*grids_1d, indexing='ij')
        target_grid = np.stack([m.ravel() for m in mesh], axis=1)
        n_grid_points = target_grid.shape[0]

    # Monte Carlo integration over other variables
    rng = np.random.default_rng(42)

    density_values = np.zeros(n_grid_points)

    for i in range(n_grid_points):
        # Sample random values for other variables
        mc_points = np.zeros((n_mc_samples, joint.dim))

        # Set target values
        if n_target == 1:
            mc_points[:, target_indices[0]] = target_grid[i]
        else:
            for j, idx in enumerate(target_indices):
                mc_points[:, idx] = target_grid[i, j]

        # Random values for other variables
        for idx in other_indices:
            mc_points[:, idx] = rng.uniform(0, 1, n_mc_samples)

        # Average joint density over MC samples
        rho = joint.density(mc_points, clamp=True)
        density_values[i] = np.mean(rho)

    # Normalize
    if n_target == 1:
        normalizer = np.trapz(density_values, target_grid)
    else:
        normalizer = np.mean(density_values)

    if normalizer > 0:
        density_values = density_values / normalizer

    return target_grid, density_values


def sample_conditional(
    joint: JointDensity,
    given_indices: Sequence[int],
    given_values: Sequence[float],
    target_indices: Sequence[int] | None = None,
    n_samples: int = 100,
    grid_size: int = 100,
) -> np.ndarray:
    """
    Sample from conditional distribution p(x_target | x_given).

    Uses inverse CDF sampling for 1D target, or rejection sampling
    for multi-dimensional targets.

    Args:
        joint: Fitted JointDensity object
        given_indices: Indices of conditioning variables
        given_values: Values for conditioning variables
        target_indices: Indices of variables to sample (default: all non-given)
        n_samples: Number of samples to generate
        grid_size: Grid size for density estimation

    Returns:
        Array of shape (n_samples, n_target) with samples
    """
    given_indices = list(given_indices)
    given_values = list(given_values)

    if target_indices is None:
        target_indices = [i for i in range(joint.dim) if i not in given_indices]
    else:
        target_indices = list(target_indices)

    n_target = len(target_indices)

    # Get conditional density on grid
    grid, density = conditional_density(
        joint,
        given_indices=given_indices,
        given_values=given_values,
        target_indices=target_indices,
        grid_size=grid_size,
    )

    rng = np.random.default_rng()

    if n_target == 1:
        # 1D: inverse CDF sampling
        # Build CDF
        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]  # Normalize

        # Sample uniform and invert
        u = rng.uniform(0, 1, n_samples)
        samples = np.interp(u, cdf, grid)
        return samples.reshape(-1, 1)

    else:
        # Multi-D: rejection sampling
        max_density = np.max(density) * 1.2

        samples = []
        while len(samples) < n_samples:
            # Propose from uniform
            candidates = rng.uniform(0, 1, size=(n_samples * 10, n_target))

            # Evaluate density at candidates
            full_points = np.zeros((len(candidates), joint.dim))
            for idx, val in zip(given_indices, given_values):
                full_points[:, idx] = val
            for i, idx in enumerate(target_indices):
                full_points[:, idx] = candidates[:, i]

            rho = joint.density(full_points, clamp=True)

            # Compute marginal for normalization
            # This is approximate - we use the grid-based normalization
            grid_marginal, _ = conditional_density(
                joint, given_indices, given_values, target_indices, grid_size=20
            )

            # Accept/reject
            u = rng.uniform(0, max_density, len(candidates))
            accepted = candidates[u < rho]
            samples.extend(accepted)

        return np.array(samples[:n_samples])
