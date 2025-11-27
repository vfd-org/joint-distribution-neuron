#!/usr/bin/env python3
"""
Demo: 2D Correlated Gaussian with Bidirectional Inference


This script demonstrates the HCR neuron's key capability: learning a joint
distribution and performing inference in any direction.

We:
1. Generate samples from a 2D correlated Gaussian
2. Map to [0,1]^2 via CDF transformation (copula-style)
3. Fit a JointDensity using polynomial basis
4. Demonstrate bidirectional inference:
   - Forward: p(y | x) and E[y | x]
   - Reverse: p(x | y) and E[x | y]
5. Visualize results

This showcases how a single learned joint representation enables symmetric
inference, unlike traditional feedforward networks.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcrnn.basis import build_tensor_basis
from hcrnn.joint_density import JointDensity
from hcrnn.conditionals import (
    conditional_density,
    conditional_expectation,
    sample_conditional,
)


def generate_correlated_gaussian(
    n_samples: int,
    rho: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated bivariate Gaussian mapped to [0,1]^2."""
    rng = np.random.default_rng(seed)

    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    X_gaussian = rng.multivariate_normal(mean, cov, size=n_samples)

    # Map through CDF to get uniform marginals
    X_uniform = np.column_stack([
        stats.norm.cdf(X_gaussian[:, 0]),
        stats.norm.cdf(X_gaussian[:, 1]),
    ])

    return X_uniform, X_gaussian


def true_conditional_expectation(x_given: float, rho: float) -> float:
    """
    True E[Y | X = x] for bivariate Gaussian mapped through CDF.

    For standard bivariate normal with correlation rho:
    E[Y | X = x] = rho * x (in original space)

    In CDF-transformed space, this becomes more complex but can be computed.
    """
    x_original = stats.norm.ppf(x_given)
    y_expected_original = rho * x_original
    return stats.norm.cdf(y_expected_original)


def plot_results(
    X: np.ndarray,
    joint: JointDensity,
    rho: float,
    x0: float = 0.3,
    y0: float = 0.7,
):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # --- Plot 1: Data scatter with joint density contour ---
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=5, label='Data')

    # Density contour
    grid_1d = np.linspace(0.02, 0.98, 50)
    xx, yy = np.meshgrid(grid_1d, grid_1d)
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    density_grid = joint.density(grid_2d, clamp=True).reshape(50, 50)

    contour = ax.contour(xx, yy, density_grid, levels=10, cmap='viridis', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Data + Learned Joint Density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Mark conditioning points
    ax.axvline(x0, color='red', linestyle='--', alpha=0.7, label=f'x={x0}')
    ax.axhline(y0, color='blue', linestyle='--', alpha=0.7, label=f'y={y0}')
    ax.legend(loc='upper left', fontsize=8)

    # --- Plot 2: p(y | x = x0) ---
    ax = axes[0, 1]
    grid_y, density_y = conditional_density(
        joint, given_indices=[0], given_values=[x0], target_indices=[1], grid_size=100
    )
    ax.plot(grid_y, density_y, 'r-', linewidth=2, label=f'Estimated p(y|x={x0})')
    ax.fill_between(grid_y, density_y, alpha=0.3, color='red')

    # Mark conditional expectation
    exp_y = conditional_expectation(
        joint, target_index=1, given_indices=[0], given_values=[x0]
    )
    true_exp_y = true_conditional_expectation(x0, rho)
    ax.axvline(exp_y, color='darkred', linestyle='-', linewidth=2, label=f'E[y|x={x0}]={exp_y:.3f}')
    ax.axvline(true_exp_y, color='green', linestyle='--', linewidth=2, label=f'True E={true_exp_y:.3f}')

    ax.set_xlabel('Y')
    ax.set_ylabel('Density')
    ax.set_title(f'Forward: p(Y | X={x0})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # --- Plot 3: p(x | y = y0) ---
    ax = axes[0, 2]
    grid_x, density_x = conditional_density(
        joint, given_indices=[1], given_values=[y0], target_indices=[0], grid_size=100
    )
    ax.plot(grid_x, density_x, 'b-', linewidth=2, label=f'Estimated p(x|y={y0})')
    ax.fill_between(grid_x, density_x, alpha=0.3, color='blue')

    # Mark conditional expectation
    exp_x = conditional_expectation(
        joint, target_index=0, given_indices=[1], given_values=[y0]
    )
    true_exp_x = true_conditional_expectation(y0, rho)  # Symmetric for standard normal
    ax.axvline(exp_x, color='darkblue', linestyle='-', linewidth=2, label=f'E[x|y={y0}]={exp_x:.3f}')
    ax.axvline(true_exp_x, color='green', linestyle='--', linewidth=2, label=f'True E={true_exp_x:.3f}')

    ax.set_xlabel('X')
    ax.set_ylabel('Density')
    ax.set_title(f'Reverse: p(X | Y={y0})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # --- Plot 4: E[y | x] curve ---
    ax = axes[1, 0]
    x_vals = np.linspace(0.1, 0.9, 10)  # Reduced for speed
    exp_y_vals = [
        conditional_expectation(joint, target_index=1, given_indices=[0], given_values=[x])
        for x in x_vals
    ]
    true_exp_y_vals = [true_conditional_expectation(x, rho) for x in x_vals]

    ax.plot(x_vals, exp_y_vals, 'r-', linewidth=2, label='Estimated E[Y|X]')
    ax.plot(x_vals, true_exp_y_vals, 'g--', linewidth=2, label='True E[Y|X]')
    ax.set_xlabel('X')
    ax.set_ylabel('E[Y | X]')
    ax.set_title('Forward Regression: E[Y | X]')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Plot 5: E[x | y] curve ---
    ax = axes[1, 1]
    y_vals = np.linspace(0.1, 0.9, 10)  # Reduced for speed
    exp_x_vals = [
        conditional_expectation(joint, target_index=0, given_indices=[1], given_values=[y])
        for y in y_vals
    ]
    true_exp_x_vals = [true_conditional_expectation(y, rho) for y in y_vals]

    ax.plot(y_vals, exp_x_vals, 'b-', linewidth=2, label='Estimated E[X|Y]')
    ax.plot(y_vals, true_exp_x_vals, 'g--', linewidth=2, label='True E[X|Y]')
    ax.set_xlabel('Y')
    ax.set_ylabel('E[X | Y]')
    ax.set_title('Reverse Regression: E[X | Y]')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Plot 6: Conditional samples ---
    ax = axes[1, 2]

    # Samples from p(y | x=x0)
    samples_y = sample_conditional(
        joint, given_indices=[0], given_values=[x0], n_samples=100  # Reduced for speed
    )
    ax.scatter([x0] * len(samples_y), samples_y.flatten(), alpha=0.5, s=20, c='red', label=f'p(y|x={x0})')

    # Samples from p(x | y=y0)
    samples_x = sample_conditional(
        joint, given_indices=[1], given_values=[y0], n_samples=100  # Reduced for speed
    )
    ax.scatter(samples_x.flatten(), [y0] * len(samples_x), alpha=0.5, s=20, c='blue', label=f'p(x|y={y0})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Conditional Samples')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("HCR Neuron Demo: 2D Correlated Gaussian")
    print("=" * 60)

    # Parameters
    n_samples = 2000  # Reduced for speed
    rho = 0.7  # Correlation coefficient
    degree = 3  # Polynomial degree per dimension (reduced for speed)

    print(f"\n1. Generating {n_samples} samples from bivariate Gaussian")
    print(f"   Correlation: ρ = {rho}")

    X_uniform, X_gaussian = generate_correlated_gaussian(n_samples, rho)
    print(f"   Data shape: {X_uniform.shape}")
    print(f"   X range: [{X_uniform[:, 0].min():.3f}, {X_uniform[:, 0].max():.3f}]")
    print(f"   Y range: [{X_uniform[:, 1].min():.3f}, {X_uniform[:, 1].max():.3f}]")

    # Build basis and fit density
    print(f"\n2. Building polynomial basis (degree {degree} per dimension)")
    basis = build_tensor_basis(dim=2, degrees_per_dim=degree)
    print(f"   Number of basis functions: {basis.num_basis}")

    print("\n3. Fitting joint density...")
    joint = JointDensity(basis)
    joint.fit(X_uniform)

    # Check integral
    integral = joint.integrate(n_samples=20000)
    print(f"   Density integrates to: {integral:.4f} (should be ≈1.0)")

    # Demonstrate bidirectional inference
    print("\n4. Bidirectional Inference Demo")
    print("-" * 40)

    x0, y0 = 0.3, 0.7

    # Forward: E[Y | X = x0]
    exp_y_given_x = conditional_expectation(
        joint, target_index=1, given_indices=[0], given_values=[x0]
    )
    true_exp_y = true_conditional_expectation(x0, rho)
    print(f"\n   Forward inference at x = {x0}:")
    print(f"   Estimated E[Y | X={x0}] = {exp_y_given_x:.4f}")
    print(f"   True      E[Y | X={x0}] = {true_exp_y:.4f}")
    print(f"   Error: {abs(exp_y_given_x - true_exp_y):.4f}")

    # Reverse: E[X | Y = y0]
    exp_x_given_y = conditional_expectation(
        joint, target_index=0, given_indices=[1], given_values=[y0]
    )
    true_exp_x = true_conditional_expectation(y0, rho)
    print(f"\n   Reverse inference at y = {y0}:")
    print(f"   Estimated E[X | Y={y0}] = {exp_x_given_y:.4f}")
    print(f"   True      E[X | Y={y0}] = {true_exp_x:.4f}")
    print(f"   Error: {abs(exp_x_given_y - true_exp_x):.4f}")

    # Compute errors across range
    print("\n5. Error Analysis")
    print("-" * 40)
    test_points = np.linspace(0.15, 0.85, 7)  # Reduced for speed
    errors_forward = []
    errors_reverse = []

    for t in test_points:
        est_forward = conditional_expectation(
            joint, target_index=1, given_indices=[0], given_values=[t]
        )
        true_forward = true_conditional_expectation(t, rho)
        errors_forward.append(abs(est_forward - true_forward))

        est_reverse = conditional_expectation(
            joint, target_index=0, given_indices=[1], given_values=[t]
        )
        true_reverse = true_conditional_expectation(t, rho)
        errors_reverse.append(abs(est_reverse - true_reverse))

    print(f"   Mean Absolute Error (forward E[Y|X]): {np.mean(errors_forward):.4f}")
    print(f"   Mean Absolute Error (reverse E[X|Y]): {np.mean(errors_reverse):.4f}")
    print(f"   Max Error (forward):  {np.max(errors_forward):.4f}")
    print(f"   Max Error (reverse):  {np.max(errors_reverse):.4f}")

    # Sampling demonstration
    print("\n6. Conditional Sampling")
    print("-" * 40)
    samples_y = sample_conditional(
        joint, given_indices=[0], given_values=[0.5], n_samples=200  # Reduced for speed
    )
    print(f"   Samples from p(Y | X=0.5):")
    print(f"   Mean: {samples_y.mean():.4f}, Std: {samples_y.std():.4f}")
    print(f"   (True mean should be close to 0.5 for X=0.5)")

    # Create plots
    print("\n7. Creating visualization...")
    fig = plot_results(X_uniform, joint, rho, x0=x0, y0=y0)

    # Save figure
    output_path = Path(__file__).parent / "demo_output.png"
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"   Saved plot to: {output_path}")

    plt.close(fig)  # Free memory

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    # Summary
    print("""
Key Observations:
-----------------
1. The HCR neuron learns a single joint density p(x, y)
2. From this ONE representation, we can compute:
   - p(y | x) (forward inference)
   - p(x | y) (reverse inference)
   - Any conditional expectations, modes, samples
3. This is fundamentally different from traditional NNs that only
   learn p(y | x) and cannot do reverse inference

Limitations:
------------
1. Polynomial representation can go negative in some regions
2. Accuracy depends on polynomial degree and data size
3. Scales poorly to high dimensions (curse of dimensionality)
4. Numerical integration limits practical dimensionality

Next Steps:
-----------
1. Multi-layer HCR networks (stacking neurons)
2. More sophisticated basis functions
3. VFD-style continuous field representations
""")


if __name__ == "__main__":
    main()
