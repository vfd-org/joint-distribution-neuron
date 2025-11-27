#!/usr/bin/env python3
"""
Demo: 3D Joint Density with Marginalization and Conditioning (v0.3)

This script demonstrates the v0.3 features:
1. N-dimensional joint density estimation using total-degree basis
2. Marginalization: p(x0, x1, x2) -> p(x0, x1) or p(x0)
3. Conditioning: p(x0, x1 | x2 = fixed_value)
4. Comparison of total-degree vs tensor-product basis efficiency

The key insight is that for orthonormal polynomial bases:
- Marginalization is exact (integration picks out degree-0 terms)
- Conditioning creates a slice through the joint distribution
- Total-degree basis scales better: O(d+k choose k) vs O(k^d)
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hcrnn.basis import build_total_degree_basis, build_tensor_basis, count_total_degree_terms
from hcrnn.joint_density import JointDensity


def generate_3d_correlated_data(
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate 3D correlated data from a multivariate Gaussian,
    then transform to [0,1]^3 via CDF.

    The correlation structure:
    - x0 and x1 are positively correlated
    - x0 and x2 are negatively correlated
    - x1 and x2 are uncorrelated
    """
    rng = np.random.default_rng(seed)

    # Covariance matrix with specific correlations
    cov = np.array([
        [1.0,  0.7, -0.5],
        [0.7,  1.0,  0.0],
        [-0.5, 0.0,  1.0],
    ])

    mean = np.array([0.0, 0.0, 0.0])

    # Sample from multivariate normal
    X_normal = rng.multivariate_normal(mean, cov, size=n_samples)

    # Transform to [0,1] via standard normal CDF
    from scipy import stats
    X_uniform = stats.norm.cdf(X_normal)

    return X_uniform


def generate_3d_nonlinear_data(
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate 3D data with nonlinear dependencies.

    x0 ~ Uniform[0,1]
    x1 ~ Beta(2, 2) (centered)
    x2 = f(x0, x1) + noise (nonlinear combination)
    """
    rng = np.random.default_rng(seed)

    x0 = rng.uniform(0, 1, n_samples)
    x1 = rng.beta(2, 2, n_samples)

    # Nonlinear dependence: x2 depends on product and sum
    x2_mean = 0.3 * x0 + 0.3 * x1 + 0.4 * x0 * x1
    x2 = np.clip(x2_mean + rng.normal(0, 0.1, n_samples), 0, 1)

    return np.column_stack([x0, x1, x2])


def compare_basis_scaling():
    """Compare number of basis functions for total-degree vs tensor-product."""
    print("\n" + "=" * 60)
    print("Basis Scaling Comparison")
    print("=" * 60)

    print(f"\n{'Dim':>4} {'Degree':>6} {'Total-Deg':>12} {'Tensor':>12} {'Ratio':>8}")
    print("-" * 50)

    for dim in [2, 3, 4, 5]:
        for degree in [2, 3, 4]:
            n_total = count_total_degree_terms(dim, degree)
            n_tensor = (degree + 1) ** dim
            ratio = n_tensor / n_total
            print(f"{dim:>4} {degree:>6} {n_total:>12} {n_tensor:>12} {ratio:>8.1f}x")

    print("\n-> Total-degree basis is increasingly efficient in higher dimensions")


def demo_marginalization(density: JointDensity, X: np.ndarray):
    """Demonstrate marginalization operations."""
    print("\n" + "=" * 60)
    print("Marginalization Demo")
    print("=" * 60)

    print(f"\nOriginal density: dim={density.dim}, basis_size={density.basis.num_basis}")

    # Marginalize to get p(x0, x1)
    marginal_01 = density.marginalize([0, 1])
    print(f"p(x0, x1): dim={marginal_01.dim}, basis_size={marginal_01.basis.num_basis}")

    # Marginalize to get p(x0)
    marginal_0 = density.marginalize([0])
    print(f"p(x0):     dim={marginal_0.dim}, basis_size={marginal_0.basis.num_basis}")

    # Marginalize to get p(x2)
    marginal_2 = density.marginalize([2])
    print(f"p(x2):     dim={marginal_2.dim}, basis_size={marginal_2.basis.num_basis}")

    # Verify marginals integrate to 1
    print("\nVerifying marginals integrate to 1:")
    for name, marg in [("p(x0,x1)", marginal_01), ("p(x0)", marginal_0), ("p(x2)", marginal_2)]:
        integral = marg.integrate(n_samples=10000)
        print(f"  {name}: integral = {integral:.4f}")

    return marginal_01, marginal_0, marginal_2


def demo_conditioning(density: JointDensity, X: np.ndarray):
    """Demonstrate conditioning operations."""
    print("\n" + "=" * 60)
    print("Conditioning Demo")
    print("=" * 60)

    # Condition on x2 = 0.3 (low) and x2 = 0.7 (high)
    x2_values = [0.3, 0.5, 0.7]
    conditionals = {}

    print("\nConditioning on different x2 values:")
    for x2_val in x2_values:
        cond = density.condition_on({2: x2_val})
        conditionals[x2_val] = cond
        print(f"  p(x0, x1 | x2={x2_val}): {cond}")

        # Compute expected values
        expected = cond.expected_value(n_samples=5000)
        print(f"    E[x0 | x2={x2_val}] = {expected[0]:.4f}")
        print(f"    E[x1 | x2={x2_val}] = {expected[1]:.4f}")

    # Also condition on x0 to get 1D conditional
    print("\nConditioning to 1D:")
    cond_1d = density.condition_on({0: 0.5, 1: 0.5})
    print(f"  p(x2 | x0=0.5, x1=0.5): {cond_1d}")
    expected_x2 = cond_1d.expected_value(n_samples=5000)
    print(f"    E[x2 | x0=0.5, x1=0.5] = {expected_x2[0]:.4f}")

    return conditionals


def plot_3d_demo_results(
    X: np.ndarray,
    density: JointDensity,
    marginal_01: JointDensity,
    marginal_0: JointDensity,
    conditionals: dict,
) -> plt.Figure:
    """Create visualization of 3D density results."""
    fig = plt.figure(figsize=(15, 10))

    # --- Plot 1: 3D scatter of original data ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis',
                alpha=0.5, s=10)
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('x2')
    ax1.set_title('3D Training Data')

    # --- Plot 2: Marginal p(x0, x1) as heatmap ---
    ax2 = fig.add_subplot(2, 3, 2)
    n_grid = 30
    x0_grid = np.linspace(0.05, 0.95, n_grid)
    x1_grid = np.linspace(0.05, 0.95, n_grid)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    grid_points = np.column_stack([X0.ravel(), X1.ravel()])

    rho_marginal = marginal_01.density(grid_points, clamp=True)
    rho_marginal = rho_marginal.reshape(n_grid, n_grid)

    im = ax2.imshow(rho_marginal, origin='lower', aspect='auto',
                    extent=[0, 1, 0, 1], cmap='hot')
    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    ax2.set_title('Marginal p(x0, x1)')
    plt.colorbar(im, ax=ax2, label='density')

    # --- Plot 3: Marginal p(x0) ---
    ax3 = fig.add_subplot(2, 3, 3)
    x0_1d = np.linspace(0.05, 0.95, 100).reshape(-1, 1)
    rho_x0 = marginal_0.density(x0_1d, clamp=True)
    ax3.plot(x0_1d, rho_x0, 'b-', linewidth=2)
    ax3.hist(X[:, 0], bins=30, density=True, alpha=0.3, color='blue', label='Data histogram')
    ax3.set_xlabel('x0')
    ax3.set_ylabel('density')
    ax3.set_title('Marginal p(x0)')
    ax3.legend()

    # --- Plot 4: Conditional p(x0, x1 | x2) for different x2 ---
    ax4 = fig.add_subplot(2, 3, 4)
    colors = ['blue', 'green', 'red']
    for (x2_val, cond), color in zip(conditionals.items(), colors):
        # Sample from conditional
        samples = cond.sample(500)
        ax4.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c=color,
                   label=f'x2={x2_val}')
    ax4.set_xlabel('x0')
    ax4.set_ylabel('x1')
    ax4.set_title('Samples from p(x0, x1 | x2)')
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # --- Plot 5: Conditional expectations vs x2 ---
    ax5 = fig.add_subplot(2, 3, 5)
    x2_values = np.linspace(0.1, 0.9, 9)
    e_x0 = []
    e_x1 = []

    for x2_val in x2_values:
        cond = density.condition_on({2: x2_val})
        exp = cond.expected_value(n_samples=3000)
        e_x0.append(exp[0])
        e_x1.append(exp[1])

    ax5.plot(x2_values, e_x0, 'b-o', label='E[x0 | x2]')
    ax5.plot(x2_values, e_x1, 'r-s', label='E[x1 | x2]')
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x2')
    ax5.set_ylabel('Conditional expectation')
    ax5.set_title('E[x0], E[x1] as function of x2')
    ax5.legend()
    ax5.set_xlim(0, 1)

    # --- Plot 6: Basis size comparison ---
    ax6 = fig.add_subplot(2, 3, 6)
    dims = [2, 3, 4, 5, 6]
    degree = 3
    total_deg_sizes = [count_total_degree_terms(d, degree) for d in dims]
    tensor_sizes = [(degree + 1) ** d for d in dims]

    x_pos = np.arange(len(dims))
    width = 0.35
    ax6.bar(x_pos - width/2, total_deg_sizes, width, label='Total-degree')
    ax6.bar(x_pos + width/2, tensor_sizes, width, label='Tensor-product')
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('Number of basis functions')
    ax6.set_title(f'Basis size comparison (degree={degree})')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(dims)
    ax6.legend()
    ax6.set_yscale('log')

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("3D Joint Density Demo: Marginalization and Conditioning (v0.3)")
    print("=" * 70)

    # Compare basis scaling first
    compare_basis_scaling()

    # Generate 3D data
    print("\n" + "=" * 60)
    print("Generating 3D Correlated Data")
    print("=" * 60)

    n_samples = 2000
    X = generate_3d_correlated_data(n_samples, seed=42)

    print(f"\nGenerated {n_samples} samples in [0,1]^3")
    print("Correlation structure:")
    corr = np.corrcoef(X.T)
    print(f"  corr(x0, x1) = {corr[0, 1]:.3f} (positive)")
    print(f"  corr(x0, x2) = {corr[0, 2]:.3f} (negative)")
    print(f"  corr(x1, x2) = {corr[1, 2]:.3f} (near zero)")

    # Fit 3D density using total-degree basis
    print("\n" + "=" * 60)
    print("Fitting 3D Joint Density")
    print("=" * 60)

    total_degree = 3
    basis = build_total_degree_basis(dim=3, total_degree=total_degree)
    print(f"\nUsing total-degree basis: degree={total_degree}, num_basis={basis.num_basis}")
    print(f"  (Tensor-product would have {(total_degree + 1)**3} = {(total_degree + 1)**3} terms)")

    density = JointDensity(basis)
    density.fit(X, store_data=True)
    print(f"\nFitted density: {density}")

    # Verify it integrates to ~1
    integral = density.integrate(n_samples=50000)
    print(f"Integral over [0,1]^3: {integral:.4f}")

    # Demonstrate marginalization
    marginal_01, marginal_0, marginal_2 = demo_marginalization(density, X)

    # Demonstrate conditioning
    conditionals = demo_conditioning(density, X)

    # Create visualization
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)

    fig = plot_3d_demo_results(X, density, marginal_01, marginal_0, conditionals)
    output_path = Path(__file__).parent / "demo_3d_mixed_moments.png"
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Observations:
-----------------
1. TOTAL-DEGREE BASIS
   - More efficient than tensor-product for higher dimensions
   - For dim=5, degree=3: 56 terms vs 1024 (18x fewer)
   - Still captures correlations and dependencies

2. MARGINALIZATION
   - Analytically exact for orthonormal polynomial bases
   - Integration picks out degree-0 terms in marginalized dimensions
   - Produces valid lower-dimensional densities

3. CONDITIONING
   - Creates slices through the joint distribution
   - E[x0 | x2] shows how expected x0 changes with x2
   - Captures the correlation structure in the data

4. BIDIRECTIONAL INFERENCE
   - Same density supports:
     * p(x0, x1, x2) - full joint
     * p(x0, x1) - any marginal
     * p(x0 | x1=v1, x2=v2) - any conditional
   - This is the core of the "joint distribution neuron" concept
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
