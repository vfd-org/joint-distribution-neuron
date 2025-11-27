#!/usr/bin/env python3
"""
Demo: Multi-Layer HCR Network (HCRNN)

This script demonstrates the key capabilities of multi-layer HCR networks:

1. Stacked bidirectional inference - forward AND reverse through multiple layers
2. Uncertainty propagation - track how uncertainty contracts/expands
3. Learned reversible mappings - approximate invertibility
4. Resonance-based coherence - smooth, stable representations

The multi-layer architecture enables:
- Hierarchical feature learning
- Distributed constraint satisfaction
- Automatic irrelevant correlation filtering (via resonance decay)

This is where HCR neurons begin to act as a coherent network.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hcrnn.network import HCRNetwork, LayerSpec


def generate_spiral_data(
    n_samples: int,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D -> 2D spiral mapping data.

    Input: points on a 2D spiral
    Output: transformed version (rotation + scaling)
    """
    rng = np.random.default_rng(seed)

    # Generate spiral in input space
    t = np.linspace(0, 4 * np.pi, n_samples)
    r = t / (4 * np.pi)

    X = np.column_stack([
        r * np.cos(t) + rng.standard_normal(n_samples) * noise,
        r * np.sin(t) + rng.standard_normal(n_samples) * noise,
    ])

    # Output: rotated and scaled version
    theta = np.pi / 3
    scale = 1.5
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    Y = scale * (X @ R.T) + rng.standard_normal((n_samples, 2)) * noise

    return X, Y


def generate_xor_like_data(
    n_samples: int,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate nonlinear XOR-like transformation data.

    A classic nonlinear problem that requires hidden representations.
    """
    rng = np.random.default_rng(seed)

    # Input: random points in [-1, 1]^2
    X = rng.uniform(-1, 1, size=(n_samples, 2))

    # Output: XOR-like nonlinear transformation
    Y = np.column_stack([
        X[:, 0] * X[:, 1],  # Product (nonlinear)
        np.sin(X[:, 0] * np.pi) * np.cos(X[:, 1] * np.pi),  # Trigonometric
    ])
    Y += rng.standard_normal((n_samples, 2)) * noise

    return X, Y


def plot_multilayer_results(
    net: HCRNetwork,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> plt.Figure:
    """Create visualization of multilayer network results."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # --- Plot 1: Input space ---
    ax = axes[0, 0]
    ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, s=20, label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.7, s=30, label='Test')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('Input Space')
    ax.legend()

    # --- Plot 2: Output space (target vs predicted) ---
    ax = axes[0, 1]
    Y_pred = net.forward(X_test)
    ax.scatter(Y_test[:, 0], Y_test[:, 1], c='green', alpha=0.5, s=20, label='Target')
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], c='orange', alpha=0.7, s=30, label='Predicted')
    ax.set_xlabel('Y₁')
    ax.set_ylabel('Y₂')
    ax.set_title('Output Space: Target vs Predicted')
    ax.legend()

    # --- Plot 3: Reverse inference ---
    ax = axes[0, 2]
    X_recon = net.reverse(Y_test)
    ax.scatter(X_test[:, 0], X_test[:, 1], c='blue', alpha=0.5, s=20, label='Original')
    ax.scatter(X_recon[:, 0], X_recon[:, 1], c='purple', alpha=0.7, s=30, label='Reconstructed')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title('Reverse Inference: Y → X')
    ax.legend()

    # --- Plot 4: Layer activations (forward) ---
    ax = axes[1, 0]
    _, intermediates = net.forward(X_test, return_intermediates=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(intermediates)))
    for i, (act, color) in enumerate(zip(intermediates, colors)):
        if act.shape[1] >= 2:
            ax.scatter(act[:, 0], act[:, 1], c=[color], alpha=0.5, s=15, label=f'Layer {i}')
        else:
            ax.scatter(act[:, 0], np.zeros(len(act)), c=[color], alpha=0.5, s=15, label=f'Layer {i}')

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_title('Layer Activations (Forward)')
    ax.legend(fontsize=8)

    # --- Plot 5: Uncertainty across layers ---
    ax = axes[1, 1]
    uncertainties = []
    for layer_idx in range(net.num_layers):
        var = net.compute_uncertainty(X_test, layer_idx=layer_idx)
        mean_var = np.mean(var)
        uncertainties.append(mean_var)

    ax.bar(range(len(uncertainties)), uncertainties, color='coral')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Variance')
    ax.set_title('Uncertainty by Layer')
    ax.set_xticks(range(len(uncertainties)))

    # --- Plot 6: Forward-Reverse cycle error ---
    ax = axes[1, 2]

    # Compute cycle errors for each test point
    Y_pred = net.forward(X_test)
    X_cycle = net.reverse(Y_pred)
    cycle_errors = np.sqrt(np.sum((X_test - X_cycle) ** 2, axis=1))

    ax.hist(cycle_errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(cycle_errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(cycle_errors):.3f}')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title('Cycle Consistency: X → Y → X')
    ax.legend()

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Multi-Layer HCR Network (HCRNN) Demo")
    print("=" * 70)

    # =========================================================================
    # PART 1: Simple 2-layer network on spiral data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: 2-Layer Network on Spiral Transformation")
    print("=" * 70)

    n_train, n_test = 300, 50
    X_train, Y_train = generate_spiral_data(n_train, noise=0.05)
    X_test, Y_test = generate_spiral_data(n_test, noise=0.05, seed=123)

    print(f"\n1. Generated {n_train} training samples, {n_test} test samples")
    print(f"   Input shape: {X_train.shape}, Output shape: {Y_train.shape}")

    # Define network architecture
    specs = [
        LayerSpec(input_dim=2, output_dim=4, basis_degree=2, resonance_decay=0.1),
        LayerSpec(input_dim=4, output_dim=2, basis_degree=2, resonance_decay=0.1),
    ]

    print("\n2. Creating 2-layer HCRNN:")
    print("   Architecture: 2 → 4 → 2")
    print("   Basis degree: 2 per layer")

    net = HCRNetwork(specs, seed=42)
    print(f"   Network: {net}")

    print("\n3. Training with alternating least squares...")
    net.fit(X_train, Y_train, method="alternating", max_iter=5, verbose=True)

    print("\n4. Evaluating performance:")
    errors = net.reconstruction_error(X_test, Y_test)
    print(f"   Forward RMSE:  {errors['forward_rmse']:.4f}")
    print(f"   Reverse RMSE:  {errors['reverse_rmse']:.4f}")
    print(f"   Cycle MSE:     {errors['cycle_mse']:.4f}")

    # Test bidirectional inference
    print("\n5. Bidirectional Inference Demo:")
    x_sample = X_test[0:1]
    y_true = Y_test[0:1]

    y_pred = net.forward(x_sample)
    x_recon = net.reverse(y_pred)

    print(f"   Input X:        {x_sample[0]}")
    print(f"   Target Y:       {y_true[0]}")
    print(f"   Predicted Y:    {y_pred[0]}")
    print(f"   Reconstructed X: {x_recon[0]}")
    print(f"   X reconstruction error: {np.linalg.norm(x_sample - x_recon):.4f}")

    # Create visualization
    print("\n6. Creating visualization...")
    fig1 = plot_multilayer_results(net, X_train, Y_train, X_test, Y_test)
    output_path1 = Path(__file__).parent / "hcrnn_spiral_demo.png"
    fig1.savefig(output_path1, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {output_path1}")
    plt.close(fig1)

    # =========================================================================
    # PART 2: 3-layer network with bottleneck (autoencoder-like)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: 3-Layer Bottleneck Network (Autoencoder-like)")
    print("=" * 70)

    # Use XOR-like data for more challenge
    X_train2, Y_train2 = generate_xor_like_data(400, noise=0.05)
    X_test2, Y_test2 = generate_xor_like_data(60, noise=0.05, seed=456)

    print(f"\n1. Generated XOR-like nonlinear transformation data")
    print(f"   Input/Output: 2D → 2D")

    # Bottleneck architecture: compress then expand
    specs2 = [
        LayerSpec(input_dim=2, output_dim=4, basis_degree=2),
        LayerSpec(input_dim=4, output_dim=2, basis_degree=2),  # Bottleneck
        LayerSpec(input_dim=2, output_dim=2, basis_degree=2),
    ]

    print("\n2. Creating 3-layer bottleneck HCRNN:")
    print("   Architecture: 2 → 4 → 2 → 2")

    net2 = HCRNetwork(specs2, seed=42)

    print("\n3. Training...")
    net2.fit(X_train2, Y_train2, method="alternating", max_iter=5, verbose=True)

    print("\n4. Evaluating:")
    errors2 = net2.reconstruction_error(X_test2, Y_test2)
    print(f"   Forward RMSE: {errors2['forward_rmse']:.4f}")
    print(f"   Reverse RMSE: {errors2['reverse_rmse']:.4f}")

    # Show layer-by-layer uncertainty
    print("\n5. Uncertainty propagation through layers:")
    for i in range(net2.num_layers):
        var = net2.compute_uncertainty(X_test2, layer_idx=i)
        print(f"   Layer {i}: mean variance = {np.mean(var):.4f}")

    # Create second visualization
    print("\n6. Creating visualization...")
    fig2 = plot_multilayer_results(net2, X_train2, Y_train2, X_test2, Y_test2)
    output_path2 = Path(__file__).parent / "hcrnn_bottleneck_demo.png"
    fig2.savefig(output_path2, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {output_path2}")
    plt.close(fig2)

    # =========================================================================
    # PART 3: Demonstrate resonance regularization effect
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 3: Resonance Regularization Effect")
    print("=" * 70)

    print("\nComparing networks with different resonance decay rates:")

    X_train3, Y_train3 = generate_spiral_data(200, noise=0.1)
    X_test3, Y_test3 = generate_spiral_data(30, noise=0.1, seed=789)

    decay_rates = [0.0, 0.1, 0.5]
    results = []

    for decay in decay_rates:
        specs_test = [
            LayerSpec(input_dim=2, output_dim=4, basis_degree=3, resonance_decay=decay),
            LayerSpec(input_dim=4, output_dim=2, basis_degree=3, resonance_decay=decay),
        ]
        net_test = HCRNetwork(specs_test, seed=42)
        net_test.fit(X_train3, Y_train3, method="alternating", max_iter=3, verbose=False)

        errors_test = net_test.reconstruction_error(X_test3, Y_test3)

        # Compute total coefficient magnitude (proxy for complexity)
        total_coeff_norm = 0
        for layer in net_test.layers:
            if layer.joint.coeffs is not None:
                total_coeff_norm += np.sum(layer.joint.coeffs ** 2)

        results.append({
            'decay': decay,
            'forward_rmse': errors_test['forward_rmse'],
            'coeff_norm': total_coeff_norm,
        })

        print(f"   Decay={decay:.1f}: Forward RMSE={errors_test['forward_rmse']:.4f}, "
              f"Coeff Norm={total_coeff_norm:.2f}")

    print("\n   → Higher resonance decay encourages smoother, lower-complexity solutions")
    print("   → This implements the 'coherence filter' from VFD theory")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("""
Key Observations:
-----------------
1. MULTI-LAYER BIDIRECTIONAL INFERENCE
   - Forward: X → hidden → Y
   - Reverse: Y → hidden → X
   - Both directions work through the SAME learned joint distributions

2. UNCERTAINTY PROPAGATION
   - Variance tracked at each layer
   - Shows how information flows through the network
   - Can identify bottlenecks and information loss

3. CYCLE CONSISTENCY
   - X → Y → X should recover original (approximately)
   - Measures the "reversibility" of the learned mapping
   - Lower cycle error = better bidirectional model

4. RESONANCE REGULARIZATION
   - Penalizes high-frequency (high-order) polynomial coefficients
   - Encourages smooth, stable, coherent representations
   - Implements VFD "coherence filter" principle

What Makes HCRNN Special:
-------------------------
- Traditional NNs: learn p(Y|X), cannot reverse
- HCRNNs: learn p(X,Y) at each layer, enabling:
  * Forward inference
  * Reverse inference
  * Uncertainty quantification
  * Constraint propagation

This is the foundation for:
- Field-based computation
- Bidirectional reasoning
- Coherent distributed representations
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
