"""Tests for topology builder functions."""

from __future__ import annotations

import numpy as np
import pytest

from hcrnn import (
    HCRNetwork,
    LayerSpec,
    build_default_network,
    build_network_from_topology,
)


class TestBuildDefaultNetwork:
    """Tests for build_default_network function."""

    def test_returns_hcrnetwork(self):
        """Should return an HCRNetwork instance."""
        net = build_default_network(dim_in=2, dim_out=2)
        assert isinstance(net, HCRNetwork)

    def test_correct_input_output_dims(self):
        """Network should have correct input/output dimensions."""
        net = build_default_network(dim_in=3, dim_out=5, hidden_layers=2)

        # First layer input
        assert net.specs[0].input_dim == 3

        # Last layer output
        assert net.specs[-1].output_dim == 5

    def test_hidden_layers_count(self):
        """Should create correct number of hidden layers."""
        # 2 hidden layers = 3 total layers (in→h1, h1→h2, h2→out)
        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=2)
        assert net.num_layers == 3

        # 0 hidden layers = 1 total layer (in→out)
        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        assert net.num_layers == 1

        # 5 hidden layers = 6 total layers
        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=5)
        assert net.num_layers == 6

    def test_hidden_width(self):
        """Hidden layers should have specified width."""
        net = build_default_network(
            dim_in=2, dim_out=3, hidden_layers=2, hidden_width=8
        )

        # First layer output = hidden width
        assert net.specs[0].output_dim == 8

        # Middle layers = hidden width
        assert net.specs[1].input_dim == 8
        assert net.specs[1].output_dim == 8

        # Last layer input = hidden width
        assert net.specs[2].input_dim == 8

    def test_degree_parameter(self):
        """All layers should use specified degree."""
        net = build_default_network(dim_in=2, dim_out=2, degree=5)

        for spec in net.specs:
            assert spec.basis_degree == 5

    def test_resonance_decay_parameter(self):
        """All layers should use specified resonance_decay."""
        net = build_default_network(dim_in=2, dim_out=2, resonance_decay=0.05)

        for spec in net.specs:
            assert spec.resonance_decay == 0.05

    def test_seed_parameter(self):
        """Same seed should produce same initialization."""
        net1 = build_default_network(dim_in=2, dim_out=2, seed=123)
        net2 = build_default_network(dim_in=2, dim_out=2, seed=123)

        np.testing.assert_array_equal(net1.layers[0].W, net2.layers[0].W)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initializations."""
        net1 = build_default_network(dim_in=2, dim_out=2, seed=123)
        net2 = build_default_network(dim_in=2, dim_out=2, seed=456)

        assert not np.allclose(net1.layers[0].W, net2.layers[0].W)

    def test_unfitted_by_default(self):
        """Network should be unfitted after creation."""
        net = build_default_network(dim_in=2, dim_out=2)
        assert not net._is_fitted

    def test_can_fit_after_build(self):
        """Built network should be trainable."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=1)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        assert net._is_fitted

    def test_invalid_hidden_layers(self):
        """Should reject negative hidden_layers."""
        with pytest.raises(ValueError, match="hidden_layers must be non-negative"):
            build_default_network(dim_in=2, dim_out=2, hidden_layers=-1)

    def test_invalid_hidden_width(self):
        """Should reject hidden_width < 1."""
        with pytest.raises(ValueError, match="hidden_width must be at least 1"):
            build_default_network(dim_in=2, dim_out=2, hidden_width=0)

    def test_invalid_degree(self):
        """Should reject degree < 1."""
        with pytest.raises(ValueError, match="degree must be at least 1"):
            build_default_network(dim_in=2, dim_out=2, degree=0)


class TestBuildNetworkFromTopology:
    """Tests for build_network_from_topology function."""

    def test_returns_hcrnetwork(self):
        """Should return an HCRNetwork instance."""
        net = build_network_from_topology([2, 4, 2])
        assert isinstance(net, HCRNetwork)

    def test_correct_layer_count(self):
        """Should create correct number of layers."""
        # [2, 4, 2] = 2 layers
        net = build_network_from_topology([2, 4, 2])
        assert net.num_layers == 2

        # [3, 8, 8, 4, 2] = 4 layers
        net = build_network_from_topology([3, 8, 8, 4, 2])
        assert net.num_layers == 4

    def test_correct_dimensions(self):
        """Layer dimensions should match topology."""
        topology = [3, 8, 4, 2]
        net = build_network_from_topology(topology)

        assert net.specs[0].input_dim == 3
        assert net.specs[0].output_dim == 8

        assert net.specs[1].input_dim == 8
        assert net.specs[1].output_dim == 4

        assert net.specs[2].input_dim == 4
        assert net.specs[2].output_dim == 2

    def test_degree_parameter(self):
        """All layers should use specified degree."""
        net = build_network_from_topology([2, 4, 2], degree=4)

        for spec in net.specs:
            assert spec.basis_degree == 4

    def test_resonance_decay_parameter(self):
        """All layers should use specified resonance_decay."""
        net = build_network_from_topology([2, 4, 2], resonance_decay=0.2)

        for spec in net.specs:
            assert spec.resonance_decay == 0.2

    def test_seed_parameter(self):
        """Same seed should produce same initialization."""
        net1 = build_network_from_topology([2, 4, 2], seed=42)
        net2 = build_network_from_topology([2, 4, 2], seed=42)

        np.testing.assert_array_equal(net1.layers[0].W, net2.layers[0].W)

    def test_minimal_topology(self):
        """Should work with 2-element topology (single layer)."""
        net = build_network_from_topology([5, 3])
        assert net.num_layers == 1
        assert net.specs[0].input_dim == 5
        assert net.specs[0].output_dim == 3

    def test_invalid_topology_too_short(self):
        """Should reject topology with fewer than 2 elements."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            build_network_from_topology([5])

        with pytest.raises(ValueError, match="at least 2 elements"):
            build_network_from_topology([])

    def test_invalid_topology_zero_dim(self):
        """Should reject topology with zero dimensions."""
        with pytest.raises(ValueError, match="at least 1"):
            build_network_from_topology([2, 0, 2])

    def test_accepts_tuple(self):
        """Should accept tuple as topology."""
        net = build_network_from_topology((2, 4, 2))
        assert net.num_layers == 2

    def test_can_fit_after_build(self):
        """Built network should be trainable."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 3))
        Y = rng.uniform(0, 1, size=(50, 2))

        net = build_network_from_topology([3, 4, 2])
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        assert net._is_fitted


class TestTopologyBuilderIntegration:
    """Integration tests for topology builders."""

    def test_build_default_equals_from_topology(self):
        """build_default_network should match equivalent from_topology."""
        # build_default_network(dim_in=3, dim_out=2, hidden_layers=2, hidden_width=4)
        # should be equivalent to build_network_from_topology([3, 4, 4, 2])
        seed = 42

        net1 = build_default_network(
            dim_in=3, dim_out=2, hidden_layers=2, hidden_width=4, seed=seed
        )
        net2 = build_network_from_topology([3, 4, 4, 2], seed=seed)

        # Same structure
        assert net1.num_layers == net2.num_layers

        for s1, s2 in zip(net1.specs, net2.specs):
            assert s1.input_dim == s2.input_dim
            assert s1.output_dim == s2.output_dim

        # Same initialization (same seed)
        for l1, l2 in zip(net1.layers, net2.layers):
            np.testing.assert_array_equal(l1.W, l2.W)

    def test_forward_reverse_consistency(self):
        """Built networks should support forward and reverse."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(30, 2))
        Y = rng.uniform(0, 1, size=(30, 3))

        net = build_default_network(dim_in=2, dim_out=3, hidden_layers=1)
        net.fit(X, Y, method="alternating", max_iter=3, verbose=False)

        # Forward
        Y_pred = net.forward(X)
        assert Y_pred.shape == (30, 3)

        # Reverse
        X_recon = net.reverse(Y_pred)
        assert X_recon.shape == (30, 2)
