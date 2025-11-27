"""Tests for Pipeline class."""

from __future__ import annotations

import numpy as np
import pytest

from hcrnn import (
    HCRNetwork,
    LayerSpec,
    Pipeline,
    build_default_network,
)


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_init_with_single_stage(self):
        """Should accept single-stage pipeline."""
        net = build_default_network(dim_in=2, dim_out=2)
        pipe = Pipeline([net])

        assert pipe.num_stages == 1

    def test_init_with_multiple_stages(self):
        """Should accept multi-stage pipeline."""
        net1 = build_default_network(dim_in=2, dim_out=4)
        net2 = build_default_network(dim_in=4, dim_out=2)
        pipe = Pipeline([net1, net2])

        assert pipe.num_stages == 2

    def test_init_empty_raises(self):
        """Should raise ValueError for empty stages."""
        with pytest.raises(ValueError, match="at least one stage"):
            Pipeline([])

    def test_init_wrong_type_raises(self):
        """Should raise TypeError for non-HCRNetwork stages."""
        with pytest.raises(TypeError, match="must be HCRNetwork"):
            Pipeline(["not a network"])

        with pytest.raises(TypeError, match="must be HCRNetwork"):
            net = build_default_network(dim_in=2, dim_out=2)
            Pipeline([net, "invalid"])

    def test_stages_property(self):
        """stages property should return list of networks."""
        net1 = build_default_network(dim_in=2, dim_out=4)
        net2 = build_default_network(dim_in=4, dim_out=2)
        pipe = Pipeline([net1, net2])

        assert pipe.stages[0] is net1
        assert pipe.stages[1] is net2

    def test_is_fitted_unfitted(self):
        """is_fitted should be False when stages are unfitted."""
        net = build_default_network(dim_in=2, dim_out=2)
        pipe = Pipeline([net])

        assert not pipe.is_fitted

    def test_is_fitted_fitted(self):
        """is_fitted should be True when all stages are fitted."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(30, 2))
        Y = rng.uniform(0, 1, size=(30, 2))

        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        net.fit(X, Y, method="alternating", max_iter=2, verbose=False)

        pipe = Pipeline([net])
        assert pipe.is_fitted


class TestPipelineForward:
    """Tests for Pipeline.forward method."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted two-stage pipeline."""
        rng = np.random.default_rng(42)

        # Stage 1: 2 → 4
        X1 = rng.uniform(0, 1, size=(50, 2))
        Y1 = rng.uniform(0, 1, size=(50, 4))
        net1 = build_default_network(dim_in=2, dim_out=4, hidden_layers=0)
        net1.fit(X1, Y1, method="alternating", max_iter=2, verbose=False)

        # Stage 2: 4 → 2
        X2 = rng.uniform(0, 1, size=(50, 4))
        Y2 = rng.uniform(0, 1, size=(50, 2))
        net2 = build_default_network(dim_in=4, dim_out=2, hidden_layers=0)
        net2.fit(X2, Y2, method="alternating", max_iter=2, verbose=False)

        return Pipeline([net1, net2])

    def test_forward_shape(self, fitted_pipeline):
        """Forward should return correct shape."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        Y = fitted_pipeline.forward(X)

        assert Y.shape == (10, 2)

    def test_forward_with_intermediates(self, fitted_pipeline):
        """Forward with return_intermediates should return intermediates."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        Y, intermediates = fitted_pipeline.forward(X, return_intermediates=True)

        assert Y.shape == (10, 2)
        assert len(intermediates) == 2
        assert intermediates[0].shape == (10, 4)  # After stage 1
        assert intermediates[1].shape == (10, 2)  # After stage 2

    def test_forward_propagates_through_stages(self, fitted_pipeline):
        """Forward should pass data through each stage."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        # Manual forward
        h1 = fitted_pipeline.stages[0].forward(X)
        h2 = fitted_pipeline.stages[1].forward(h1)

        # Pipeline forward
        Y = fitted_pipeline.forward(X)

        np.testing.assert_allclose(Y, h2, rtol=1e-10)


class TestPipelineReverse:
    """Tests for Pipeline.reverse method."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted two-stage pipeline."""
        rng = np.random.default_rng(42)

        net1 = build_default_network(dim_in=2, dim_out=4, hidden_layers=0)
        net1.fit(
            rng.uniform(0, 1, size=(50, 2)),
            rng.uniform(0, 1, size=(50, 4)),
            method="alternating", max_iter=2, verbose=False
        )

        net2 = build_default_network(dim_in=4, dim_out=2, hidden_layers=0)
        net2.fit(
            rng.uniform(0, 1, size=(50, 4)),
            rng.uniform(0, 1, size=(50, 2)),
            method="alternating", max_iter=2, verbose=False
        )

        return Pipeline([net1, net2])

    def test_reverse_shape(self, fitted_pipeline):
        """Reverse should return correct shape."""
        rng = np.random.default_rng(123)
        Y = rng.uniform(0, 1, size=(10, 2))

        X = fitted_pipeline.reverse(Y)

        assert X.shape == (10, 2)

    def test_reverse_with_intermediates(self, fitted_pipeline):
        """Reverse with return_intermediates should return intermediates."""
        rng = np.random.default_rng(123)
        Y = rng.uniform(0, 1, size=(10, 2))

        X, intermediates = fitted_pipeline.reverse(Y, return_intermediates=True)

        assert X.shape == (10, 2)
        assert len(intermediates) == 2
        # First intermediate: reverse of stage 2 (2 → 4)
        assert intermediates[0].shape == (10, 4)
        # Second intermediate: reverse of stage 1 (4 → 2)
        assert intermediates[1].shape == (10, 2)

    def test_reverse_propagates_in_order(self, fitted_pipeline):
        """Reverse should pass data through stages in reverse order."""
        rng = np.random.default_rng(123)
        Y = rng.uniform(0, 1, size=(10, 2))

        # Manual reverse (stage2 then stage1)
        h1 = fitted_pipeline.stages[1].reverse(Y)
        h0 = fitted_pipeline.stages[0].reverse(h1)

        # Pipeline reverse
        X = fitted_pipeline.reverse(Y)

        np.testing.assert_allclose(X, h0, rtol=1e-10)


class TestPipelineUncertainty:
    """Tests for Pipeline.forward_with_uncertainty method."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for uncertainty tests."""
        rng = np.random.default_rng(42)

        net1 = build_default_network(dim_in=2, dim_out=3, hidden_layers=0)
        net1.fit(
            rng.uniform(0, 1, size=(100, 2)),
            rng.uniform(0, 1, size=(100, 3)),
            method="alternating", max_iter=3, verbose=False
        )

        net2 = build_default_network(dim_in=3, dim_out=2, hidden_layers=0)
        net2.fit(
            rng.uniform(0, 1, size=(100, 3)),
            rng.uniform(0, 1, size=(100, 2)),
            method="alternating", max_iter=3, verbose=False
        )

        return Pipeline([net1, net2])

    def test_forward_with_uncertainty_returns_tuple(self, fitted_pipeline):
        """forward_with_uncertainty should return (output, variances)."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        result = fitted_pipeline.forward_with_uncertainty(X)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_with_uncertainty_output_shape(self, fitted_pipeline):
        """Output should have correct shape."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        output, variances = fitted_pipeline.forward_with_uncertainty(X)

        assert output.shape == (10, 2)

    def test_forward_with_uncertainty_variance_list(self, fitted_pipeline):
        """Variances should be list with one per stage."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        output, variances = fitted_pipeline.forward_with_uncertainty(X)

        assert len(variances) == 2

    def test_forward_with_uncertainty_variance_shapes(self, fitted_pipeline):
        """Variance arrays should have correct shapes."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, size=(10, 2))

        output, variances = fitted_pipeline.forward_with_uncertainty(X)

        # Variance at each stage corresponds to output dimension at that stage
        # Stage 1: 2 → 3, variance has shape (N, 3)
        assert variances[0].shape == (10, 3)
        # Stage 2: 3 → 2, variance has shape (N, 2)
        assert variances[1].shape == (10, 2)


class TestPipelineFitStages:
    """Tests for Pipeline.fit_stages method."""

    def test_fit_stages_trains_all(self):
        """fit_stages should train all stages."""
        rng = np.random.default_rng(42)

        net1 = build_default_network(dim_in=2, dim_out=3, hidden_layers=0)
        net2 = build_default_network(dim_in=3, dim_out=2, hidden_layers=0)
        pipe = Pipeline([net1, net2])

        data = [
            (rng.uniform(0, 1, (30, 2)), rng.uniform(0, 1, (30, 3))),
            (rng.uniform(0, 1, (30, 3)), rng.uniform(0, 1, (30, 2))),
        ]

        pipe.fit_stages(data, method="alternating", max_iter=2, verbose=False)

        assert pipe.is_fitted

    def test_fit_stages_wrong_count_raises(self):
        """fit_stages should raise if data count != stage count."""
        net1 = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        pipe = Pipeline([net1])

        rng = np.random.default_rng(42)
        data = [
            (rng.uniform(0, 1, (10, 2)), rng.uniform(0, 1, (10, 2))),
            (rng.uniform(0, 1, (10, 2)), rng.uniform(0, 1, (10, 2))),
        ]

        with pytest.raises(ValueError, match="Expected 1"):
            pipe.fit_stages(data, verbose=False)

    def test_fit_stages_returns_self(self):
        """fit_stages should return self for chaining."""
        rng = np.random.default_rng(42)

        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        pipe = Pipeline([net])

        data = [(rng.uniform(0, 1, (30, 2)), rng.uniform(0, 1, (30, 2)))]
        result = pipe.fit_stages(data, max_iter=1, verbose=False)

        assert result is pipe


class TestPipelineFitEndToEnd:
    """Tests for Pipeline.fit_end_to_end method."""

    def test_fit_end_to_end_trains_all(self):
        """fit_end_to_end should train all stages."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 2))
        Y = rng.uniform(0, 1, size=(50, 2))

        net1 = build_default_network(dim_in=2, dim_out=4, hidden_layers=0)
        net2 = build_default_network(dim_in=4, dim_out=2, hidden_layers=0)
        pipe = Pipeline([net1, net2])

        pipe.fit_end_to_end(X, Y, method="alternating", max_iter=2, verbose=False)

        assert pipe.is_fitted

    def test_fit_end_to_end_returns_self(self):
        """fit_end_to_end should return self for chaining."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(30, 2))
        Y = rng.uniform(0, 1, size=(30, 2))

        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        pipe = Pipeline([net])

        result = pipe.fit_end_to_end(X, Y, max_iter=1, verbose=False)

        assert result is pipe


class TestPipelineReconstructionError:
    """Tests for Pipeline.reconstruction_error method."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline."""
        rng = np.random.default_rng(42)

        net1 = build_default_network(dim_in=2, dim_out=3, hidden_layers=0)
        net1.fit(
            rng.uniform(0, 1, (50, 2)),
            rng.uniform(0, 1, (50, 3)),
            method="alternating", max_iter=2, verbose=False
        )

        net2 = build_default_network(dim_in=3, dim_out=2, hidden_layers=0)
        net2.fit(
            rng.uniform(0, 1, (50, 3)),
            rng.uniform(0, 1, (50, 2)),
            method="alternating", max_iter=2, verbose=False
        )

        return Pipeline([net1, net2])

    def test_reconstruction_error_returns_dict(self, fitted_pipeline):
        """reconstruction_error should return dict."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, (20, 2))

        result = fitted_pipeline.reconstruction_error(X)

        assert isinstance(result, dict)

    def test_reconstruction_error_has_roundtrip(self, fitted_pipeline):
        """reconstruction_error should include roundtrip_rmse."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, (20, 2))

        result = fitted_pipeline.reconstruction_error(X)

        assert "roundtrip_rmse" in result
        assert result["roundtrip_rmse"] >= 0

    def test_reconstruction_error_with_y(self, fitted_pipeline):
        """reconstruction_error with Y should include forward_rmse."""
        rng = np.random.default_rng(123)
        X = rng.uniform(0, 1, (20, 2))
        Y = rng.uniform(0, 1, (20, 2))

        result = fitted_pipeline.reconstruction_error(X, Y)

        assert "roundtrip_rmse" in result
        assert "forward_rmse" in result


class TestPipelineDunderMethods:
    """Tests for Pipeline special methods."""

    def test_len(self):
        """__len__ should return number of stages."""
        net1 = build_default_network(dim_in=2, dim_out=2)
        net2 = build_default_network(dim_in=2, dim_out=2)
        pipe = Pipeline([net1, net2])

        assert len(pipe) == 2

    def test_getitem(self):
        """__getitem__ should return stage by index."""
        net1 = build_default_network(dim_in=2, dim_out=2)
        net2 = build_default_network(dim_in=2, dim_out=2)
        pipe = Pipeline([net1, net2])

        assert pipe[0] is net1
        assert pipe[1] is net2

    def test_repr(self):
        """__repr__ should return informative string."""
        net = build_default_network(dim_in=2, dim_out=2)
        pipe = Pipeline([net])

        repr_str = repr(pipe)

        assert "Pipeline" in repr_str
        assert "1 stages" in repr_str
        assert "not fitted" in repr_str


class TestPipelineIntegration:
    """Integration tests for Pipeline."""

    def test_autoencoder_pipeline(self):
        """Pipeline should work as autoencoder."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0.2, 0.8, size=(100, 4))

        # Encoder: 4 → 2
        encoder = build_default_network(dim_in=4, dim_out=2, hidden_layers=0)
        # Decoder: 2 → 4
        decoder = build_default_network(dim_in=2, dim_out=4, hidden_layers=0)

        pipe = Pipeline([encoder, decoder])

        # Train encoder to compress
        H = rng.uniform(0.2, 0.8, size=(100, 2))
        encoder.fit(X, H, method="alternating", max_iter=3, verbose=False)

        # Train decoder to reconstruct
        decoder.fit(H, X, method="alternating", max_iter=3, verbose=False)

        # Test round-trip
        X_test = rng.uniform(0.2, 0.8, size=(20, 4))
        X_recon = pipe.forward(X_test)

        # Should have same shape
        assert X_recon.shape == X_test.shape

    def test_pipeline_saves_state(self):
        """Pipeline stages maintain their trained state."""
        rng = np.random.default_rng(42)

        net = build_default_network(dim_in=2, dim_out=2, hidden_layers=0)
        net.fit(
            rng.uniform(0, 1, (50, 2)),
            rng.uniform(0, 1, (50, 2)),
            method="alternating", max_iter=2, verbose=False
        )

        # Store original weights
        original_W = net.layers[0].W.copy()

        # Create pipeline and use it
        pipe = Pipeline([net])
        X = rng.uniform(0, 1, (10, 2))
        _ = pipe.forward(X)

        # Weights should be unchanged
        np.testing.assert_array_equal(net.layers[0].W, original_W)
