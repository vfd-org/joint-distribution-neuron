"""
Pipeline for composing multiple HCRNetwork stages.

This module provides a Pipeline class that chains multiple HCRNetwork
instances together, supporting sequential forward/reverse inference
and optional uncertainty propagation.

Usage:
    from hcrnn import HCRNetwork, LayerSpec, Pipeline

    # Create individual networks
    encoder = HCRNetwork([LayerSpec(input_dim=10, output_dim=4)])
    decoder = HCRNetwork([LayerSpec(input_dim=4, output_dim=10)])

    # Compose into pipeline
    pipeline = Pipeline([encoder, decoder])
    pipeline.fit(X, Y)

    # Forward through all stages
    output = pipeline.forward(X)

    # Reverse through all stages
    reconstructed = pipeline.reverse(output)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from hcrnn.network import HCRNetwork


class Pipeline:
    """
    A pipeline that composes multiple HCRNetwork stages sequentially.

    Each stage is an HCRNetwork. Forward passes the data through each
    network in order; reverse passes through in reverse order.

    The pipeline supports:
    - Sequential forward inference through all stages
    - Sequential reverse inference through all stages (reversed order)
    - Optional uncertainty propagation via variance computation
    - Individual stage access for fine-grained control

    Example:
        >>> from hcrnn import HCRNetwork, LayerSpec, Pipeline
        >>>
        >>> # Two-stage pipeline: compress then expand
        >>> compress = HCRNetwork([LayerSpec(input_dim=10, output_dim=4)])
        >>> expand = HCRNetwork([LayerSpec(input_dim=4, output_dim=10)])
        >>>
        >>> pipe = Pipeline([compress, expand])
        >>> pipe.fit_stages([(X, H), (H, Y)])  # Train each stage
        >>>
        >>> Y_pred = pipe.forward(X)
        >>> X_recon = pipe.reverse(Y_pred)
    """

    def __init__(self, stages: Sequence[HCRNetwork]):
        """
        Initialize pipeline with a sequence of HCRNetwork stages.

        Args:
            stages: Sequence of HCRNetwork instances to chain together

        Raises:
            ValueError: If stages is empty
            TypeError: If any stage is not an HCRNetwork
        """
        if len(stages) == 0:
            raise ValueError("Pipeline requires at least one stage")

        for i, stage in enumerate(stages):
            if not isinstance(stage, HCRNetwork):
                raise TypeError(
                    f"Stage {i} must be HCRNetwork, got {type(stage).__name__}"
                )

        self._stages: List[HCRNetwork] = list(stages)

    @property
    def stages(self) -> List[HCRNetwork]:
        """Return the list of pipeline stages."""
        return self._stages

    @property
    def num_stages(self) -> int:
        """Return the number of stages in the pipeline."""
        return len(self._stages)

    @property
    def is_fitted(self) -> bool:
        """Return True if all stages are fitted."""
        return all(stage._is_fitted for stage in self._stages)

    def forward(
        self,
        X: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Forward pass through all pipeline stages.

        Args:
            X: Input data of shape (N, input_dim)
            return_intermediates: If True, also return outputs from each stage

        Returns:
            If return_intermediates is False:
                Final output array of shape (N, output_dim)
            If return_intermediates is True:
                Tuple of (final_output, list_of_intermediate_outputs)
        """
        X = np.asarray(X)
        intermediates = []
        current = X

        for stage in self._stages:
            current = stage.forward(current)
            intermediates.append(current)

        if return_intermediates:
            return current, intermediates
        return current

    def reverse(
        self,
        Y: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Reverse pass through all pipeline stages (in reverse order).

        Args:
            Y: Output-space data of shape (N, output_dim)
            return_intermediates: If True, also return outputs from each stage

        Returns:
            If return_intermediates is False:
                Reconstructed input array of shape (N, input_dim)
            If return_intermediates is True:
                Tuple of (reconstructed, list_of_intermediate_outputs)
        """
        Y = np.asarray(Y)
        intermediates = []
        current = Y

        # Reverse order
        for stage in reversed(self._stages):
            current = stage.reverse(current)
            intermediates.append(current)

        if return_intermediates:
            return current, intermediates
        return current

    def forward_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass with uncertainty (variance) at each stage.

        Computes the output and the variance from each stage's
        final layer joint density.

        Args:
            X: Input data of shape (N, input_dim)

        Returns:
            Tuple of:
                - Final output array of shape (N, output_dim)
                - List of variance arrays, one per stage
        """
        X = np.asarray(X)
        current = X
        variances = []

        for stage in self._stages:
            # Compute uncertainty at the last layer of this stage
            var = stage.compute_uncertainty(current, layer_idx=-1)
            variances.append(var)

            # Forward through stage
            current = stage.forward(current)

        return current, variances

    def fit_stages(
        self,
        stage_data: Sequence[Tuple[np.ndarray, np.ndarray]],
        method: str = "alternating",
        max_iter: int = 10,
        verbose: bool = True,
    ) -> "Pipeline":
        """
        Fit each stage with its corresponding (X, Y) data pair.

        This method allows training each stage independently with
        explicit intermediate targets.

        Args:
            stage_data: Sequence of (X, Y) tuples, one per stage.
                        len(stage_data) must equal num_stages
            method: Training method ("alternating", "cmaes", "coordinate")
            max_iter: Maximum iterations per stage
            verbose: Print progress

        Returns:
            self, for method chaining

        Raises:
            ValueError: If len(stage_data) != num_stages
        """
        if len(stage_data) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} (X, Y) pairs, got {len(stage_data)}"
            )

        for i, (stage, (X, Y)) in enumerate(zip(self._stages, stage_data)):
            if verbose:
                print(f"Training stage {i + 1}/{self.num_stages}")
            stage.fit(X, Y, method=method, max_iter=max_iter, verbose=verbose)

        return self

    def fit_end_to_end(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        method: str = "alternating",
        max_iter: int = 10,
        verbose: bool = True,
    ) -> "Pipeline":
        """
        Fit pipeline end-to-end by propagating through stages.

        For multi-stage pipelines, this uses the forward pass of
        earlier (already-fitted) stages to generate intermediate
        targets for later stages. First stage is trained on (X, ?),
        where ? is an interpolation toward Y.

        Note: For complex pipelines, explicit fit_stages() with
        known intermediate representations may work better.

        Args:
            X: Input data of shape (N, input_dim)
            Y: Target output of shape (N, output_dim)
            method: Training method
            max_iter: Maximum iterations per stage
            verbose: Print progress

        Returns:
            self, for method chaining
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        N = X.shape[0]

        current_input = X

        for i, stage in enumerate(self._stages):
            if verbose:
                print(f"Training stage {i + 1}/{self.num_stages}")

            # Determine target for this stage
            if i == self.num_stages - 1:
                # Last stage: target is final Y
                target = Y
            else:
                # Intermediate stage: create interpolated target
                # based on stage output dimensions
                output_dim = stage.specs[-1].output_dim
                alpha = (i + 1) / self.num_stages

                # Try to interpolate if dimensions allow
                if current_input.shape[1] >= output_dim and Y.shape[1] >= output_dim:
                    target = (
                        (1 - alpha) * current_input[:, :output_dim]
                        + alpha * Y[:, :output_dim]
                    )
                else:
                    # Fall back to random target in [0, 1]
                    rng = np.random.default_rng(42 + i)
                    target = rng.uniform(0.1, 0.9, size=(N, output_dim))

            stage.fit(
                current_input, target,
                method=method, max_iter=max_iter, verbose=verbose
            )

            # Propagate through this stage for next stage's input
            current_input = stage.forward(current_input)

        return self

    def reconstruction_error(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute reconstruction errors for the pipeline.

        Args:
            X: Input data
            Y: Optional target output. If None, only round-trip error is computed.

        Returns:
            Dictionary with error metrics:
                - roundtrip_rmse: RMSE of X → forward → reverse → X
                - forward_rmse: RMSE of forward(X) vs Y (if Y provided)
        """
        X = np.asarray(X)

        # Forward then reverse
        Y_pred = self.forward(X)
        X_recon = self.reverse(Y_pred)

        roundtrip_rmse = np.sqrt(np.mean((X - X_recon) ** 2))

        result = {"roundtrip_rmse": roundtrip_rmse}

        if Y is not None:
            Y = np.asarray(Y)
            forward_rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
            result["forward_rmse"] = forward_rmse

        return result

    def __repr__(self) -> str:
        stage_strs = [repr(s) for s in self._stages]
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"Pipeline({self.num_stages} stages, {fitted})"

    def __len__(self) -> int:
        return self.num_stages

    def __getitem__(self, idx: int) -> HCRNetwork:
        return self._stages[idx]
