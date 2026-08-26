"""
Utility functions for fitting a parametric convex function to data.

M. Schaller, A. Bemporad, March 19, 2025
"""

from __future__ import annotations

from typing import Any, Iterator

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _batches(
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    projection: jnp.ndarray,
    weights: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[Batch]:
    permutation = jax.random.permutation(shuffle_key, model_input.shape[0])
    for start in range(0, model_input.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield model_input[indices], parameter[indices], projection[indices], weights[indices]


def _copy_params(params: Params) -> Params:
    """Copy a JAX pytree of model parameters."""
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, params)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


def _prediction_matrices(a_matrix: np.ndarray, b_matrix: np.ndarray, horizon: int):
    """Build stacked single-shooting prediction matrices."""
    nx, nu = b_matrix.shape
    a_ro = np.zeros((horizon * nx, nx))
    b_ro = np.zeros((horizon * nx, horizon * nu))

    for k in range(1, horizon + 1):
        row = slice((k - 1) * nx, k * nx)
        a_ro[row] = np.linalg.matrix_power(a_matrix, k)
        for j in range(1, k + 1):
            col = slice((j - 1) * nu, j * nu)
            b_ro[row, col] = np.linalg.matrix_power(a_matrix, k - j) @ b_matrix

    return a_ro, b_ro
