"""
Utility functions for fitting a parametric convex function to data.

M. Schaller, A. Bemporad, March 19, 2025
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _make_positive(x: jnp.ndarray) -> jnp.ndarray:
    """Map recurrent ICNN weights to the nonnegative orthant."""
    return jax.nn.softplus(x)


@dataclass(frozen=True)
class Section:
    start: int = 0
    end: int = 0
    shape: tuple = (0, 0)

    def __iter__(self):
        yield self.start
        yield self.end
        yield self.shape

    def __getitem__(self, index: int):
        return (self.start, self.end, self.shape)[index]

    @property
    def size(self) -> int:
        return self.end - self.start


def _rand_uniform(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    """Initialize dense parameters with fan-in scaling."""
    fan_in = shape[-1] if len(shape) > 1 else 1
    return jax.random.uniform(key, shape, minval=-0.5, maxval=0.5, dtype=jnp.float64) / np.sqrt(fan_in)


def _make_sections(input_dim: int, widths: Sequence[int]) -> tuple[tuple[int, ...], Params, int]:
    """Plan flat psi-output slices for W, V, and omega tensors."""
    layer_dims = (int(input_dim), *tuple(int(width) for width in widths), 1)
    sections_w: list[Section] = []
    sections_v: list[Section] = []
    sections_o: list[Section] = []
    offset = 0

    for layer_idx in range(2, len(layer_dims)):
        offset = _append_section(
            sections_w,
            offset,
            (layer_dims[layer_idx], layer_dims[layer_idx - 1]),
        )
    for layer_idx in range(1, len(layer_dims)):
        offset = _append_section(sections_v, offset, (layer_dims[layer_idx], input_dim))
    for layer_idx in range(1, len(layer_dims)):
        offset = _append_section(sections_o, offset, (layer_dims[layer_idx],))

    sections = {"W": tuple(sections_w), "V": tuple(sections_v), "omega": tuple(sections_o)}
    return layer_dims, sections, offset


def _init_psi_params(
    key: jax.Array,
    parameter_dim: int,
    hyper_widths: Sequence[int],
    output_dim: int,
) -> Params:
    """Initialize the residual/feedforward psi network from the PCF source code."""
    dims = (int(parameter_dim), *tuple(int(width) for width in hyper_widths), int(output_dim))
    keys = jax.random.split(key, 3 * (len(dims) - 1))
    key_idx = 0
    W_psi: list[jnp.ndarray] = []
    V_psi: list[jnp.ndarray] = []
    omega_psi: list[jnp.ndarray] = []

    for layer_idx in range(1, len(dims)):
        if layer_idx > 1:
            W_psi.append(_rand_uniform(keys[key_idx], (dims[layer_idx], dims[layer_idx - 1])))
            key_idx += 1
        V_psi.append(_rand_uniform(keys[key_idx], (dims[layer_idx], parameter_dim)))
        key_idx += 1
        omega_psi.append(_rand_uniform(keys[key_idx], (dims[layer_idx],)))
        key_idx += 1

    return {"W_psi": W_psi, "V_psi": V_psi, "omega_psi": omega_psi}


def _copy_params(params: Params) -> Params:
    """Copy a JAX pytree of model parameters."""
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, params)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


def _append_section(section, offset, shape, size=None):
    """Add section in psi output p, such that p[start:end].reshape(shape)
    is a matrix or vector in main network; returns offset for next section"""
    
    if size is None:
        size = np.prod(shape)
    section.append(Section(start=offset, end=offset + int(size), shape=shape))
    return offset + int(size)


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
