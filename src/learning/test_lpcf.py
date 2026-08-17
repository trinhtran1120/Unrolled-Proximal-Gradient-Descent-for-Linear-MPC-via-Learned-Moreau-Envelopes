"""Parametric convex-function network for (fixed-gamma) Moreau envelopes.

The LPCF model represents f(q, theta), where q is the convex variable and
theta is a parameter. 
In the fixed-step-size MPC setting, theta is x0.
Meanwhile, in the adaptive step-size MPC setting, theta is (x0, rho)
"""

from __future__ import annotations
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

def act_p(x: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained weights to nonnegative weights."""
    return jax.nn.softplus(x)

def activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing activation used in the convex network."""
    return jax.nn.softplus(x)

def activation_derivative(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of softplus, used for manual q-sensitivities."""
    return jax.nn.sigmoid(x)

def glorot_uniform(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
        """Initialize a dense layer with Glorot-uniform weights."""
        fan_in, fan_out = shape[1], shape[0]
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def _validate_widths(widths: Sequence[int], name: str) -> tuple[int, ...]:
    widths = tuple(int(width) for width in widths)
    if not widths:
        raise ValueError(f"{name} must contain at least one hidden layer")
    if any(width <= 0 for width in widths):
        raise ValueError(f"all {name} entries must be positive")
    return widths

def _lpcf_shapes(
          
) -> tuple[tuple]