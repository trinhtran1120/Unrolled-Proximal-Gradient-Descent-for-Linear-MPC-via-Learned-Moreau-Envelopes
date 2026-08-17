"""Small utility layer for the MPC-specific PCF trainer."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

params_t = dict[str, Any]
batch_t = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

default_convex_widths = (32, 32)
default_hyper_widths = (64, 64)
init_scale = 0.05

def act_p(x: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained convex-network weights to nonnegative values."""
    return jax.nn.softplus(x)


def activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing activation used by the variable network."""
    return jax.nn.softplus(x)


def glorot_uniform(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    """Initialize a dense layer with Glorot-uniform weights."""
    fan_in, fan_out = shape[1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def validate_widths(widths: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate hidden-layer widths."""
    widths = tuple(int(width) for width in widths)
    if not widths:
        raise ValueError(f"{name} must contain at least one hidden layer")
    if any(width <= 0 for width in widths):
        raise ValueError(f"all {name} entries must be positive")
    return widths


def copy_params(params: params_t) -> params_t:
    """Copy a JAX pytree of model parameters."""
    return jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        params,
    )


def to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj


def feasibility_weight(feasibility_data: params_t | None) -> jnp.ndarray:
    """Return the soft feasibility penalty weight."""
    if feasibility_data is None:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.asarray(feasibility_data.get("weight", 0.0), dtype=jnp.float32)


def gradient_loss_scale(
    feasibility_data: params_t | None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return the scale mapping normalized q-gradients back to original units."""
    if feasibility_data is None or "gradient_scale" not in feasibility_data:
        return jnp.asarray(1.0, dtype=dtype)
    return jnp.asarray(feasibility_data["gradient_scale"], dtype=dtype)


def soft_feasibility_penalty(
    qb: jnp.ndarray,
    thetab: jnp.ndarray,
    g_pred: jnp.ndarray,
    alpha: jnp.ndarray,
    feasibility_data: params_t | None,
) -> jnp.ndarray:
    """Penalize relu(g_matrix u_hat - b(theta)) in original MPC units."""
    if feasibility_data is None:
        return jnp.array(0.0, dtype=qb.dtype)

    q_mean = jnp.asarray(feasibility_data["q_mean"], dtype=qb.dtype)
    q_std = jnp.asarray(feasibility_data["q_std"], dtype=qb.dtype)
    theta_mean = jnp.asarray(feasibility_data["theta_mean"], dtype=thetab.dtype)
    theta_std = jnp.asarray(feasibility_data["theta_std"], dtype=thetab.dtype)
    env_scale = jnp.asarray(feasibility_data["env_scale"], dtype=qb.dtype)
    g_matrix = jnp.asarray(feasibility_data["g_matrix"], dtype=qb.dtype)
    b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=qb.dtype)
    b_theta = jnp.asarray(feasibility_data["b_theta"], dtype=qb.dtype)

    q_original = qb * q_std + q_mean
    theta_original = thetab * theta_std + theta_mean
    gamma_feature = feasibility_data.get("gamma_feature", "none")
    if gamma_feature == "gamma":
        x0_dim = int(feasibility_data["x0_dim"])
        x0_original = theta_original[:, :x0_dim]
        gamma = theta_original[:, x0_dim : x0_dim + 1]
    elif gamma_feature == "log_gamma":
        x0_dim = int(feasibility_data["x0_dim"])
        x0_original = theta_original[:, :x0_dim]
        gamma = jnp.exp(theta_original[:, x0_dim : x0_dim + 1])
    else:
        x0_original = theta_original
        gamma = jnp.asarray(feasibility_data["gamma"], dtype=qb.dtype)

    g_original = g_pred * env_scale / q_std
    u_hat = q_original - gamma * g_original
    b_value = b_offset[None, :] + jnp.matmul(x0_original, b_theta.T)
    violation = jax.nn.relu(jnp.matmul(u_hat, g_matrix.T) - b_value)
    return jnp.mean(alpha * jnp.sum(violation**2, axis=1))


def batch_iterator(
    q: jnp.ndarray,
    theta: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[batch_t]:
    """Yield shuffled mini-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    permutation = jax.random.permutation(shuffle_key, q.shape[0])
    for start in range(0, q.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield q[indices], theta[indices], y[indices], g[indices], w[indices]


def as_training_arrays(
    q: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    q_dim: int,
    theta_dim: int,
    w: np.ndarray,
) -> batch_t:
    """Convert and validate training arrays."""
    qj = jnp.asarray(q, dtype=jnp.float32)
    thetaj = jnp.asarray(theta, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)
    wj = jnp.asarray(w, dtype=jnp.float32)

    if qj.ndim != 2 or qj.shape[1] != q_dim:
        raise ValueError(f"q must have shape (n, {q_dim})")
    if thetaj.ndim != 2 or thetaj.shape != (qj.shape[0], theta_dim):
        raise ValueError(f"theta must have shape (n, {theta_dim})")
    if yj.shape != (qj.shape[0],):
        raise ValueError("y must have shape (n,)")
    if gj.shape != qj.shape:
        raise ValueError(f"g must have shape (n, {q_dim})")
    if wj.shape != (qj.shape[0],):
        raise ValueError("w must have shape (n,)")
    if np.any(np.asarray(w) <= 0.0):
        raise ValueError("all w entries must be strictly positive")
    return qj, thetaj, yj, gj, wj
