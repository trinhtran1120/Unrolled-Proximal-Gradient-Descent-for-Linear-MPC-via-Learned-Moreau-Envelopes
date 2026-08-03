"""ICNN utilities trained with Moreau-envelope value and gradient supervision."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

DEFAULT_WIDTHS = (64, 64, 64)
INIT_SCALE = 0.05
LINEAR_INIT_SCALE = 0.01


def act_p(x: jnp.ndarray) -> jnp.ndarray:
    """Project ICNN weights to the nonnegative orthant with a smooth map."""
    return jax.nn.softplus(x)


def glorot_uniform(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    """Initialize a dense layer with Glorot-uniform weights."""
    fan_in, fan_out = shape[1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _validate_widths(widths: Sequence[int]) -> tuple[int, ...]:
    widths = tuple(int(width) for width in widths)
    if not widths:
        raise ValueError("widths must contain at least one hidden layer")
    if any(width <= 0 for width in widths):
        raise ValueError("all hidden-layer widths must be positive")
    return widths


def init_icnn_params(
    key: jax.Array,
    n_in: int,
    widths: Sequence[int],
) -> Params:
    """Initialize ICNN parameters."""
    if n_in <= 0:
        raise ValueError("n_in must be positive")

    widths = _validate_widths(widths)
    keys = jax.random.split(key, 3 * len(widths) + 3)
    key_idx = 0

    input_weights: list[jnp.ndarray] = []
    state_weights: list[jnp.ndarray] = []
    biases: list[jnp.ndarray] = []

    for layer_idx, width in enumerate(widths):
        input_weights.append(glorot_uniform(keys[key_idx], (width, n_in)))
        key_idx += 1

        if layer_idx == 0:
            state_weights.append(jnp.zeros((width, n_in)))
        else:
            shape = (width, widths[layer_idx - 1])
            state_weights.append(INIT_SCALE * jax.random.normal(keys[key_idx], shape))
            key_idx += 1

        biases.append(jnp.zeros((width,)))

    output_weights = INIT_SCALE * jax.random.normal(keys[-3], (widths[-1],))
    linear_weights = LINEAR_INIT_SCALE * jax.random.normal(keys[-2], (n_in,))
    bias = jnp.array(0.0)

    return {
        "U": input_weights,
        "W": state_weights,
        "b": biases,
        "v": output_weights,
        "a": linear_weights,
        "c": bias,
    }


@jax.jit
def icnn_forward(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the scalar ICNN output for one input vector."""
    input_weights, state_weights, biases = params["U"], params["W"], params["b"]
    activation = jax.nn.softplus

    z = activation(jnp.dot(input_weights[0], x) + biases[0])
    for layer_idx in range(1, len(input_weights)):
        state_term = jnp.dot(act_p(state_weights[layer_idx]), z)
        input_term = jnp.dot(input_weights[layer_idx], x)
        z = activation(state_term + input_term + biases[layer_idx])

    raw_output = jnp.dot(act_p(params["v"]), z) + jnp.dot(params["a"], x) + params["c"]
    return jax.nn.softplus(raw_output)


batched_forward = jax.vmap(icnn_forward, in_axes=(None, 0))


@jax.jit
def grad_wrt_x(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """Return the gradient of the scalar ICNN output with respect to one input."""
    return jax.grad(icnn_forward, argnums=1)(params, x)


batched_grad_wrt_x = jax.jit(jax.vmap(grad_wrt_x, in_axes=(None, 0)))


@jax.jit
def loss_fn(
    params: Params,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    gb: jnp.ndarray,
    grad_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Return total loss and its value/gradient MSE components."""
    y_pred = batched_forward(params, xb)
    g_pred = batched_grad_wrt_x(params, xb)[:, : gb.shape[1]]

    value_mse = jnp.mean((y_pred - yb) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - gb) ** 2, axis=1))
    return value_mse + grad_weight * grad_mse, (value_mse, grad_mse)


def batch_iterator(
    X: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[Batch]:
    """Yield shuffled mini-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    permutation = jax.random.permutation(shuffle_key, X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield X[indices], y[indices], g[indices]


def make_train_step(optimizer: optax.GradientTransformation):
    """Build one JIT-compiled optimizer step."""

    @jax.jit
    def train_step(
        params: Params,
        opt_state: optax.OptState,
        xb: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        grad_weight: float,
    ):
        (loss, (vmse, gmse)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xb, yb, gb, grad_weight
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, vmse, gmse

    return train_step


def _as_training_arrays(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_in: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Xj = jnp.asarray(X, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)

    if Xj.ndim != 2 or Xj.shape[1] != n_in:
        raise ValueError(f"X must have shape (N, {n_in})")
    if yj.shape != (Xj.shape[0],):
        raise ValueError("y must have shape (N,)")
    if gj.ndim != 2 or gj.shape[0] != Xj.shape[0] or gj.shape[1] > Xj.shape[1]:
        raise ValueError(
            "g must have shape (N, grad_dim), where grad_dim is no larger "
            f"than the input dimension {Xj.shape[1]}"
        )

    return Xj, yj, gj


def _copy_params(params: Params) -> Params:
    return jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        params,
    )


def train_icnn(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_in: int,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    lr: float = 1e-3,
    grad_weight: float = 1.0,
    l2_reg: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 0,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
) -> Params:
    """Train an ICNN with value and input-gradient supervision."""
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")

    key = jax.random.PRNGKey(seed)
    widths = _validate_widths(widths)
    Xj, yj, gj = _as_training_arrays(X, y, g, n_in)
    has_validation = X_val is not None or y_val is not None or g_val is not None
    if has_validation:
        if X_val is None or y_val is None or g_val is None:
            raise ValueError("X_val, y_val, and g_val must be provided together")
        Xvj, yvj, gvj = _as_training_arrays(X_val, y_val, g_val, n_in)

    params = init_icnn_params(key, n_in=n_in, widths=widths)
    best_params = params
    best_val = float("inf")

    optimizer = optax.adamw(learning_rate=lr, weight_decay=l2_reg)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for xb, yb, gb in batch_iterator(Xj, yj, gj, batch_size, batch_key):
            params, opt_state, _loss, _vmse, _gmse = train_step(
                params, opt_state, xb, yb, gb, grad_weight
            )

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            with jax.disable_jit():
                if has_validation:
                    eval_X, eval_y, eval_g = Xvj, yvj, gvj
                    metric_label = "val"
                else:
                    eval_X, eval_y, eval_g = Xj[:1024], yj[:1024], gj[:1024]
                    metric_label = "train"

                yp = batched_forward(params, eval_X)
                gp = batched_grad_wrt_x(params, eval_X)[:, : eval_g.shape[1]]
                vm = jnp.mean((yp - eval_y) ** 2)
                gm = jnp.mean(jnp.sum((gp - eval_g) ** 2, axis=1))
                selection_obj = vm + grad_weight * gm

                if float(selection_obj) < best_val:
                    best_val = float(selection_obj)
                    best_params = _copy_params(params)
                print(
                    f"Epoch {epoch:4d} | {metric_label} value MSE: {vm:.4e} "
                    f"| {metric_label} grad MSE: {gm:.4e}"
                )

    return best_params


def to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj
