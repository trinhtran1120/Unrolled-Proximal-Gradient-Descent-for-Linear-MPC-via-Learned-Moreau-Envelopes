"""Parametric convex-function network for fixed-gamma Moreau envelopes.

The LPCF model represents f(q, theta), where q is the convex variable and
theta is a parameter. In the fixed-step-size MPC setting, theta is x0.

The gradient with respect to q is computed by a manual forward sensitivity
recursion rather than by calling jax.grad.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

DEFAULT_CONVEX_WIDTHS = (32, 32)
DEFAULT_HYPER_WIDTHS = (64, 64)
INIT_SCALE = 0.05


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
    q_dim: int,
    convex_widths: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int]:
    """Return emitted tensor shapes, layer dimensions, and flat output size."""
    if q_dim <= 0:
        raise ValueError("q_dim must be positive")

    widths = _validate_widths(convex_widths, "convex_widths")
    layer_dims = (q_dim, *widths, 1)
    shapes: list[tuple[int, ...]] = []

    hidden_dims = layer_dims[1:-1]
    for layer_idx, hidden_dim in enumerate(hidden_dims):
        if layer_idx > 0:
            shapes.append((hidden_dim, hidden_dims[layer_idx - 1]))  # W_l
        shapes.append((hidden_dim, q_dim))  # V_l
        shapes.append((hidden_dim,))  # b_l

    shapes.append((1, hidden_dims[-1]))  # W_out
    shapes.append((1, q_dim))  # V_out
    shapes.append((1,))  # b_out

    emit_dim = int(sum(np.prod(shape) for shape in shapes))
    return tuple(shapes), layer_dims, emit_dim


def init_mlp_params(
    key: jax.Array,
    input_dim: int,
    hidden_widths: Sequence[int],
    output_dim: int,
) -> Params:
    """Initialize an ordinary MLP."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")

    hidden_widths = _validate_widths(hidden_widths, "hidden_widths")
    dims = (input_dim, *hidden_widths, output_dim)
    keys = jax.random.split(key, len(dims) - 1)

    weights: list[jnp.ndarray] = []
    biases: list[jnp.ndarray] = []
    for layer_idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        scale = 1.0 if layer_idx < len(dims) - 2 else INIT_SCALE
        weights.append(scale * glorot_uniform(keys[layer_idx], (fan_out, fan_in)))
        biases.append(jnp.zeros((fan_out,)))

    return {"A": weights, "b": biases}


def init_lpcf_params(
    key: jax.Array,
    q_dim: int,
    theta_dim: int,
    convex_widths: Sequence[int] = DEFAULT_CONVEX_WIDTHS,
    hyper_widths: Sequence[int] = DEFAULT_HYPER_WIDTHS,
) -> Params:
    """Initialize LPCF parameters for f(q, theta)."""
    if theta_dim <= 0:
        raise ValueError("theta_dim must be positive")

    convex_widths = _validate_widths(convex_widths, "convex_widths")
    hyper_widths = _validate_widths(hyper_widths, "hyper_widths")
    shapes, layer_dims, emit_dim = _lpcf_shapes(q_dim, convex_widths)

    return {
        "q_dim": int(q_dim),
        "theta_dim": int(theta_dim),
        "convex_widths": convex_widths,
        "hyper_widths": hyper_widths,
        "layer_dims": layer_dims,
        "shapes": shapes,
        "hyper": init_mlp_params(key, theta_dim, hyper_widths, emit_dim),
    }


def hyper_forward(hyper_params: Params, theta: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the hypernetwork psi(theta)."""
    h = theta
    weights, biases = hyper_params["A"], hyper_params["b"]
    for A, b in zip(weights[:-1], biases[:-1]):
        h = jax.nn.relu(jnp.dot(A, h) + b)
    return jnp.dot(weights[-1], h) + biases[-1]


def unpack_lpcf_weights(params: Params, emitted: jnp.ndarray) -> Params:
    """Unpack the hypernetwork output into LPCF matrices and vectors."""
    shapes = params["shapes"]
    convex_widths = params["convex_widths"]
    offset = 0

    hidden_W: list[jnp.ndarray] = []
    input_V: list[jnp.ndarray] = []
    biases: list[jnp.ndarray] = []

    def take(shape: tuple[int, ...]) -> jnp.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        value = emitted[offset : offset + size].reshape(shape)
        offset += size
        return value

    shape_idx = 0
    for layer_idx in range(len(convex_widths)):
        if layer_idx > 0:
            hidden_W.append(take(shapes[shape_idx]))
            shape_idx += 1
        input_V.append(take(shapes[shape_idx]))
        shape_idx += 1
        biases.append(take(shapes[shape_idx]))
        shape_idx += 1

    output_W = take(shapes[shape_idx])
    output_V = take(shapes[shape_idx + 1])
    output_b = take(shapes[shape_idx + 2])

    return {
        "W": hidden_W,
        "V": input_V,
        "b": biases,
        "W_out": output_W,
        "V_out": output_V,
        "b_out": output_b,
    }


def lpcf_forward(params: Params, q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Evaluate scalar f(q, theta), convex in q for each fixed theta."""
    emitted = hyper_forward(params["hyper"], theta)
    weights = unpack_lpcf_weights(params, emitted)

    z = activation(jnp.dot(weights["V"][0], q) + weights["b"][0])
    for layer_idx in range(1, len(weights["V"])):
        state_term = jnp.dot(act_p(weights["W"][layer_idx - 1]), z)
        input_term = jnp.dot(weights["V"][layer_idx], q)
        z = activation(state_term + input_term + weights["b"][layer_idx])

    raw_output = (
        jnp.dot(act_p(weights["W_out"][0]), z)
        + jnp.dot(weights["V_out"][0], q)
        + weights["b_out"][0]
    )
    return raw_output


def value_and_grad_wrt_q(
    params: Params,
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate f(q, theta) and manually propagate df/dq."""
    emitted = hyper_forward(params["hyper"], theta)
    weights = unpack_lpcf_weights(params, emitted)

    pre = jnp.dot(weights["V"][0], q) + weights["b"][0]
    z = activation(pre)
    dz_dq = activation_derivative(pre)[:, None] * weights["V"][0]

    for layer_idx in range(1, len(weights["V"])):
        W_pos = act_p(weights["W"][layer_idx - 1])
        pre = (
            jnp.dot(W_pos, z)
            + jnp.dot(weights["V"][layer_idx], q)
            + weights["b"][layer_idx]
        )
        dpre_dq = jnp.dot(W_pos, dz_dq) + weights["V"][layer_idx]
        z = activation(pre)
        dz_dq = activation_derivative(pre)[:, None] * dpre_dq

    W_out_pos = act_p(weights["W_out"][0])
    raw_output = (
        jnp.dot(W_out_pos, z)
        + jnp.dot(weights["V_out"][0], q)
        + weights["b_out"][0]
    )
    grad_q = jnp.dot(W_out_pos, dz_dq) + weights["V_out"][0]
    return raw_output, grad_q


def grad_wrt_q(params: Params, q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Return manual gradient of f with respect to q."""
    _, grad_q = value_and_grad_wrt_q(params, q, theta)
    return grad_q


batched_forward = jax.vmap(lpcf_forward, in_axes=(None, 0, 0))
batched_value_and_grad_wrt_q = jax.vmap(value_and_grad_wrt_q, in_axes=(None, 0, 0))
batched_grad_wrt_q = jax.vmap(grad_wrt_q, in_axes=(None, 0, 0))


def loss_fn(
    params: Params,
    qb: jnp.ndarray,
    thetab: jnp.ndarray,
    yb: jnp.ndarray,
    gb: jnp.ndarray,
    wb: jnp.ndarray,
    grad_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Return value-plus-manual-gradient matching loss."""
    y_pred, g_pred = batched_value_and_grad_wrt_q(params, qb, thetab)

    weight_scale = jnp.maximum(jnp.mean(wb), 1e-12)
    normalized_weights = wb / weight_scale
    value_mse = jnp.mean(normalized_weights * (y_pred - yb) ** 2)
    grad_mse = jnp.mean(normalized_weights * jnp.sum((g_pred - gb) ** 2, axis=1))
    return value_mse + grad_weight * grad_mse, (value_mse, grad_mse)


def batch_iterator(
    Q: jnp.ndarray,
    Theta: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[Batch]:
    """Yield shuffled mini-batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    permutation = jax.random.permutation(shuffle_key, Q.shape[0])
    for start in range(0, Q.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield Q[indices], Theta[indices], y[indices], g[indices], w[indices]


def make_train_step(optimizer: optax.GradientTransformation):
    """Build one optimizer step.

    The model gradient with respect to q is manual. Optimization of network
    parameters still uses JAX autodiff through the scalar training loss.
    """

    def train_step(
        params: Params,
        opt_state: optax.OptState,
        qb: jnp.ndarray,
        thetab: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        wb: jnp.ndarray,
        grad_weight: float,
    ):
        def hyper_loss(hyper_params: Params):
            trainable_params = dict(params)
            trainable_params["hyper"] = hyper_params
            return loss_fn(trainable_params, qb, thetab, yb, gb, wb, grad_weight)

        (loss, (vmse, gmse)), grads = jax.value_and_grad(hyper_loss, has_aux=True)(
            params["hyper"]
        )
        updates, opt_state = optimizer.update(grads, opt_state, params["hyper"])
        params = dict(params)
        params["hyper"] = optax.apply_updates(params["hyper"], updates)
        return params, opt_state, loss, vmse, gmse

    return train_step


def _as_training_arrays(
    Q: np.ndarray,
    Theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    q_dim: int,
    theta_dim: int,
    w: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Qj = jnp.asarray(Q, dtype=jnp.float32)
    Thetaj = jnp.asarray(Theta, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)
    wj = jnp.asarray(w, dtype=jnp.float32)

    if Qj.ndim != 2 or Qj.shape[1] != q_dim:
        raise ValueError(f"Q must have shape (N, {q_dim})")
    if Thetaj.ndim != 2 or Thetaj.shape != (Qj.shape[0], theta_dim):
        raise ValueError(f"Theta must have shape (N, {theta_dim})")
    if yj.shape != (Qj.shape[0],):
        raise ValueError("y must have shape (N,)")
    if gj.shape != Qj.shape:
        raise ValueError(f"g must have shape (N, {q_dim})")
    if wj.shape != (Qj.shape[0],):
        raise ValueError("w must have shape (N,)")
    if np.any(np.asarray(w) <= 0.0):
        raise ValueError("all sample weights must be strictly positive")

    return Qj, Thetaj, yj, gj, wj


def _copy_params(params: Params) -> Params:
    return jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        params,
    )


def train_lpcf(
    Q: np.ndarray,
    Theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    q_dim: int,
    theta_dim: int,
    w: np.ndarray | None = None,
    convex_widths: Sequence[int] = DEFAULT_CONVEX_WIDTHS,
    hyper_widths: Sequence[int] = DEFAULT_HYPER_WIDTHS,
    lr: float = 1e-3,
    grad_weight: float = 1.0,
    l2_reg: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 0,
    Q_val: np.ndarray | None = None,
    Theta_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
) -> Params:
    """Train an LPCF with value and manual q-gradient supervision."""
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if w is None:
        w = np.ones(Q.shape[0])

    key = jax.random.PRNGKey(seed)
    Qj, Thetaj, yj, gj, wj = _as_training_arrays(
        Q, Theta, y, g, q_dim, theta_dim, w
    )

    has_validation = (
        Q_val is not None
        or Theta_val is not None
        or y_val is not None
        or g_val is not None
        or w_val is not None
    )
    if has_validation:
        if Q_val is None or Theta_val is None or y_val is None or g_val is None:
            raise ValueError("Q_val, Theta_val, y_val, and g_val must be provided together")
        if w_val is None:
            w_val = np.ones(Q_val.shape[0])
        Qvj, Thetavj, yvj, gvj, wvj = _as_training_arrays(
            Q_val, Theta_val, y_val, g_val, q_dim, theta_dim, w_val
        )

    params = init_lpcf_params(
        key,
        q_dim=q_dim,
        theta_dim=theta_dim,
        convex_widths=convex_widths,
        hyper_widths=hyper_widths,
    )
    best_params = params
    best_val = float("inf")

    optimizer = optax.adamw(learning_rate=lr, weight_decay=l2_reg)
    opt_state = optimizer.init(params["hyper"])
    train_step = make_train_step(optimizer)

    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for qb, thetab, yb, gb, wb in batch_iterator(
            Qj, Thetaj, yj, gj, wj, batch_size, batch_key
        ):
            params, opt_state, _loss, _vmse, _gmse = train_step(
                params, opt_state, qb, thetab, yb, gb, wb, grad_weight
            )

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            with jax.disable_jit():
                if has_validation:
                    eval_Q, eval_Theta, eval_y, eval_g, eval_w = (
                        Qvj,
                        Thetavj,
                        yvj,
                        gvj,
                        wvj,
                    )
                    metric_label = "val"
                else:
                    eval_Q, eval_Theta, eval_y, eval_g, eval_w = (
                        Qj[:1024],
                        Thetaj[:1024],
                        yj[:1024],
                        gj[:1024],
                        wj[:1024],
                    )
                    metric_label = "train"

                yp, gp = batched_value_and_grad_wrt_q(params, eval_Q, eval_Theta)
                eval_w = eval_w / jnp.maximum(jnp.mean(eval_w), 1e-12)
                vm = jnp.mean(eval_w * (yp - eval_y) ** 2)
                gm = jnp.mean(eval_w * jnp.sum((gp - eval_g) ** 2, axis=1))
                selection_obj = vm + grad_weight * gm

                if float(selection_obj) < best_val:
                    best_val = float(selection_obj)
                    best_params = _copy_params(params)
                print(
                    f"Epoch {epoch:4d} | {metric_label} value MSE: {vm:.4e} "
                    f"| {metric_label} grad MSE: {gm:.4e}"
                )

    return best_params


def convexity_check(
    params: Params,
    key: jax.Array,
    q_dim: int,
    theta_dim: int,
    num_checks: int = 100,
    tolerance: float = 1e-5,
) -> bool:
    """Randomly test Jensen's inequality for fixed theta."""
    if num_checks <= 0:
        raise ValueError("num_checks must be positive")

    keys = jax.random.split(key, 4)
    q1 = jax.random.normal(keys[0], (num_checks, q_dim))
    q2 = jax.random.normal(keys[1], (num_checks, q_dim))
    theta = jax.random.normal(keys[2], (num_checks, theta_dim))
    t = jax.random.uniform(keys[3], (num_checks, 1))

    q_mix = t * q1 + (1.0 - t) * q2
    f_mix = batched_forward(params, q_mix, theta)
    f_rhs = (
        t[:, 0] * batched_forward(params, q1, theta)
        + (1.0 - t[:, 0]) * batched_forward(params, q2, theta)
    )
    return bool(jnp.all(f_mix <= f_rhs + tolerance))


def to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj
