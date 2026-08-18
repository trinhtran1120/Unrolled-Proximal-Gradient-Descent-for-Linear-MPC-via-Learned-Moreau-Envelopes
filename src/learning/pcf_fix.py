from __future__ import annotations

from functools import partial
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = dict[str, Any]
Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
Section = tuple[int, int, tuple[int, ...]]

convex_widths = (32, 32)
hyper_widths = (64, 64)


def MAKE_POSITIVE(x: jnp.ndarray) -> jnp.ndarray:
    """Smoothly map ICNN recurrent weights to the nonnegative orthant."""
    return jax.nn.softplus(x)


def activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing activation used by the main ICNN and psi."""
    return jax.nn.softplus(x)


def output_activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing output shaped like 0.5 * dist(q, F)^2."""
    return jax.nn.softplus(x)


def rand_uniform(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    """Initialize dense parameters with fan-in scaling."""
    fan_in = shape[-1] if len(shape) > 1 else 1
    return jax.random.uniform(key, shape, minval=-0.5, maxval=0.5) / np.sqrt(fan_in)


def make_sections(q_dim: int, widths: Sequence[int]) -> tuple[tuple[int, ...], Params, int]:
    """Plan slices for psi(theta)'s flat output.

    Flatten order follows ``pcf_sc.py``: all W, then all V, then all omega.
    """
    layer_dims = (int(q_dim), *tuple(int(width) for width in widths), 1)
    sections_w: list[Section] = []
    sections_v: list[Section] = []
    sections_o: list[Section] = []
    offset = 0

    def append(sections: list[Section], shape: tuple[int, ...]) -> None:
        nonlocal offset
        size = int(np.prod(shape))
        sections.append((offset, offset + size, shape))
        offset += size

    for layer_idx in range(2, len(layer_dims)):
        append(sections_w, (layer_dims[layer_idx], layer_dims[layer_idx - 1]))
    for layer_idx in range(1, len(layer_dims)):
        append(sections_v, (layer_dims[layer_idx], q_dim))
    for layer_idx in range(1, len(layer_dims)):
        append(sections_o, (layer_dims[layer_idx],))

    sections = {
        "W": tuple(sections_w),
        "V": tuple(sections_v),
        "omega": tuple(sections_o),
    }
    return layer_dims, sections, offset


def init_psi_params(
    key: jax.Array,
    theta_dim: int,
    hyper_widths: Sequence[int],
    output_dim: int,
) -> Params:
    """Initialize the residual/feedforward psi network from ``pcf_sc.py``."""
    dims = (int(theta_dim), *tuple(int(width) for width in hyper_widths), int(output_dim))
    keys = jax.random.split(key, 3 * (len(dims) - 1))
    key_idx = 0
    W_psi: list[jnp.ndarray] = []
    V_psi: list[jnp.ndarray] = []
    omega_psi: list[jnp.ndarray] = []

    for layer_idx in range(1, len(dims)):
        if layer_idx > 1:
            W_psi.append(
                rand_uniform(keys[key_idx], (dims[layer_idx], dims[layer_idx - 1]))
            )
            key_idx += 1
        V_psi.append(rand_uniform(keys[key_idx], (dims[layer_idx], theta_dim)))
        key_idx += 1
        omega_psi.append(rand_uniform(keys[key_idx], (dims[layer_idx],)))
        key_idx += 1

    return {"W_psi": W_psi, "V_psi": V_psi, "omega_psi": omega_psi}


def init_lpcf_params(
    key: jax.Array,
    q_dim: int,
    theta_dim: int,
    convex_widths: Sequence[int] = convex_widths,
    hyper_widths: Sequence[int] = hyper_widths,
) -> Params:
    """Initialize LPCF metadata and trainable psi parameters."""
    convex_widths = tuple(int(width) for width in convex_widths)
    hyper_widths = tuple(int(width) for width in hyper_widths)
    key_psi, _ = jax.random.split(key)
    layer_dims, sections, output_dim = make_sections(q_dim, convex_widths)
    shapes = tuple(
        [section[2] for section in sections["W"]]
        + [section[2] for section in sections["V"]]
        + [section[2] for section in sections["omega"]]
    )
    sections_static = (sections["W"], sections["V"], sections["omega"])

    return {
        "q_dim": int(q_dim),
        "theta_dim": int(theta_dim),
        "convex_widths": convex_widths,
        "hyper_widths": hyper_widths,
        "layer_dims": layer_dims,
        "sections": sections,
        "sections_static": sections_static,
        "shapes": shapes,
        "output_dim": int(output_dim),
        "positive_map": "softplus",
        "activation": "softplus",
        "output_activation": "squared_relu",
        "hyper": init_psi_params(key_psi, theta_dim, hyper_widths, output_dim),
    }


@jax.jit
def psi_flat_function(theta: jnp.ndarray, hyper: Params) -> jnp.ndarray:
    """Evaluate psi(theta), matching the residual/feedforward form in pcf_sc."""
    W_psi, V_psi, omega_psi = hyper["W_psi"], hyper["V_psi"], hyper["omega_psi"]
    out = jnp.dot(V_psi[0], theta) + omega_psi[0]
    for layer_idx in range(1, len(V_psi)):
        out = activation(out)
        out = (
            jnp.dot(W_psi[layer_idx - 1], out)
            + jnp.dot(V_psi[layer_idx], theta)
            + omega_psi[layer_idx]
        )
    return out.squeeze()


@partial(jax.jit, static_argnames=("sections",))
def unpack_lpcf_tensors_from_sections(
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    emitted: jnp.ndarray,
) -> Params:
    """Unpack psi(theta)'s flat vector into fan-in-scaled main ICNN tensors."""
    tensors: Params = {"W": [], "raw_W": [], "V": [], "omega": []}
    sections_w, sections_v, sections_o = sections
    for start, end, shape in sections_w:
        raw = emitted[start:end].reshape(shape)
        fan_in = shape[1]
        tensors["raw_W"].append(raw)
        tensors["W"].append(MAKE_POSITIVE(raw) / fan_in)
    for start, end, shape in sections_v:
        fan_in = shape[1]
        tensors["V"].append(emitted[start:end].reshape(shape) / np.sqrt(fan_in))
    for start, end, shape in sections_o:
        tensors["omega"].append(emitted[start:end].reshape(shape))
    return tensors


def unpack_lpcf_tensors(params: Params, emitted: jnp.ndarray) -> Params:
    """Public unpack helper used by tests and diagnostics."""
    return unpack_lpcf_tensors_from_sections(params["sections_static"], emitted)


@partial(jax.jit, static_argnames=("sections",))
def lpcf_forward(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the scalar LPCF output for one ``(q, theta)`` pair."""
    emitted = psi_flat_function(theta, hyper)
    tensors = unpack_lpcf_tensors_from_sections(sections, emitted)
    W, V, omega = tensors["W"], tensors["V"], tensors["omega"]

    z = activation(jnp.dot(V[0], q) + omega[0])
    for layer_idx in range(1, len(V) - 1):
        z = activation(
            jnp.dot(W[layer_idx - 1], z)
            + jnp.dot(V[layer_idx], q)
            + omega[layer_idx]
        )

    raw = (
        jnp.dot(W[-1], z).squeeze()
        + jnp.dot(V[-1], q).squeeze()
        + omega[-1].squeeze()
    )
    return output_activation(raw)


@partial(jax.jit, static_argnames=("sections",))
def _batched_value_and_grad_from_hyper(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate batched values and q-gradients."""
    values = jax.vmap(lpcf_forward, in_axes=(None, None, 0, 0))(
        hyper, sections, q, theta
    )
    grads = jax.vmap(
        jax.grad(lpcf_forward, argnums=2),
        in_axes=(None, None, 0, 0),
    )(hyper, sections, q, theta)
    return values, grads


def value_and_grad(params: Params, q: np.ndarray, theta: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return batched values and q-gradients."""
    return _batched_value_and_grad_from_hyper(
        params["hyper"],
        params["sections_static"],
        jnp.asarray(q, dtype=jnp.float32),
        jnp.asarray(theta, dtype=jnp.float32),
    )


def predict(params: Params, q: np.ndarray, theta: np.ndarray) -> jnp.ndarray:
    """Return batched scalar predictions."""
    values, _ = value_and_grad(params, q, theta)
    return values


def grad(params: Params, q: np.ndarray, theta: np.ndarray) -> jnp.ndarray:
    """Return batched q-gradients."""
    _, grads = value_and_grad(params, q, theta)
    return grads


@partial(jax.jit, static_argnames=("sections",))
def _loss_from_hyper(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    q: jnp.ndarray,
    theta: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    grad_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Return total loss and value/gradient components."""
    y_pred, g_pred = _batched_value_and_grad_from_hyper(hyper, sections, q, theta)
    normalized_weights = w / jnp.maximum(jnp.mean(w), 1e-12)

    value_mse = jnp.mean(normalized_weights * (y_pred - y) ** 2)
    grad_mse = jnp.mean(normalized_weights * jnp.sum((g_pred - g) ** 2, axis=1))
    return value_weight * value_mse + grad_weight * grad_mse, (value_mse, grad_mse)


def loss_fn(
    params: Params,
    q: jnp.ndarray,
    theta: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    grad_weight: float = 1.0,
    value_weight: float = 1.0,
    feasibility_data: Params | None = None,
    l2_reg: float = 0.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Public loss wrapper compatible with the existing training script."""
    del l2_reg
    feasibility_weight = 0.0 if feasibility_data is None else float(feasibility_data.get("weight", 0.0))
    objective, (value_mse, grad_mse) = _loss_from_hyper(
        params["hyper"],
        params["sections_static"],
        q,
        theta,
        y,
        g,
        w,
        grad_weight,
        value_weight,
    )
    feas_mse = jnp.array(0.0, dtype=q.dtype)
    if feasibility_weight:
        _, g_pred = _batched_value_and_grad_from_hyper(
            params["hyper"], params["sections_static"], q, theta
        )
        q_mean = jnp.asarray(feasibility_data["q_mean"], dtype=q.dtype)
        q_std = jnp.asarray(feasibility_data["q_std"], dtype=q.dtype)
        theta_mean = jnp.asarray(feasibility_data["theta_mean"], dtype=theta.dtype)
        theta_std = jnp.asarray(feasibility_data["theta_std"], dtype=theta.dtype)
        env_scale = jnp.asarray(feasibility_data["env_scale"], dtype=q.dtype)
        gamma = jnp.asarray(feasibility_data["gamma"], dtype=q.dtype)
        q_orig = q * q_std + q_mean
        grad_orig = g_pred * env_scale / q_std
        theta_orig = theta * theta_std + theta_mean
        x0 = theta_orig[:, : int(feasibility_data["x0_dim"])]
        if feasibility_data.get("gamma_feature") == "log_gamma":
            gamma = jnp.exp(theta_orig[:, int(feasibility_data["x0_dim"])])
        elif feasibility_data.get("gamma_feature") == "gamma":
            gamma = theta_orig[:, int(feasibility_data["x0_dim"])]
        gamma_factor = gamma[:, None] if jnp.ndim(gamma) else gamma
        g_matrix = jnp.asarray(feasibility_data["g_matrix"], dtype=q.dtype)
        b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=q.dtype)
        b_theta = jnp.asarray(feasibility_data["b_theta"], dtype=q.dtype)
        violation = (q_orig - gamma_factor * grad_orig) @ g_matrix.T - (
            b_offset[None, :] + x0 @ b_theta.T
        )
        feas_mse = jnp.mean(jax.nn.relu(violation) ** 2)
        objective += feasibility_weight * feas_mse
    return objective, (value_mse, grad_mse, feas_mse)


def batch_iterator(
    q: jnp.ndarray,
    theta: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[Batch]:
    """Yield shuffled mini-batches."""
    permutation = jax.random.permutation(shuffle_key, q.shape[0])
    for start in range(0, q.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield q[indices], theta[indices], y[indices], g[indices], w[indices]


def make_train_step(
    optimizer: optax.GradientTransformation,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
):
    """Build one JIT-compiled optimizer step."""

    @jax.jit
    def train_step(
        hyper: Params,
        opt_state: optax.OptState,
        qb: jnp.ndarray,
        thetab: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        wb: jnp.ndarray,
        grad_weight: float,
        value_weight: float,
    ):
        (loss, (vmse, gmse)), grads = jax.value_and_grad(
            _loss_from_hyper,
            has_aux=True,
        )(
            hyper,
            sections,
            qb,
            thetab,
            yb,
            gb,
            wb,
            grad_weight,
            value_weight,
        )
        updates, opt_state = optimizer.update(grads, opt_state, hyper)
        hyper = optax.apply_updates(hyper, updates)
        return hyper, opt_state, loss, vmse, gmse

    return train_step


def _copy_params(params: Params) -> Params:
    return jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        params,
    )


def train_lpcf(
    q: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    q_dim: int,
    theta_dim: int,
    w: np.ndarray | None = None,
    convex_widths: Sequence[int] = convex_widths,
    hyper_widths: Sequence[int] = hyper_widths,
    lr: float = 1e-3,
    lr_decay_rate: float = 1.0,
    lr_decay_steps: int = 1000,
    selection_metric: str = "objective",
    grad_weight: float = 1.0,
    value_weight: float = 1.0,
    l2_reg: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 0,
    q_val: np.ndarray | None = None,
    theta_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    feasibility_data: Params | None = None,
    eval_interval: int = 50,
) -> Params:
    """Train an LPCF with value and q-gradient supervision."""
    key = jax.random.PRNGKey(seed)
    if w is None:
        w = np.ones(q.shape[0])

    qj = jnp.asarray(q, dtype=jnp.float32)
    thetaj = jnp.asarray(theta, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)
    wj = jnp.asarray(w, dtype=jnp.float32)

    has_validation = q_val is not None or theta_val is not None or y_val is not None or g_val is not None or w_val is not None
    if has_validation:
        if w_val is None:
            w_val = np.ones(q_val.shape[0])
        qvj = jnp.asarray(q_val, dtype=jnp.float32)
        thetavj = jnp.asarray(theta_val, dtype=jnp.float32)
        yvj = jnp.asarray(y_val, dtype=jnp.float32)
        gvj = jnp.asarray(g_val, dtype=jnp.float32)
        wvj = jnp.asarray(w_val, dtype=jnp.float32)

    params = init_lpcf_params(key, q_dim, theta_dim, convex_widths, hyper_widths)
    best_params = _copy_params(params)
    best_metric = float("inf")
    sections = params["sections_static"]

    if lr_decay_rate == 1.0:
        learning_rate = lr
    else:
        learning_rate = optax.exponential_decay(
            init_value=lr,
            transition_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=False,
        )
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2_reg)
    opt_state = optimizer.init(params["hyper"])
    train_step = make_train_step(optimizer, sections)
    eval_interval = max(1, int(eval_interval))

    def metric_value(objective, parts):
        if selection_metric == "value":
            return float(parts[0])
        if selection_metric == "grad":
            return float(parts[1])
        if selection_metric == "feasibility":
            return float(parts[2])
        return float(objective)

    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for qb, thetab, yb, gb, wb in batch_iterator(qj, thetaj, yj, gj, wj, batch_size, batch_key):
            params["hyper"], opt_state, _loss, _vmse, _gmse = train_step(
                params["hyper"],
                opt_state,
                qb,
                thetab,
                yb,
                gb,
                wb,
                grad_weight,
                value_weight,
            )

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            train_obj, train_parts = loss_fn(
                params,
                qj,
                thetaj,
                yj,
                gj,
                wj,
                grad_weight,
                value_weight,
                feasibility_data,
                l2_reg,
            )
            eval_obj, eval_parts = train_obj, train_parts
            val_msg = ""
            if has_validation:
                eval_obj, eval_parts = loss_fn(
                    params,
                    qvj,
                    thetavj,
                    yvj,
                    gvj,
                    wvj,
                    grad_weight,
                    value_weight,
                    feasibility_data,
                    l2_reg,
                )
                val_msg = (
                    f"\n           val value mse: {eval_parts[0]:.4e} "
                    f"| val grad mse: {eval_parts[1]:.4e} "
                    f"| val feasibility mse: {eval_parts[2]:.4e}"
                )

            metric = metric_value(eval_obj, eval_parts)
            if metric < best_metric:
                best_metric = metric
                best_params = _copy_params(params)

            print(
                f"epoch {epoch:4d}\n"
                f"           train value mse: {train_parts[0]:.4e} "
                f"| train grad mse: {train_parts[1]:.4e} "
                f"| train feasibility mse: {train_parts[2]:.4e}"
                f"{val_msg}"
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


def _print_stats(name: str, x: jnp.ndarray) -> None:
    x = np.asarray(x)
    print(
        f"  {name:<18} shape={x.shape} mean={np.mean(x): .3e} "
        f"std={np.std(x):.3e} min={np.min(x):.3e} max={np.max(x):.3e}"
    )
