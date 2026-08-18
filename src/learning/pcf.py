from __future__ import annotations

from functools import partial
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils import (
    Batch,
    Params,
    Section,
    _copy_params,
    _init_psi_params,
    _make_positive,
    _make_sections,
    _to_serializable,
)

convex_widths = (32, 32)
hyper_widths = (64, 64)


def _activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing activation"""
    return jax.nn.softplus(x)


def _output(x: jnp.ndarray) -> jnp.ndarray:
    """Nonnegative output"""
    return jax.nn.softplus(x)


def init(
    key: jax.Array,
    input_dim: int,
    parameter_dim: int,
    convex_widths: Sequence[int] = convex_widths,
    hyper_widths: Sequence[int] = hyper_widths,
) -> Params:
    """Initialize PCF metadata and trainable psi parameters."""
    convex_widths = tuple(int(width) for width in convex_widths)
    hyper_widths = tuple(int(width) for width in hyper_widths)

    key_psi, _ = jax.random.split(key)
    layer_dims, sections, output_dim = _make_sections(input_dim, convex_widths)
    shapes = tuple(
        [section[2] for section in sections["W"]]
        + [section[2] for section in sections["V"]]
        + [section[2] for section in sections["omega"]]
    )
    sections = (sections["W"], sections["V"], sections["omega"])

    return {
        "input_dim": int(input_dim),
        "parameter_dim": int(parameter_dim),
        "convex_widths": convex_widths,
        "hyper_widths": hyper_widths,
        "layer_dims": layer_dims,
        "sections": sections,
        "shapes": shapes,
        "output_dim": int(output_dim),
        "hyper": _init_psi_params(key_psi, parameter_dim, hyper_widths, output_dim),
    }


@jax.jit
def _psi(parameter: jnp.ndarray, hyper: Params) -> jnp.ndarray:
    """Evaluate psi(parameter), matching the residual/feedforward form in pcf_sc."""
    W_psi, V_psi, omega_psi = hyper["W_psi"], hyper["V_psi"], hyper["omega_psi"]
    out = jnp.dot(V_psi[0], parameter) + omega_psi[0]
    for layer_idx in range(1, len(V_psi)):
        out = _activation(out)
        out = (
            jnp.dot(W_psi[layer_idx - 1], out)
            + jnp.dot(V_psi[layer_idx], parameter)
            + omega_psi[layer_idx]
        )
    return out.reshape(-1)


@partial(jax.jit, static_argnames=("sections",))
def unpack(
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    emitted: jnp.ndarray,
) -> Params:
    """Unpack psi(parameter)'s flat vector into main ICNN tensors."""
    tensors: Params = {"W": [], "V": [], "omega": []}
    sections_w, sections_v, sections_o = sections
    for start, end, shape in sections_w:
        raw = emitted[start:end].reshape(shape)
        fan_in = shape[1]
        tensors["W"].append(_make_positive(raw) / fan_in)
    for start, end, shape in sections_v:
        fan_in = shape[1]
        tensors["V"].append(emitted[start:end].reshape(shape) / np.sqrt(fan_in))
    for start, end, shape in sections_o:
        tensors["omega"].append(emitted[start:end].reshape(shape))
    return tensors


@partial(jax.jit, static_argnames=("sections",))
def forward(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the scalar PCF output for one ``(model_input, parameter)`` pair."""
    emitted = _psi(parameter, hyper)
    tensors = unpack(sections, emitted)
    W, V, omega = tensors["W"], tensors["V"], tensors["omega"]

    z = _activation(jnp.dot(V[0], model_input) + omega[0])
    for layer_idx in range(1, len(V) - 1):
        z = _activation(
            jnp.dot(W[layer_idx - 1], z)
            + jnp.dot(V[layer_idx], model_input)
            + omega[layer_idx]
        )

    output_raw = (
        jnp.dot(W[-1], z).squeeze()
        + jnp.dot(V[-1], model_input).squeeze()
        + omega[-1].squeeze()
    )
    return _output(output_raw)


@partial(jax.jit, static_argnames=("sections",))
def value_and_grad(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate batched values and model_input-gradients."""
    values = jax.vmap(forward, in_axes=(None, None, 0, 0))(
        hyper, sections, model_input, parameter
    )
    grads = jax.vmap(
        jax.grad(forward, argnums=2),
        in_axes=(None, None, 0, 0),
    )(hyper, sections, model_input, parameter)
    return values, grads


class PCF:
    def __init__(self, params: Params):
        self.params = params

    def __call__(self, model_input: np.ndarray, parameter: np.ndarray) -> jnp.ndarray:
        values, _ = self.value_and_grad(model_input, parameter)
        return values

    def predict(self, model_input: np.ndarray, parameter: np.ndarray) -> jnp.ndarray:
        values, _ = self.value_and_grad(model_input, parameter)
        return values

    def grad(self, model_input: np.ndarray, parameter: np.ndarray) -> jnp.ndarray:
        _, grads = self.value_and_grad(model_input, parameter)
        return grads

    def value_and_grad(self, model_input: np.ndarray, parameter: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return value_and_grad(
            self.params["hyper"],
            self.params["sections"],
            jnp.asarray(model_input, dtype=jnp.float32),
            jnp.asarray(parameter, dtype=jnp.float32),
        )

    def to_dict(self) -> Params:
        return self.params

    def to_jsonable(self) -> Any:
        return _to_serializable(self.params)


def loss(
    hyper: Params,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    feasibility_data: Params,
    grad_weight: float = 1.0,
    value_weight: float = 1.0,
    feasibility_weight: float = 0.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Return total loss and value/gradient/feasibility components."""
    y_pred, g_pred = value_and_grad(hyper, sections, model_input, parameter)
    normalized_weights = w / jnp.maximum(jnp.mean(w), 1e-12)

    value_mse = jnp.mean(normalized_weights * (y_pred - y) ** 2)
    grad_mse = jnp.mean(normalized_weights * jnp.sum((g_pred - g) ** 2, axis=1))

    input_mean = jnp.asarray(feasibility_data["input_mean"], dtype=model_input.dtype)
    input_std = jnp.asarray(feasibility_data["input_std"], dtype=model_input.dtype)
    parameter_mean = jnp.asarray(feasibility_data["parameter_mean"], dtype=parameter.dtype)
    parameter_std = jnp.asarray(feasibility_data["parameter_std"], dtype=parameter.dtype)
    env_scale = jnp.asarray(feasibility_data["env_scale"], dtype=model_input.dtype)
    gamma = jnp.asarray(feasibility_data["gamma"], dtype=model_input.dtype)
    input_orig = model_input * input_std + input_mean
    grad_orig = g_pred * env_scale / input_std
    parameter_orig = parameter * parameter_std + parameter_mean
    parameter_part = parameter_orig[:, : int(feasibility_data["parameter_dim"])]
    gamma_factor = gamma[:, None] if jnp.ndim(gamma) else gamma
    g_matrix = jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype)
    b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype)
    b_theta = jnp.asarray(feasibility_data["b_theta"], dtype=model_input.dtype)
    violation = (input_orig - gamma_factor * grad_orig) @ g_matrix.T - (
        b_offset[None, :] + parameter_part @ b_theta.T
    )
    feas_mse = jnp.mean(jax.nn.relu(violation) ** 2)

    objective = value_weight * value_mse + grad_weight * grad_mse + feasibility_weight * feas_mse
    return objective, (value_mse, grad_mse, feas_mse)


def _batches(
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    y: jnp.ndarray,
    g: jnp.ndarray,
    w: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[Batch]:
    """Yield shuffled mini-batches."""
    permutation = jax.random.permutation(shuffle_key, model_input.shape[0])
    for start in range(0, model_input.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield model_input[indices], parameter[indices], y[indices], g[indices], w[indices]


def _step_fn(
    optimizer: optax.GradientTransformation,
    sections: tuple[tuple[Section, ...], tuple[Section, ...], tuple[Section, ...]],
    feasibility_data: Params,
    feasibility_weight: float,
):
    """Build one JIT-compiled optimizer step."""

    @jax.jit
    def train_step(
        hyper: Params,
        opt_state: optax.OptState,
        input_b: jnp.ndarray,
        parameter_b: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        wb: jnp.ndarray,
        grad_weight: float,
        value_weight: float,
    ):
        (objective, (vmse, gmse, fmse)), grads = jax.value_and_grad(
            loss,
            has_aux=True,
        )(
            hyper,
            sections,
            input_b,
            parameter_b,
            yb,
            gb,
            wb,
            feasibility_data,
            grad_weight,
            value_weight,
            feasibility_weight,
        )
        updates, opt_state = optimizer.update(grads, opt_state, hyper)
        hyper = optax.apply_updates(hyper, updates)
        return hyper, opt_state, objective, vmse, gmse, fmse

    return train_step


def train(
    model_input: np.ndarray,
    parameter: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    input_dim: int,
    parameter_dim: int,
    feasibility_data: Params,
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
    input_val: np.ndarray | None = None,
    parameter_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    feasibility_weight: float = 0.0,
    eval_interval: int = 50,
) -> Params:
    """Train an PCF with value and model_input-gradient supervision."""
    key = jax.random.PRNGKey(seed)
    if w is None:
        w = np.ones(model_input.shape[0])

    input_j = jnp.asarray(model_input, dtype=jnp.float32)
    parameter_j = jnp.asarray(parameter, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.float32)
    gj = jnp.asarray(g, dtype=jnp.float32)
    wj = jnp.asarray(w, dtype=jnp.float32)

    has_validation = all(v is not None for v in (input_val, parameter_val, y_val, g_val))
    if has_validation:
        if w_val is None:
            w_val = np.ones(input_val.shape[0])
        input_vj = jnp.asarray(input_val, dtype=jnp.float32)
        parameter_vj = jnp.asarray(parameter_val, dtype=jnp.float32)
        yvj = jnp.asarray(y_val, dtype=jnp.float32)
        gvj = jnp.asarray(g_val, dtype=jnp.float32)
        wvj = jnp.asarray(w_val, dtype=jnp.float32)

    params = init(key, input_dim, parameter_dim, convex_widths, hyper_widths)
    best_params = _copy_params(params)
    best_metric = float("inf")
    sections = params["sections"]

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
    train_step = _step_fn(optimizer, sections, feasibility_data, feasibility_weight)
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
        for input_b, parameter_b, yb, gb, wb in _batches(input_j, parameter_j, yj, gj, wj, batch_size, batch_key):
            params["hyper"], opt_state, _objective, _vmse, _gmse, _fmse = train_step(
                params["hyper"],
                opt_state,
                input_b,
                parameter_b,
                yb,
                gb,
                wb,
                grad_weight,
                value_weight,
            )

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            train_obj, train_parts = loss(
                params["hyper"],
                params["sections"],
                input_j,
                parameter_j,
                yj,
                gj,
                wj,
                feasibility_data,
                grad_weight,
                value_weight,
                feasibility_weight,
            )
            eval_obj, eval_parts = train_obj, train_parts
            val_msg = ""
            if has_validation:
                eval_obj, eval_parts = loss(
                    params["hyper"],
                    params["sections"],
                    input_vj,
                    parameter_vj,
                    yvj,
                    gvj,
                    wvj,
                    feasibility_data,
                    grad_weight,
                    value_weight,
                    feasibility_weight,
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


def _print_stats(name: str, x: jnp.ndarray) -> None:
    x = np.asarray(x)
    print(
        f"  {name:<18} shape={x.shape} mean={np.mean(x): .3e} "
        f"std={np.std(x):.3e} min={np.min(x):.3e} max={np.max(x):.3e}"
    )
