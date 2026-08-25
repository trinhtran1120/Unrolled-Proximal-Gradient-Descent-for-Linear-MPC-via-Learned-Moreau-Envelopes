from __future__ import annotations

from typing import Any, Iterator, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from utils import Params, _copy_params, _rand_uniform, _to_serializable

hidden_widths = (64, 64)
dtype = jnp.float64


def init(
    key: jax.Array,
    input_dim: int,
    parameter_dim: int,
    output_dim: int,
    widths: Sequence[int] = hidden_widths,
) -> Params:
    """Initialize a standard MLP for h_theta(U, x0) = [V_hat; s_hat]."""
    dims = (int(input_dim + parameter_dim), *tuple(int(w) for w in widths), int(output_dim))
    keys = jax.random.split(key, 2 * (len(dims) - 1))
    weights = []
    biases = []

    for layer_idx in range(len(dims) - 1):
        weights.append(_rand_uniform(keys[2 * layer_idx], (dims[layer_idx + 1], dims[layer_idx])))
        weights[-1] = weights[-1].astype(dtype)
        biases.append(jnp.zeros((dims[layer_idx + 1],), dtype=dtype))

    return {
        "input_dim": int(input_dim),
        "parameter_dim": int(parameter_dim),
        "output_dim": int(output_dim),
        "widths": tuple(int(w) for w in widths),
        "weights": weights,
        "biases": biases,
    }


def forward(params: Params, model_input: jnp.ndarray, parameter: jnp.ndarray) -> jnp.ndarray:
    """Evaluate raw [V_hat; s_hat] predictions."""
    z = jnp.concatenate((model_input, parameter), axis=-1)
    for W, b in zip(params["weights"][:-1], params["biases"][:-1]):
        z = jnp.tanh(z @ W.T + b)
    return z @ params["weights"][-1].T + params["biases"][-1]


def predict(params: Params, model_input: jnp.ndarray, parameter: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(forward, in_axes=(None, 0, 0))(params, model_input, parameter)


def _constraint_parameter(parameter: jnp.ndarray, feasibility_data: Params) -> jnp.ndarray:
    if "parameter_mean" not in feasibility_data or "parameter_std" not in feasibility_data:
        return parameter
    mean = jnp.asarray(feasibility_data["parameter_mean"], dtype=parameter.dtype)
    std = jnp.asarray(feasibility_data["parameter_std"], dtype=parameter.dtype)
    return parameter * std + mean


def state_constraint_rhs(parameter: jnp.ndarray, b_offset: jnp.ndarray, b_theta: jnp.ndarray) -> jnp.ndarray:
    return b_offset + parameter[..., : b_theta.shape[1]] @ b_theta.T


def affine_projection_layer(
    z_hat: jnp.ndarray,
    parameter: jnp.ndarray,
    g_matrix: jnp.ndarray,
    b_offset: jnp.ndarray,
    b_theta: jnp.ndarray,
    parameter_mean: jnp.ndarray | None = None,
    parameter_std: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project raw [V; s] predictions onto Gx V + s = b(x0)."""
    input_dim = g_matrix.shape[1]
    slack_dim = g_matrix.shape[0]
    if z_hat.shape[-1] != input_dim + slack_dim:
        raise ValueError(f"z_hat last dimension is {z_hat.shape[-1]}; expected {input_dim + slack_dim}")

    v_hat = z_hat[..., :input_dim]
    s_hat = z_hat[..., input_dim:]
    parameter_for_b = parameter
    if parameter_mean is not None and parameter_std is not None:
        parameter_for_b = parameter * parameter_std + parameter_mean
    b = state_constraint_rhs(parameter_for_b, b_offset, b_theta)
    residual = v_hat @ g_matrix.T + s_hat - b
    ee_t = g_matrix @ g_matrix.T + jnp.eye(slack_dim, dtype=z_hat.dtype)
    correction = jnp.linalg.solve(ee_t, residual[..., None])[..., 0]
    v_tilde = v_hat - correction @ g_matrix
    s_tilde = s_hat - correction
    return v_tilde, s_tilde


def projection_loss(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    projection: jnp.ndarray,
    weights: jnp.ndarray,
    feasibility_data: Params,
    eq_weight: float = 1.0,
    slack_positive_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Loss from the learned neural projection section of the paper."""
    z_hat = predict(params, model_input, parameter)
    input_dim = projection.shape[1]
    v_hat = z_hat[:, :input_dim]
    s_hat = z_hat[:, input_dim:]

    g_matrix = jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype)
    b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype)
    b_theta = jnp.asarray(feasibility_data["b_theta"], dtype=parameter.dtype)
    parameter_for_b = _constraint_parameter(parameter, feasibility_data)
    b = state_constraint_rhs(parameter_for_b, b_offset, b_theta)
    normalized_weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)

    projection_mse = jnp.mean(normalized_weights * jnp.sum((v_hat - projection) ** 2, axis=1))
    equality_mse = jnp.mean(normalized_weights * jnp.sum((v_hat @ g_matrix.T + s_hat - b) ** 2, axis=1))
    slack_mse = jnp.mean(normalized_weights * jnp.sum(jax.nn.relu(-s_hat) ** 2, axis=1))
    objective = projection_mse + eq_weight * equality_mse + slack_positive_weight * slack_mse
    return objective, (projection_mse, equality_mse, slack_mse)


def corrected_projection(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    feasibility_data: Params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run h_theta and then the closed-form affine projection layer."""
    z_hat = predict(params, model_input, parameter)
    return affine_projection_layer(
        z_hat,
        parameter,
        jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_theta"], dtype=parameter.dtype),
        jnp.asarray(feasibility_data["parameter_mean"], dtype=parameter.dtype)
        if "parameter_mean" in feasibility_data
        else None,
        jnp.asarray(feasibility_data["parameter_std"], dtype=parameter.dtype)
        if "parameter_std" in feasibility_data
        else None,
    )


def _batches(
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    projection: jnp.ndarray,
    weights: jnp.ndarray,
    batch_size: int,
    shuffle_key: jax.Array,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    permutation = jax.random.permutation(shuffle_key, model_input.shape[0])
    for start in range(0, model_input.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield model_input[indices], parameter[indices], projection[indices], weights[indices]


def _step_fn(optimizer: optax.GradientTransformation, feasibility_data: Params, eq_weight: float, slack_positive_weight: float):
    @jax.jit
    def train_step(params, opt_state, input_b, parameter_b, projection_b, weights_b):
        (objective, parts), grads = jax.value_and_grad(projection_loss, has_aux=True)(
            params,
            input_b,
            parameter_b,
            projection_b,
            weights_b,
            feasibility_data,
            eq_weight,
            slack_positive_weight,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, objective, parts

    return train_step


def train(
    model_input: np.ndarray,
    parameter: np.ndarray,
    projection: np.ndarray,
    input_dim: int,
    parameter_dim: int,
    feasibility_data: Params,
    w: np.ndarray | None = None,
    widths: Sequence[int] = hidden_widths,
    lr: float = 1e-3,
    lr_decay_rate: float = 1.0,
    lr_decay_steps: int = 1000,
    selection_metric: str = "objective",
    eq_weight: float = 1.0,
    slack_positive_weight: float = 1.0,
    l2_reg: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    seed: int = 0,
    input_val: np.ndarray | None = None,
    parameter_val: np.ndarray | None = None,
    projection_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    eval_interval: int = 50,
) -> Params:
    """Train a projection MLP with equality/slack penalties from the paper."""
    key = jax.random.PRNGKey(seed)
    if w is None:
        w = np.ones(model_input.shape[0])

    input_j = jnp.asarray(model_input, dtype=dtype)
    parameter_j = jnp.asarray(parameter, dtype=dtype)
    projection_j = jnp.asarray(projection, dtype=dtype)
    wj = jnp.asarray(w, dtype=dtype)

    has_validation = all(v is not None for v in (input_val, parameter_val, projection_val))
    if has_validation:
        if w_val is None:
            w_val = np.ones(input_val.shape[0])
        input_vj = jnp.asarray(input_val, dtype=dtype)
        parameter_vj = jnp.asarray(parameter_val, dtype=dtype)
        projection_vj = jnp.asarray(projection_val, dtype=dtype)
        wvj = jnp.asarray(w_val, dtype=dtype)

    output_dim = int(input_dim + feasibility_data["g_matrix"].shape[0])
    params = init(key, input_dim, parameter_dim, output_dim, widths)
    trainable_params = {"weights": params["weights"], "biases": params["biases"]}
    model_metadata = {
        "input_dim": params["input_dim"],
        "parameter_dim": params["parameter_dim"],
        "output_dim": params["output_dim"],
        "widths": params["widths"],
    }
    best_trainable_params = _copy_params(trainable_params)
    best_metric = float("inf")

    if lr_decay_rate == 1.0:
        learning_rate = lr
    else:
        learning_rate = optax.exponential_decay(lr, lr_decay_steps, lr_decay_rate, staircase=False)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2_reg)
    opt_state = optimizer.init(trainable_params)
    train_step = _step_fn(optimizer, feasibility_data, eq_weight, slack_positive_weight)
    eval_interval = max(1, int(eval_interval))

    def metric_value(objective, parts):
        if selection_metric == "projection":
            return float(parts[0])
        if selection_metric == "equality":
            return float(parts[1])
        if selection_metric == "slack":
            return float(parts[2])
        return float(objective)

    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for input_b, parameter_b, projection_b, weights_b in _batches(input_j, parameter_j, projection_j, wj, batch_size, batch_key):
            trainable_params, opt_state, _objective, _parts = train_step(
                trainable_params,
                opt_state,
                input_b,
                parameter_b,
                projection_b,
                weights_b,
            )

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            train_obj, train_parts = projection_loss(
                trainable_params, input_j, parameter_j, projection_j, wj, feasibility_data, eq_weight, slack_positive_weight
            )
            eval_obj, eval_parts = train_obj, train_parts
            val_msg = ""
            if has_validation:
                eval_obj, eval_parts = projection_loss(
                    trainable_params,
                    input_vj,
                    parameter_vj,
                    projection_vj,
                    wvj,
                    feasibility_data,
                    eq_weight,
                    slack_positive_weight,
                )
                val_msg = (
                    f"\n           val projection mse: {eval_parts[0]:.4e} "
                    f"| val equality mse: {eval_parts[1]:.4e} "
                    f"| val slack mse: {eval_parts[2]:.4e}"
                )

            metric = metric_value(eval_obj, eval_parts)
            if metric < best_metric:
                best_metric = metric
                best_trainable_params = _copy_params(trainable_params)

            print(
                f"epoch {epoch:4d}\n"
                f"           train projection mse: {train_parts[0]:.4e} "
                f"| train equality mse: {train_parts[1]:.4e} "
                f"| train slack mse: {train_parts[2]:.4e}"
                f"{val_msg}"
            )

    return {**model_metadata, **best_trainable_params}


class ProjectionMLP:
    def __init__(self, params: Params):
        self.params = params

    def __call__(self, model_input: np.ndarray, parameter: np.ndarray) -> jnp.ndarray:
        return predict(self.params, jnp.asarray(model_input, dtype=dtype), jnp.asarray(parameter, dtype=dtype))

    def corrected(self, model_input: np.ndarray, parameter: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return corrected_projection(
            self.params,
            jnp.asarray(model_input, dtype=dtype),
            jnp.asarray(parameter, dtype=dtype),
            self.params["feasibility"],
        )

    def to_dict(self) -> Params:
        return self.params

    def to_jsonable(self) -> Any:
        return _to_serializable(self.params)
