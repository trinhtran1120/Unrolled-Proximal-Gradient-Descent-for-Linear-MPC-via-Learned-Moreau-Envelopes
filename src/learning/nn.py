from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import unfreeze

jax.config.update("jax_enable_x64", True)

from utils import Params, _batches, _copy_params, _dense_layer_names, to_jsonable

hidden_widths = (64, 64)
dtype = jnp.float64

# ============================================================================
# Projection Layer
# ============================================================================
def precompute_projection(g_matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    g_matrix_np = np.asarray(g_matrix)
    slack_dim = g_matrix_np.shape[0]
    E = np.concatenate((g_matrix_np, np.eye(slack_dim, dtype=g_matrix_np.dtype)), axis=1)
    E_pinv = np.linalg.solve(E @ E.T, E).T

    return jnp.asarray(E), jnp.asarray(E_pinv)


def projection_layer(
    z_bar: jnp.ndarray,
    parameter: jnp.ndarray,
    E: jnp.ndarray,
    b_offset: jnp.ndarray,
    b_para: jnp.ndarray,
    E_pinv: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    input_dim = E.shape[1] - E.shape[0]
    b = b_offset + parameter[..., : b_para.shape[1]] @ b_para.T
    z_hat = z_bar - (z_bar @ E.T - b) @ E_pinv.T
    v_hat = z_hat[..., :input_dim]
    s_hat = z_hat[..., input_dim:]

    return v_hat, s_hat


# ============================================================================
# Neural Projection MLP
# ============================================================================


class NeuralProjection(nn.Module):
    hidden_layers: tuple[int, ...]
    output_dim: int

    def setup(self) -> None:
        self.kernel_init = nn.initializers.xavier_uniform()
        self.bias_init = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, model_input: jnp.ndarray, parameter: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = jnp.concatenate((model_input, parameter), axis=-1)
        for width in self.hidden_layers:
            z = nn.Dense(width, dtype=dtype, param_dtype=dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(z)
            z = nn.gelu(z, approximate=False)

        output = nn.Dense(self.output_dim, dtype=dtype, param_dtype=dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(z)
        v_bar = output[..., : model_input.shape[-1]]
        s_bar = output[..., model_input.shape[-1] :]
        return v_bar, s_bar


# ============================================================================
# Loss and Training
# ============================================================================
def _constraint_residual(v: jnp.ndarray, s: jnp.ndarray, parameter: jnp.ndarray, feasibility_data: Params) -> jnp.ndarray:
    Gx = jnp.asarray(feasibility_data["g_matrix"], dtype=v.dtype)
    b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=v.dtype)
    b_para = jnp.asarray(feasibility_data["b_para"], dtype=parameter.dtype)
    b = b_offset + parameter[..., : b_para.shape[1]] @ b_para.T
    return v @ Gx.T + s - b


def loss(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    projection: jnp.ndarray,
    weights: jnp.ndarray,
    feasibility_data: Params,
    eq_weight: float = 1.0,
    slack_positive_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    model = NeuralProjection(hidden_layers=tuple(params["hidden_layers"]), output_dim=params["output_dim"])
    v_bar, s_bar = model.apply({"params": params["flax_params"]}, model_input, parameter)
    v_hat, s_hat = projection_layer(
        jnp.concatenate((v_bar, s_bar), axis=-1),
        parameter,
        jnp.asarray(feasibility_data["E"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_para"], dtype=parameter.dtype),
        jnp.asarray(feasibility_data["E_pinv"], dtype=model_input.dtype),
    )

    residual = _constraint_residual(v_hat, s_hat, parameter, feasibility_data)
    sample_weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)

    projection_mse = jnp.mean(sample_weights * jnp.sum((v_hat - projection) ** 2, axis=1))
    equality_mse = jnp.mean(sample_weights * jnp.sum(residual**2, axis=1))
    slack_mse = jnp.mean(sample_weights * jnp.sum(jax.nn.relu(-s_hat) ** 2, axis=1))
    objective = projection_mse + eq_weight * equality_mse + slack_positive_weight * slack_mse
    return objective, (projection_mse, equality_mse, slack_mse)


def _gelu_grad(x: jnp.ndarray) -> jnp.ndarray:
    cdf = 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
    pdf = jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)
    return cdf + x * pdf


def _manual_forward(flax_params: Params, model_input: jnp.ndarray, parameter: jnp.ndarray):
    x = jnp.concatenate((model_input, parameter), axis=-1)
    activations = [x]
    preactivations = []

    layer_names = _dense_layer_names(flax_params)
    for name in layer_names[:-1]:
        layer = flax_params[name]
        x = activations[-1] @ layer["kernel"] + layer["bias"]
        preactivations.append(x)
        activations.append(nn.gelu(x, approximate=False))

    layer = flax_params[layer_names[-1]]
    z_hat = activations[-1] @ layer["kernel"] + layer["bias"]
    activations.append(z_hat)
    return z_hat, activations, preactivations, layer_names


def loss_and_grad(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    projection: jnp.ndarray,
    weights: jnp.ndarray,
    feasibility_data: Params,
    eq_weight: float = 1.0,
    slack_positive_weight: float = 1.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Params]:
    z_bar, activations, preactivations, layer_names = _manual_forward(params["flax_params"], model_input, parameter)
    E = jnp.asarray(feasibility_data["E"], dtype=model_input.dtype)
    E_pinv = jnp.asarray(feasibility_data["E_pinv"], dtype=model_input.dtype)
    v_hat, s_hat = projection_layer(
        z_bar,
        parameter,
        E,
        jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_para"], dtype=parameter.dtype),
        E_pinv,
    )

    Gx = jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype)
    residual = _constraint_residual(v_hat, s_hat, parameter, feasibility_data)
    sample_weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)
    scaled_weights = sample_weights[:, None] / model_input.shape[0]

    projection_mse = jnp.mean(sample_weights * jnp.sum((v_hat - projection) ** 2, axis=1))
    equality_mse = jnp.mean(sample_weights * jnp.sum(residual**2, axis=1))
    slack_mse = jnp.mean(sample_weights * jnp.sum(jax.nn.relu(-s_hat) ** 2, axis=1))
    objective = projection_mse + eq_weight * equality_mse + slack_positive_weight * slack_mse

    dz_v = 2.0 * scaled_weights * (v_hat - projection)
    dz_v += 2.0 * eq_weight * scaled_weights * (residual @ Gx)
    dz_s = 2.0 * eq_weight * scaled_weights * residual
    dz_s += 2.0 * slack_positive_weight * scaled_weights * jnp.where(s_hat < 0.0, s_hat, 0.0)
    dz = jnp.concatenate((dz_v, dz_s), axis=1) @ (jnp.eye(params["output_dim"], dtype=model_input.dtype) - E_pinv @ E)

    grads = {}
    for index in range(len(layer_names) - 1, -1, -1):
        name = layer_names[index]
        grads[name] = {
            "kernel": activations[index].T @ dz,
            "bias": jnp.sum(dz, axis=0),
        }
        if index > 0:
            dz = (dz @ params["flax_params"][name]["kernel"].T) * _gelu_grad(preactivations[index - 1])

    return objective, (projection_mse, equality_mse, slack_mse), grads


def corrected_projection(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    feasibility_data: Params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    model = NeuralProjection(hidden_layers=tuple(params["hidden_layers"]), output_dim=params["output_dim"])
    v_bar, s_bar = model.apply({"params": params["flax_params"]}, model_input, parameter)
    return projection_layer(
        jnp.concatenate((v_bar, s_bar), axis=-1),
        parameter,
        jnp.asarray(feasibility_data["E"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_para"], dtype=parameter.dtype),
        jnp.asarray(feasibility_data["E_pinv"], dtype=model_input.dtype),
    )


def _train_step_fn(
    optimizer: optax.GradientTransformation,
    model_metadata: Params,
    feasibility_data: Params,
    eq_weight: float,
    slack_positive_weight: float,
):
    feasibility_arrays = {
        "g_matrix": jnp.asarray(feasibility_data["g_matrix"], dtype=dtype),
        "b_offset": jnp.asarray(feasibility_data["b_offset"], dtype=dtype),
        "b_para": jnp.asarray(feasibility_data["b_para"], dtype=dtype),
        "E": jnp.asarray(feasibility_data["E"], dtype=dtype),
        "E_pinv": jnp.asarray(feasibility_data["E_pinv"], dtype=dtype),
    }

    @jax.jit
    def train_step(flax_params, opt_state, input_b, parameter_b, projection_b, weights_b):
        params = {**model_metadata, "flax_params": flax_params}
        objective, parts, grads = loss_and_grad(
            params,
            input_b,
            parameter_b,
            projection_b,
            weights_b,
            feasibility_arrays,
            eq_weight,
            slack_positive_weight,
        )
        updates, opt_state = optimizer.update(grads, opt_state, flax_params)
        flax_params = optax.apply_updates(flax_params, updates)
        return flax_params, opt_state, objective, parts

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
    input_j = jnp.asarray(model_input, dtype=dtype)
    parameter_j = jnp.asarray(parameter, dtype=dtype)
    projection_j = jnp.asarray(projection, dtype=dtype)
    weights_j = jnp.ones(input_j.shape[0], dtype=dtype) if w is None else jnp.asarray(w, dtype=dtype)

    has_validation = all(v is not None for v in (input_val, parameter_val, projection_val))
    if has_validation:
        input_vj = jnp.asarray(input_val, dtype=dtype)
        parameter_vj = jnp.asarray(parameter_val, dtype=dtype)
        projection_vj = jnp.asarray(projection_val, dtype=dtype)
        weights_vj = jnp.ones(input_vj.shape[0], dtype=dtype) if w_val is None else jnp.asarray(w_val, dtype=dtype)

    slack_dim = int(feasibility_data["g_matrix"].shape[0])
    output_dim = input_dim + slack_dim
    hidden_layers = tuple(int(width) for width in widths)
    model = NeuralProjection(hidden_layers=hidden_layers, output_dim=output_dim)
    variables = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, input_dim), dtype=dtype),
        jnp.zeros((1, parameter_dim), dtype=dtype),
    )
    model_metadata = {
        "input_dim": input_dim,
        "parameter_dim": parameter_dim,
        "slack_dim": slack_dim,
        "output_dim": output_dim,
        "hidden_layers": hidden_layers,
    }
    flax_params = unfreeze(variables["params"])
    best_flax_params = _copy_params(flax_params)
    best_metric = float("inf")

    learning_rate = lr if lr_decay_rate == 1.0 else optax.exponential_decay(lr, lr_decay_steps, lr_decay_rate)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2_reg)
    opt_state = optimizer.init(flax_params)
    train_step = _train_step_fn(optimizer, model_metadata, feasibility_data, eq_weight, slack_positive_weight)

    key = jax.random.PRNGKey(seed)
    eval_interval = max(1, int(eval_interval))
    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for input_b, parameter_b, projection_b, weights_b in _batches(input_j, parameter_j, projection_j, weights_j, batch_size, batch_key):
            flax_params, opt_state, _objective, _parts = train_step(flax_params, opt_state, input_b, parameter_b, projection_b, weights_b)

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            params = {**model_metadata, "flax_params": flax_params}
            train_obj, train_parts = loss(params, input_j, parameter_j, projection_j, weights_j, feasibility_data, eq_weight, slack_positive_weight)
            eval_obj, eval_parts = train_obj, train_parts
            val_msg = ""
            if has_validation:
                eval_obj, eval_parts = loss(params, input_vj, parameter_vj, projection_vj, weights_vj, feasibility_data, eq_weight, slack_positive_weight)
                val_msg = (
                    f"\n           val total loss: {eval_obj:.4e} "
                    f"| val projection mse: {eval_parts[0]:.4e} "
                    f"| val equality mse: {eval_parts[1]:.4e} "
                    f"| val slack mse: {eval_parts[2]:.4e}"
                )

            if float(eval_obj) < best_metric:
                best_metric = float(eval_obj)
                best_flax_params = _copy_params(flax_params)

            print(
                f"epoch {epoch:4d}\n"
                f"           train total loss: {train_obj:.4e} "
                f"| train projection mse: {train_parts[0]:.4e} "
                f"| train equality mse: {train_parts[1]:.4e} "
                f"| train slack mse: {train_parts[2]:.4e}"
                f"{val_msg}"
            )

    return {**model_metadata, "flax_params": best_flax_params}


Network = NeuralProjection
