from __future__ import annotations

from typing import Any, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from utils import Params, _batches, _copy_params, _rand_uniform, _to_serializable

hidden_widths = (64, 64)
dtype = jnp.float64


# ============================================================================
# Affine Projection Layer
# ============================================================================


def state_constraint_rhs(parameter: jnp.ndarray, b_offset: jnp.ndarray, b_theta: jnp.ndarray) -> jnp.ndarray:
    return b_offset + parameter[..., : b_theta.shape[1]] @ b_theta.T


def affine_projection_layer(
    z_hat: jnp.ndarray,
    parameter: jnp.ndarray,
    g_matrix: jnp.ndarray,
    b_offset: jnp.ndarray,
    b_theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project raw [V_hat; s_hat] onto G_x V + s = b_x(x0)."""
    input_dim = g_matrix.shape[1]
    slack_dim = g_matrix.shape[0]
    v_hat = z_hat[..., :input_dim]
    s_hat = z_hat[..., input_dim:]

    b = state_constraint_rhs(parameter, b_offset, b_theta)
    residual = v_hat @ g_matrix.T + s_hat - b
    ee_t = g_matrix @ g_matrix.T + jnp.eye(slack_dim, dtype=z_hat.dtype)
    correction = jnp.linalg.solve(ee_t, residual[..., None])[..., 0]

    return v_hat - correction @ g_matrix, s_hat - correction


# ============================================================================
# Neural Projection MLP
# ============================================================================


class ProjectionMLP:
    """MLP h_theta(U, x0) = [V_hat; s_hat]."""

    def __init__(
        self,
        input_dim: int | None = None,
        parameter_dim: int | None = None,
        slack_dim: int | None = None,
        hidden_layers: Sequence[int] = hidden_widths,
        seed: int = 0,
        params: Params | None = None,
    ):
        if params is not None:
            self.params = params
            self.input_dim = int(params["input_dim"])
            self.parameter_dim = int(params["parameter_dim"])
            self.slack_dim = int(params["slack_dim"])
            self.hidden_layers = tuple(int(width) for width in params["hidden_layers"])
            return

        if input_dim is None or parameter_dim is None or slack_dim is None:
            raise ValueError("input_dim, parameter_dim, and slack_dim are required when params is not provided")

        self.input_dim = int(input_dim)
        self.parameter_dim = int(parameter_dim)
        self.slack_dim = int(slack_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.params = self.init(jax.random.PRNGKey(seed))

    @property
    def output_dim(self) -> int:
        return self.input_dim + self.slack_dim

    def init(self, key: jax.Array) -> Params:
        layer_dims = (self.input_dim + self.parameter_dim, *self.hidden_layers, self.output_dim)
        keys = jax.random.split(key, len(layer_dims) - 1)

        weights = []
        biases = []
        for layer_idx in range(len(layer_dims) - 1):
            weights.append(_rand_uniform(keys[layer_idx], (layer_dims[layer_idx + 1], layer_dims[layer_idx])))
            biases.append(jnp.zeros((layer_dims[layer_idx + 1],), dtype=dtype))

        return {
            "input_dim": self.input_dim,
            "parameter_dim": self.parameter_dim,
            "slack_dim": self.slack_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "weights": weights,
            "biases": biases,
        }

    def __call__(self, params: Params, model_input: jnp.ndarray, parameter: jnp.ndarray) -> jnp.ndarray:
        z = jnp.concatenate((model_input, parameter), axis=-1)
        for weight, bias in zip(params["weights"][:-1], params["biases"][:-1]):
            z = jax.nn.gelu(z @ weight.T + bias)
        return z @ params["weights"][-1].T + params["biases"][-1]

    def split(self, z_hat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return z_hat[..., : self.input_dim], z_hat[..., self.input_dim :]

    def predict(self, model_input: np.ndarray | jnp.ndarray, parameter: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return self(
            self.params,
            jnp.asarray(model_input, dtype=dtype),
            jnp.asarray(parameter, dtype=dtype),
        )

    def corrected(
        self,
        model_input: np.ndarray | jnp.ndarray,
        parameter: np.ndarray | jnp.ndarray,
        feasibility_data: Params | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        feasibility = self.params["feasibility"] if feasibility_data is None else feasibility_data
        return corrected_projection(
            self.params,
            jnp.asarray(model_input, dtype=dtype),
            jnp.asarray(parameter, dtype=dtype),
            feasibility,
        )

    def to_dict(self) -> Params:
        return self.params

    def to_jsonable(self) -> Any:
        return _to_serializable(self.params)


# ============================================================================
# Loss and Training
# ============================================================================


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
    model = ProjectionMLP(params=params)
    z_hat = model(params, model_input, parameter)
    v_hat, s_hat = model.split(z_hat)

    g_matrix = jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype)
    b_offset = jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype)
    b_theta = jnp.asarray(feasibility_data["b_theta"], dtype=parameter.dtype)
    b = state_constraint_rhs(parameter, b_offset, b_theta)
    sample_weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)

    projection_mse = jnp.mean(sample_weights * jnp.sum((v_hat - projection) ** 2, axis=1))
    equality_mse = jnp.mean(sample_weights * jnp.sum((v_hat @ g_matrix.T + s_hat - b) ** 2, axis=1))
    slack_mse = jnp.mean(sample_weights * jnp.sum(jax.nn.relu(-s_hat) ** 2, axis=1))
    objective = projection_mse + eq_weight * equality_mse + slack_positive_weight * slack_mse
    return objective, (projection_mse, equality_mse, slack_mse)


def corrected_projection(
    params: Params,
    model_input: jnp.ndarray,
    parameter: jnp.ndarray,
    feasibility_data: Params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    model = ProjectionMLP(params=params)
    z_hat = model(params, model_input, parameter)
    return affine_projection_layer(
        z_hat,
        parameter,
        jnp.asarray(feasibility_data["g_matrix"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_offset"], dtype=model_input.dtype),
        jnp.asarray(feasibility_data["b_theta"], dtype=parameter.dtype),
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
        "b_theta": jnp.asarray(feasibility_data["b_theta"], dtype=dtype),
    }

    def objective_fn(trainable_params, input_b, parameter_b, projection_b, weights_b):
        params = {**model_metadata, **trainable_params}
        return projection_loss(
            params,
            input_b,
            parameter_b,
            projection_b,
            weights_b,
            feasibility_arrays,
            eq_weight,
            slack_positive_weight,
        )

    @jax.jit
    def train_step(trainable_params, opt_state, input_b, parameter_b, projection_b, weights_b):
        (objective, parts), grads = jax.value_and_grad(objective_fn, has_aux=True)(
            trainable_params,
            input_b,
            parameter_b,
            projection_b,
            weights_b,
        )
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, objective, parts

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
        weights_vj = (
            jnp.ones(input_vj.shape[0], dtype=dtype)
            if w_val is None
            else jnp.asarray(w_val, dtype=dtype)
        )

    slack_dim = int(feasibility_data["g_matrix"].shape[0])
    model = ProjectionMLP(input_dim, parameter_dim, slack_dim, widths, seed)
    model_metadata = {
        "input_dim": model.params["input_dim"],
        "parameter_dim": model.params["parameter_dim"],
        "slack_dim": model.params["slack_dim"],
        "output_dim": model.params["output_dim"],
        "hidden_layers": model.params["hidden_layers"],
    }
    trainable_params = {"weights": model.params["weights"], "biases": model.params["biases"]}
    best_trainable_params = _copy_params(trainable_params)
    best_metric = float("inf")

    learning_rate = lr if lr_decay_rate == 1.0 else optax.exponential_decay(lr, lr_decay_steps, lr_decay_rate)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2_reg)
    opt_state = optimizer.init(trainable_params)
    train_step = _train_step_fn(optimizer, model_metadata, feasibility_data, eq_weight, slack_positive_weight)

    key = jax.random.PRNGKey(seed)
    eval_interval = max(1, int(eval_interval))
    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for input_b, parameter_b, projection_b, weights_b in _batches(
            input_j,
            parameter_j,
            projection_j,
            weights_j,
            batch_size,
            batch_key,
        ):
            trainable_params, opt_state, _objective, _parts = train_step(
                trainable_params,
                opt_state,
                input_b,
                parameter_b,
                projection_b,
                weights_b,
            )

        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            params = {**model_metadata, **trainable_params}
            train_obj, train_parts = projection_loss(
                params, input_j, parameter_j, projection_j, weights_j, feasibility_data, eq_weight, slack_positive_weight
            )
            eval_obj, eval_parts = train_obj, train_parts
            val_msg = ""
            if has_validation:
                eval_obj, eval_parts = projection_loss(
                    params,
                    input_vj,
                    parameter_vj,
                    projection_vj,
                    weights_vj,
                    feasibility_data,
                    eq_weight,
                    slack_positive_weight,
                )
                val_msg = (
                    f"\n           val projection mse: {eval_parts[0]:.4e} "
                    f"| val equality mse: {eval_parts[1]:.4e} "
                    f"| val slack mse: {eval_parts[2]:.4e}"
                )

            if float(eval_obj) < best_metric:
                best_metric = float(eval_obj)
                best_trainable_params = _copy_params(trainable_params)

            print(
                f"epoch {epoch:4d}\n"
                f"           train projection mse: {train_parts[0]:.4e} "
                f"| train equality mse: {train_parts[1]:.4e} "
                f"| train slack mse: {train_parts[2]:.4e}"
                f"{val_msg}"
            )

    return {**model_metadata, **best_trainable_params}


Network = ProjectionMLP
