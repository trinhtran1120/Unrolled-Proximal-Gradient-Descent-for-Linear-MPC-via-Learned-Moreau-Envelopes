from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import nn
from utils import _prediction_matrices


root = Path(__file__).resolve().parents[2]
data_dir = root / "data"
model_dir = root / "model"

train_data_path = data_dir / "PGM_nx=12_N=20-train_adaptive.npz"
test_data_path = data_dir / "PGM_nx=12_N=20-test_adaptive.npz"
model_path = model_dir / "linear-mpc-projection-mlp_tanh"


hidden_widths = (64, 64)
activation = "tanh"
learning_rate = 1e-3
lr_decay_rate = 0.98
lr_decay_steps = 10000
eq_weight = 0.1
slack_positive_weight = 0.1
l2_reg = 0.1
batch_size = 128
epochs = 5000
eval_interval = 1000
seed = 0
dtype = jnp.float64


def load_data(path: Path):
    with np.load(path) as data:
        nx = int(data["nx"])
        nu = int(data["nu"])
        horizon = int(data["N"])

        raw_input = np.asarray(data["input"], dtype=float).T
        if "parameter" in data and "proj" in data:
            model_input = raw_input
            parameter = np.asarray(data["parameter"], dtype=float).T
            projection = np.asarray(data["proj"], dtype=float).T
        else:
            input_dim = nu * horizon
            model_input = raw_input[:, :input_dim]
            parameter = raw_input[:, input_dim : input_dim + nx]
            projection = model_input - np.asarray(data["grad"], dtype=float).T

        if model_input.shape != projection.shape:
            raise ValueError(f"{path}: input and projection shapes differ: {model_input.shape} vs {projection.shape}")
        if model_input.shape[0] != parameter.shape[0]:
            raise ValueError(f"{path}: input and parameter sample counts differ")

        metadata = {
            "nx": nx,
            "nu": nu,
            "horizon": horizon,
            "input_dim": model_input.shape[1],
            "parameter_dim": parameter.shape[1],
        }
        return model_input, parameter, projection, np.ones(model_input.shape[0]), metadata


def feasibility_data(metadata):
    a_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0992, 0.0, 0.0],
            [0.0, -0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0016, 0.0, 0.0, 0.0992, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0992],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.0, 0.0, 0.9846, 0.0, 0.0],
            [0.0, -0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0488, 0.0, 0.0, 0.9846, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9846],
        ],
        dtype=float,
    )
    b_matrix = np.array(
        [
            [0.0, -0.0726, 0.0, 0.0726],
            [-0.0726, 0.0, 0.0726, 0.0],
            [-0.0152, 0.0152, -0.0152, 0.0152],
            [0.0, -0.0006, -0.0000, 0.0006],
            [0.0006, 0.0, -0.0006, 0.0],
            [0.0106, 0.0106, 0.0106, 0.0106],
            [0.0, -1.4512, 0.0, 1.4512],
            [-1.4512, 0.0, 1.4512, 0.0],
            [-0.3049, 0.3049, -0.3049, 0.3049],
            [0.0, -0.0236, 0.0, 0.0236],
            [0.0236, 0.0, -0.0236, 0.0],
            [0.2107, 0.2107, 0.2107, 0.2107],
        ],
        dtype=float,
    )
    xmin = np.array([-np.pi / 6, -np.pi / 6, -np.inf, -np.inf, -np.inf, -1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    xmax = np.array([np.pi / 6, np.pi / 6, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    nx = metadata["nx"]
    horizon = metadata["horizon"]
    a_ro, b_ro = _prediction_matrices(a_matrix, b_matrix, horizon)
    xmin_stacked = np.tile(xmin, horizon)
    xmax_stacked = np.tile(xmax, horizon)
    upper_idx = np.flatnonzero(np.isfinite(xmax_stacked))
    lower_idx = np.flatnonzero(np.isfinite(xmin_stacked))
    g_matrix = np.vstack((b_ro[upper_idx, :], -b_ro[lower_idx, :]))
    E, E_pinv = nn.precompute_projection(jnp.asarray(g_matrix, dtype=dtype))

    return {
        "g_matrix": g_matrix,
        "b_offset": np.concatenate((xmax_stacked[upper_idx], -xmin_stacked[lower_idx])),
        "b_para": np.vstack((-a_ro[upper_idx, :], a_ro[lower_idx, :])),
        "E": np.asarray(E),
        "E_pinv": np.asarray(E_pinv),
    }


def save(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{path}.pkl").open("wb") as f:
        pickle.dump(params, f)
    with Path(f"{path}.json").open("w") as f:
        json.dump(nn.to_jsonable(params), f)


def main():
    print(jax.devices())

    input_tr, parameter_tr, projection_tr, weight_tr, train_meta = load_data(train_data_path)
    input_te, parameter_te, projection_te, weight_te, _ = load_data(test_data_path)
    feas = feasibility_data(train_meta)

    print(f"samples: train={input_tr.shape[0]} test={input_te.shape[0]}")
    print(f"dims: input={train_meta['input_dim']} parameter={train_meta['parameter_dim']}")
    print(f"activation: {activation}")
    print(f"weights: projection=1.0 equality={eq_weight} slack_positive={slack_positive_weight}")

    params = nn.train(
        input_tr,
        parameter_tr,
        projection_tr,
        train_meta["input_dim"],
        train_meta["parameter_dim"],
        feas,
        w=weight_tr,
        widths=hidden_widths,
        activation=activation,
        lr=learning_rate,
        lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps,
        eq_weight=eq_weight,
        slack_positive_weight=slack_positive_weight,
        l2_reg=l2_reg,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        input_val=input_te,
        parameter_val=parameter_te,
        projection_val=projection_te,
        w_val=weight_te,
        eval_interval=eval_interval,
    )

    params = {
        **params,
        "model_type": "projection_mlp",
        "target": "state_projection",
        "mpc": {
            "nx": train_meta["nx"],
            "nu": train_meta["nu"],
            "horizon": train_meta["horizon"],
            "input_dim": train_meta["input_dim"],
            "parameter_dim": train_meta["parameter_dim"],
        },
        "feasibility": feas,
        "eq_weight": eq_weight,
        "slack_positive_weight": slack_positive_weight,
    }

    input_test = jnp.asarray(input_te, dtype=dtype)
    parameter_test = jnp.asarray(parameter_te, dtype=dtype)
    projection_test = jnp.asarray(projection_te, dtype=dtype)
    weight_test = jnp.asarray(weight_te, dtype=dtype)

    objective, parts = nn.loss(params, input_test, parameter_test, projection_test, weight_test, feas, eq_weight, slack_positive_weight)
    v_tilde, s_tilde = nn.corrected_projection(params, input_test, parameter_test, feas)
    corrected_mse = jnp.mean(jnp.sum((v_tilde - projection_test) ** 2, axis=1))
    min_slack = jnp.min(s_tilde)
    print(
        f"[test] objective: {objective:.4e} | projection mse: {parts[0]:.4e} "
        f"| equality mse: {parts[1]:.4e} | slack mse: {parts[2]:.4e} "
        f"| corrected projection mse: {corrected_mse:.4e} | min corrected slack: {min_slack:.4e}"
    )
    save(params, model_path)


if __name__ == "__main__":
    main()
