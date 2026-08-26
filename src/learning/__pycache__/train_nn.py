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


root = Path(__file__).resolve().parents[3]
data_dir = root / "data"
model_dir = root / "model"

train_data_path = data_dir / "PGM-rho=0.1_nx=2_N=10-train.npz"
test_data_path = data_dir / "PGM-rho=0.1_nx=2_N=10-test.npz"
model_path = model_dir / "linear-mpc-projection-mlp"


hidden_widths = (32, 32)
learning_rate = 1e-3
lr_decay_rate = 0.98
lr_decay_steps = 10000
eq_weight = 10.0
slack_positive_weight = 1.0
l2_reg = 0.0
batch_size = 128
epochs = 3000
eval_interval = 100
seed = 0
dtype = jnp.float64


def load_data(path: Path):
    with np.load(path) as data:
        model_input = np.asarray(data["input"], dtype=float).T
        parameter = np.asarray(data["parameter"], dtype=float).T
        projection = np.asarray(data["proj"], dtype=float).T

        if model_input.shape != projection.shape:
            raise ValueError(f"{path}: input and projection shapes differ: {model_input.shape} vs {projection.shape}")
        if model_input.shape[0] != parameter.shape[0]:
            raise ValueError(f"{path}: input and parameter sample counts differ")

        metadata = {
            "nx": int(data["nx"]),
            "nu": int(data["nu"]),
            "horizon": int(data["N"]),
            "input_dim": model_input.shape[1],
            "parameter_dim": parameter.shape[1],
        }
        return model_input, parameter, projection, np.ones(model_input.shape[0]), metadata


def feasibility_data(metadata):
    a_matrix = np.array([[2.0, -1.0], [1.0, 0.2]], dtype=float)
    b_matrix = np.array([[1.0], [0.0]], dtype=float)
    xmin, xmax = -5.0, 5.0

    nx = metadata["nx"]
    horizon = metadata["horizon"]
    a_ro, b_ro = _prediction_matrices(a_matrix, b_matrix, horizon)

    return {
        "g_matrix": np.vstack((b_ro, -b_ro)),
        "b_offset": np.concatenate((np.full(nx * horizon, xmax), np.full(nx * horizon, -xmin))),
        "b_theta": np.vstack((-a_ro, a_ro)),
    }


def evaluate(params, model_input, parameter, projection, weight, feas):
    objective, parts = nn.projection_loss(
        params,
        jnp.asarray(model_input, dtype=dtype),
        jnp.asarray(parameter, dtype=dtype),
        jnp.asarray(projection, dtype=dtype),
        jnp.asarray(weight, dtype=dtype),
        feas,
        eq_weight,
        slack_positive_weight,
    )
    v_tilde, s_tilde = nn.corrected_projection(
        params,
        jnp.asarray(model_input, dtype=dtype),
        jnp.asarray(parameter, dtype=dtype),
        feas,
    )
    corrected_mse = jnp.mean(jnp.sum((v_tilde - jnp.asarray(projection, dtype=dtype)) ** 2, axis=1))
    min_slack = jnp.min(s_tilde)
    return objective, parts, corrected_mse, min_slack


def export_params(params, metadata, feas):
    params = dict(params)
    params["model_type"] = "projection_mlp"
    params["target"] = "state_projection"
    params["mpc"] = {
        "nx": metadata["nx"],
        "nu": metadata["nu"],
        "horizon": metadata["horizon"],
        "input_dim": metadata["input_dim"],
        "parameter_dim": metadata["parameter_dim"],
    }
    params["feasibility"] = feas
    params["eq_weight"] = eq_weight
    params["slack_positive_weight"] = slack_positive_weight
    return params


def save(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{path}.pkl").open("wb") as f:
        pickle.dump(params, f)
    with Path(f"{path}.json").open("w") as f:
        json.dump(nn.ProjectionMLP(params).to_jsonable(), f)


def main():
    print(jax.devices())

    input_tr, parameter_tr, projection_tr, weight_tr, train_meta = load_data(train_data_path)
    input_te, parameter_te, projection_te, weight_te, _ = load_data(test_data_path)
    feas = feasibility_data(train_meta)

    print(f"samples: train={input_tr.shape[0]} test={input_te.shape[0]}")
    print(f"dims: input={train_meta['input_dim']} parameter={train_meta['parameter_dim']}")
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

    params = export_params(params, train_meta, feas)
    objective, parts, corrected_mse, min_slack = evaluate(params, input_te, parameter_te, projection_te, weight_te, feas)
    print(
        f"[test] objective: {objective:.4e} | projection mse: {parts[0]:.4e} "
        f"| equality mse: {parts[1]:.4e} | slack mse: {parts[2]:.4e} "
        f"| corrected projection mse: {corrected_mse:.4e} | min corrected slack: {min_slack:.4e}"
    )
    save(params, model_path)


if __name__ == "__main__":
    main()
