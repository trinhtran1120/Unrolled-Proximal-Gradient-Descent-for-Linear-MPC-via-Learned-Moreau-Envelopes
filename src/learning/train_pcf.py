from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pcf
from utils import _prediction_matrices


root = Path(__file__).resolve().parents[2]
data_dir = root / "data"
model_dir = root / "model"

train_data_path = data_dir / "pgm_nx=2_n=10-train_adaptive.npz"
test_data_path = data_dir / "pgm_nx=2_n=10-test_adaptive.npz"
model_path = model_dir / "linear-mpc-pcf_adaptive"


convex_widths = (16, 16)
hyper_widths = (64, 64)
learning_rate = 1e-3
lr_decay_rate = 0.98
lr_decay_steps = 10000
value_weight = 1.0
grad_weight = 10.0
feasibility_weight = 0.0
l2_reg = 0.0
batch_size = 128
epochs = 3_000
eval_interval = 50
seed = 0
normalization_eps = 1e-8
normalize_data = False
normalize_env = True
gamma = 1.0


def load_data(path: Path):
    with np.load(path) as data:
        model_input = np.asarray(data["input"], dtype=float).T
        parameter = np.asarray(data["parameter"], dtype=float).T
        grad = np.asarray(data["grad"], dtype=float).T
        env = np.asarray(data["env"], dtype=float)

        if model_input.shape != grad.shape:
            raise ValueError(f"{path}: input and grad shapes differ: {model_input.shape} vs {grad.shape}")
        if model_input.shape[0] != parameter.shape[0]:
            raise ValueError(f"{path}: input and parameter sample counts differ")

        metadata = {
            "nx": int(data["nx"]),
            "nu": int(data["nu"]),
            "horizon": int(data["N"]),
            "input_dim": model_input.shape[1],
            "parameter_dim": parameter.shape[1],
            "adaptive": bool(np.asarray(data["adaptive"]).item()),
            "rho_initial": float(data["rho_initial"]) if "rho_initial" in data else None,
        }
        return model_input, parameter, env, grad, np.ones(model_input.shape[0]), metadata


def fit_normalization(model_input, parameter, env):
    input_mean = np.zeros(model_input.shape[1])
    input_std = np.ones(model_input.shape[1])
    parameter_mean = np.zeros(parameter.shape[1])
    parameter_std = np.ones(parameter.shape[1])
    env_scale = 1.0

    if normalize_data:
        input_mean = model_input.mean(axis=0)
        input_std = np.maximum(model_input.std(axis=0), normalization_eps)
        parameter_mean = parameter.mean(axis=0)
        parameter_std = np.maximum(parameter.std(axis=0), normalization_eps)

    if normalize_env or normalize_data:
        env_scale = max(float(env.std()), normalization_eps)

    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "parameter_mean": parameter_mean,
        "parameter_std": parameter_std,
        "env_scale": env_scale,
    }


def normalize(model_input, parameter, env, grad, stats):
    input_std = np.asarray(stats["input_std"])
    env_scale = float(stats["env_scale"])
    return (
        (model_input - stats["input_mean"]) / input_std,
        (parameter - stats["parameter_mean"]) / stats["parameter_std"],
        env / env_scale,
        grad * input_std / env_scale,
    )


def feasibility_data(metadata, stats):
    a_matrix = np.array([[2.0, -1.0], [1.0, 0.2]], dtype=float)
    b_matrix = np.array([[1.0], [0.0]], dtype=float)
    xmin, xmax = -5.0, 5.0
    umin, umax = -1.0, 1.0

    nx = metadata["nx"]
    nu = metadata["nu"]
    horizon = metadata["horizon"]
    a_ro, b_ro = _prediction_matrices(a_matrix, b_matrix, horizon)
    identity = np.eye(nu * horizon)
    zeros_parameter = np.zeros((nu * horizon, metadata["parameter_dim"]))

    return {
        **stats,
        "gamma": gamma,
        "parameter_dim": metadata["parameter_dim"],
        "g_matrix": np.vstack((b_ro, -b_ro, identity, -identity)),
        "b_offset": np.concatenate(
            (
                np.full(nx * horizon, xmax),
                np.full(nx * horizon, -xmin),
                np.full(nu * horizon, umax),
                np.full(nu * horizon, -umin),
            )
        ),
        "b_theta": np.vstack((-a_ro, a_ro, zeros_parameter, zeros_parameter)),
    }


def evaluate(params, model_input, parameter, env, grad, stats):
    model_input_norm, parameter_norm, _, _ = normalize(model_input, parameter, env, grad, stats)
    model = pcf.PCF(params)
    env_pred_norm, grad_pred_norm = model.value_and_grad(model_input_norm, parameter_norm)

    env_scale = float(stats["env_scale"])
    input_std = jnp.asarray(stats["input_std"])
    env_pred = env_pred_norm * env_scale
    grad_pred = grad_pred_norm * env_scale / input_std

    value_mse = jnp.mean((env_pred - jnp.asarray(env)) ** 2)
    grad_mse = jnp.mean(jnp.sum((grad_pred - jnp.asarray(grad)) ** 2, axis=1))
    return value_mse, grad_mse


def export_params(params, metadata, stats):
    params = dict(params)
    params["model_type"] = "pcf"
    params["target"] = "squared_distance"
    params["gamma"] = gamma
    params["rho_initial"] = metadata["rho_initial"]
    params["adaptive_data"] = metadata["adaptive"]
    params["mpc"] = {
        "nx": metadata["nx"],
        "nu": metadata["nu"],
        "horizon": metadata["horizon"],
        "input_dim": metadata["input_dim"],
        "parameter_dim": metadata["parameter_dim"],
    }
    params["normalization"] = stats
    params["feasibility_weight"] = feasibility_weight
    return params


def save(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{path}.pkl").open("wb") as f:
        pickle.dump(params, f)
    with Path(f"{path}.json").open("w") as f:
        json.dump(pcf.PCF(params).to_jsonable(), f)


def main():
    print(jax.devices())

    input_tr, parameter_tr, env_tr, grad_tr, weight_tr, train_meta = load_data(train_data_path)
    input_te, parameter_te, env_te, grad_te, weight_te, _ = load_data(test_data_path)
    stats = fit_normalization(input_tr, parameter_tr, env_tr)

    input_tr, parameter_tr, env_tr, grad_tr = normalize(input_tr, parameter_tr, env_tr, grad_tr, stats)
    input_te, parameter_te, env_te, grad_te = normalize(input_te, parameter_te, env_te, grad_te, stats)
    feas = feasibility_data(train_meta, stats)

    print(f"samples: train={input_tr.shape[0]} test={input_te.shape[0]}")
    print(f"dims: input={train_meta['input_dim']} parameter={train_meta['parameter_dim']}")
    print(f"adaptive data: {train_meta['adaptive']}")
    print(f"weights: value={value_weight} grad={grad_weight} feasibility={feasibility_weight}")

    params = pcf.train(
        input_tr,
        parameter_tr,
        env_tr,
        grad_tr,
        train_meta["input_dim"],
        train_meta["parameter_dim"],
        feas,
        w=weight_tr,
        convex_widths=convex_widths,
        hyper_widths=hyper_widths,
        lr=learning_rate,
        lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps,
        grad_weight=grad_weight,
        value_weight=value_weight,
        l2_reg=l2_reg,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        input_val=input_te,
        parameter_val=parameter_te,
        y_val=env_te,
        g_val=grad_te,
        w_val=weight_te,
        feasibility_weight=feasibility_weight,
        eval_interval=eval_interval,
    )

    value_mse, grad_mse = evaluate(params, *load_data(test_data_path)[:4], stats)
    print(f"[test original] value mse: {value_mse:.4e} | grad mse: {grad_mse:.4e}")
    save(export_params(params, train_meta, stats), model_path)


if __name__ == "__main__":
    main()
