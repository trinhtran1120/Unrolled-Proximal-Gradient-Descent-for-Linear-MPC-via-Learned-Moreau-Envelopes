"""train and export the linear-mpc pcf model."""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parents[1]
data_dir = project_dir / "data"
model_dir = project_dir / "model"

train_data_path = data_dir / "PGM_nx=2_N=10-train_adaptive.npz"
test_data_path = data_dir / "PGM_nx=2_N=10-test_adaptive.npz"
model_path = model_dir / "linear-mpc-pcf_adaptive"

# The network learns psi(q; x0) = 0.5 * dist_F(x0)(q)^2. The true Moreau
# envelope for any gamma is phi^gamma = psi / gamma.
ENVELOPE_GAMMA = 1.0

convex_widths = [16, 16]
hyper_widths = [64, 64]
learning_rate = 1e-3
lr_decay_rate = 0.98
lr_decay_steps = 10000
grad_weight = 10.0
value_weight = 1.0
selection_metric = "grad"
feasibility_weight = 0.0
l2_reg = 0.0
batch_size = 128
epochs = 3000
eval_interval = 50
seed = 0
normalization_eps = 1e-8
normalize_data = False
normalize_env = True

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from pcf_fix import (
    init_lpcf_params,
    loss_fn,
    train_lpcf,
    to_serializable,
    value_and_grad,
)


def load_dataset(path: Path):
    """load mpc moreau-envelope data and split input into q and theta."""
    with np.load(path) as data:
        nx = int(data["nx"])
        nu = int(data["nu"])
        horizon = int(data["N"])
        q_dim = nu * horizon
        x0_dim = nx

        if {"q", "x0", "proj"}.issubset(data.files):
            q = np.asarray(data["q"], dtype=float).T
            x0 = np.asarray(data["x0"], dtype=float).T
            proj = np.asarray(data["proj"], dtype=float).T
            if q.shape != proj.shape:
                raise ValueError(f"{path} q and proj must have the same shape")
            if q.shape[1] != q_dim:
                raise ValueError(f"{path} q has dimension {q.shape[1]}, expected {q_dim}")
            if x0.shape != (q.shape[0], x0_dim):
                raise ValueError(f"{path} x0 must have shape {(q.shape[0], x0_dim)}, got {x0.shape}")

            displacement = q - proj
            y = 0.5 * np.sum(displacement**2, axis=1)
            g = displacement
        else:
            warnings.warn(
                f"{path} does not contain q/x0/proj; falling back to legacy "
                "input/env/grad labels. Regenerate adaptive data for "
                "projection-backed adaptive targets.",
                stacklevel=2,
            )
            x = np.asarray(data["input"], dtype=float).T
            expected_input_dim = q_dim + x0_dim
            if x.shape[1] != expected_input_dim:
                raise ValueError(
                    f"{path} input has {x.shape[1]} rows after transpose, expected "
                    f"nu*horizon + nx = {q_dim} + {x0_dim} = {expected_input_dim}"
                )

            q = x[:, :q_dim]
            x0 = x[:, q_dim:]
            y = np.asarray(data["env"], dtype=float)
            g = np.asarray(data["grad"], dtype=float).T
        if g.shape != q.shape:
            raise ValueError(f"{path} grad must have shape {q.shape}, got {g.shape}")

        theta = x0
        theta_dim = x0_dim
        adaptive = bool(np.asarray(data["adaptive"]).item()) if "adaptive" in data else False

        # get_data.jl only records rho_initial for fixed-step-size runs.
        rho_initial = float(data["rho_initial"]) if "rho_initial" in data else None

        metadata = {
            "nx": nx,
            "nu": nu,
            "horizon": horizon,
            "q_dim": q_dim,
            "theta_dim": theta_dim,
            "x0_dim": x0_dim,
            "rho_initial": rho_initial,
            "parameter": "x0",
            "adaptive_data": adaptive,
        }

        w = np.ones(q.shape[0])
        return q, theta, y, g, w, metadata


def fit_normalization(
    q: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    normalize: bool = normalize_data,
) -> dict[str, np.ndarray | float]:
    """fit normalization constants from training data only."""
    q_mean = np.zeros(q.shape[1])
    q_std = np.ones(q.shape[1])
    theta_mean = np.zeros(theta.shape[1])
    theta_std = np.ones(theta.shape[1])
    env_scale = 1.0

    if normalize:
        q_mean = q.mean(axis=0)
        q_std = np.where(q.std(axis=0) < normalization_eps, 1.0, q.std(axis=0))
        theta_mean = theta.mean(axis=0)
        theta_std = np.where(
            theta.std(axis=0) < normalization_eps,
            1.0,
            theta.std(axis=0),
        )

        env_scale = float(y.std())
        if env_scale < normalization_eps:
            env_scale = 1.0

    if normalize_env and not normalize:
        env_scale = float(y.std())
        if env_scale < normalization_eps:
            env_scale = 1.0

    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "env_scale": env_scale,
    }


def apply_normalization(
    q: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    normalization: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """normalize data consistently with q_norm=(q-mean)/std."""
    q_mean = np.asarray(normalization["q_mean"])
    q_std = np.asarray(normalization["q_std"])
    theta_mean = np.asarray(normalization["theta_mean"])
    theta_std = np.asarray(normalization["theta_std"])
    env_scale = float(normalization["env_scale"])

    q_norm = (q - q_mean) / q_std
    theta_norm = (theta - theta_mean) / theta_std
    y_norm = y / env_scale
    g_norm = g * q_std / env_scale
    return q_norm, theta_norm, y_norm, g_norm


def prediction_matrices(a_matrix: np.ndarray, b_matrix: np.ndarray, horizon: int):
    """build stacked single-shooting prediction matrices."""
    nx = a_matrix.shape[0]
    nu = b_matrix.shape[1]
    a_ro = np.zeros((horizon * nx, nx))
    b_ro = np.zeros((horizon * nx, horizon * nu))

    for k in range(1, horizon + 1):
        row = slice((k - 1) * nx, k * nx)
        a_ro[row, :] = np.linalg.matrix_power(a_matrix, k)
        for j in range(1, k + 1):
            col = slice((j - 1) * nu, j * nu)
            b_ro[row, col] = np.linalg.matrix_power(a_matrix, k - j) @ b_matrix

    return a_ro, b_ro


def make_feasibility_data(metadata, normalization):
    """build data for relu(g_matrix u_hat - b(x0)) feasibility penalty."""
    a_matrix = np.array([[2.0, -1.0], [1.0, 0.2]], dtype=float)
    b_matrix = np.array([[1.0], [0.0]], dtype=float)
    xmin = -5.0
    xmax = 5.0
    umin = -1.0
    umax = 1.0

    nx = metadata["nx"]
    nu = metadata["nu"]
    horizon = metadata["horizon"]
    a_ro, b_ro = prediction_matrices(a_matrix, b_matrix, horizon)
    identity = np.eye(nu * horizon)
    zeros_theta = np.zeros((nu * horizon, nx))

    g_matrix = np.vstack((b_ro, -b_ro, identity, -identity))
    b_offset = np.concatenate(
        (
            np.full(nx * horizon, xmax),
            np.full(nx * horizon, -xmin),
            np.full(nu * horizon, umax),
            np.full(nu * horizon, -umin),
        )
    )
    b_theta = np.vstack((-a_ro, a_ro, zeros_theta, zeros_theta))

    return {
        "weight": feasibility_weight,
        "gamma": ENVELOPE_GAMMA,
        "gamma_feature": "none",
        "x0_dim": metadata["x0_dim"],
        "g_matrix": g_matrix,
        "b_offset": b_offset,
        "b_theta": b_theta,
        "q_mean": normalization["q_mean"],
        "q_std": normalization["q_std"],
        "theta_mean": normalization["theta_mean"],
        "theta_std": normalization["theta_std"],
        "env_scale": normalization["env_scale"],
    }


def evaluate(params, qva, thetava, yva, gva, normalization):
    """compute held-out value and q-gradient mses in original mpc units."""
    qva_norm, thetava_norm, _, _ = apply_normalization(
        qva,
        thetava,
        yva,
        gva,
        normalization,
    )
    q_std = np.asarray(normalization["q_std"])
    env_scale = float(normalization["env_scale"])

    y_pred_norm, g_pred_norm = value_and_grad(
        params,
        jnp.asarray(qva_norm),
        jnp.asarray(thetava_norm),
    )
    y_pred = y_pred_norm * env_scale
    g_pred = g_pred_norm * env_scale / jnp.asarray(q_std)

    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    return val_mse, grad_mse


def evaluate_normalized(params, qva_norm, thetava_norm, yva_norm, gva_norm):
    """compute held-out value and q-gradient mses in normalized units."""
    y_pred_norm, g_pred_norm = value_and_grad(
        params,
        jnp.asarray(qva_norm),
        jnp.asarray(thetava_norm),
    )
    val_mse = jnp.mean((y_pred_norm - jnp.asarray(yva_norm)) ** 2)
    grad_mse = jnp.mean(
        jnp.sum((g_pred_norm - jnp.asarray(gva_norm)) ** 2, axis=1)
    )
    return val_mse, grad_mse


def prepare_for_export(params, metadata, normalization):
    """attach mpc metadata and normalization to the trained model."""
    params = dict(params)
    params["model_type"] = "lpcf"
    params["target"] = "moreau_envelope"
    params["flatten_order"] = "all_w_all_v_all_omega"
    params["output_activation"] = params.get("output_activation", "squared_relu")
    params["parameter"] = metadata["parameter"]
    params["gamma_feature"] = "none"
    params["envelope_gamma"] = ENVELOPE_GAMMA
    params["rho"] = ENVELOPE_GAMMA
    params["rho_initial"] = metadata["rho_initial"]
    params["mpc"] = {
        "nx": metadata["nx"],
        "nu": metadata["nu"],
        "horizon": metadata["horizon"],
        "q_dim": metadata["q_dim"],
        "theta_dim": metadata["theta_dim"],
        "x0_dim": metadata["x0_dim"],
    }
    params["adaptive_data"] = metadata["adaptive_data"]
    params["normalization"] = normalization
    params["feasibility_weight"] = feasibility_weight
    return params


def save_model(params, path: Path) -> None:
    """save the trained lpcf in pickle and json formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(f"{path}.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(Path(f"{path}.json"), "w") as f:
        json.dump(to_serializable(params), f)


if __name__ == "__main__":
    print(jax.devices())

    qtr, thetatr, ytr, gtr, wtr, train_metadata = load_dataset(train_data_path)
    qva, thetava, yva, gva, wva, test_metadata = load_dataset(test_data_path)

    print(f"number of data: {qtr.shape[0]}")
    print(f"q dimension: {train_metadata['q_dim']}")
    print(
        f"theta dimension: {train_metadata['theta_dim']} "
        f"({train_metadata['parameter']})"
    )
    print(f"envelope gamma: {ENVELOPE_GAMMA:.4e}")
    print("gamma feature: none")
    print(f"adaptive data flag: {train_metadata['adaptive_data']}")
    print(f"normalize data: {normalize_data}")
    print(f"normalize env/grad: {normalize_env}")
    print(f"feasibility weight: {feasibility_weight}")
    print(f"learning rate: {learning_rate}")
    print(f"lr decay rate: {lr_decay_rate}")
    print(f"value weight: {value_weight}")
    print(f"grad weight: {grad_weight}")
    print(f"selection metric: {selection_metric}")
    print(f"eval interval: {eval_interval}")
    print(f"seed: {seed}")

    normalization = fit_normalization(qtr, thetatr, ytr, normalize_data)
    qtr_norm, thetatr_norm, ytr_norm, gtr_norm = apply_normalization(
        qtr,
        thetatr,
        ytr,
        gtr,
        normalization,
    )
    qva_norm, thetava_norm, yva_norm, gva_norm = apply_normalization(
        qva,
        thetava,
        yva,
        gva,
        normalization,
    )
    feasibility_data = make_feasibility_data(train_metadata, normalization)

    print(f"================ seed {seed} ================")
    params = init_lpcf_params(
        jax.random.PRNGKey(seed),
        q_dim=train_metadata["q_dim"],
        theta_dim=train_metadata["theta_dim"],
        convex_widths=convex_widths,
        hyper_widths=hyper_widths,
    )
    params = train_lpcf(
        q=qtr_norm,
        theta=thetatr_norm,
        y=ytr_norm,
        g=gtr_norm,
        q_dim=train_metadata["q_dim"],
        theta_dim=train_metadata["theta_dim"],
        w=wtr,
        convex_widths=convex_widths,
        hyper_widths=hyper_widths,
        lr=learning_rate,
        lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps,
        selection_metric=selection_metric,
        grad_weight=grad_weight,
        value_weight=value_weight,
        l2_reg=l2_reg,
        batch_size=batch_size,
        epochs=epochs,
        eval_interval=eval_interval,
        q_val=qva_norm,
        theta_val=thetava_norm,
        y_val=yva_norm,
        g_val=gva_norm,
        w_val=wva,
        seed=seed,
        feasibility_data=feasibility_data,
    )
    val_obj, (val_vm, val_gm, val_fm) = loss_fn(
        params,
        jnp.asarray(qva_norm, dtype=jnp.float32),
        jnp.asarray(thetava_norm, dtype=jnp.float32),
        jnp.asarray(yva_norm, dtype=jnp.float32),
        jnp.asarray(gva_norm, dtype=jnp.float32),
        jnp.asarray(wva, dtype=jnp.float32),
        grad_weight,
        value_weight,
        feasibility_data,
        l2_reg,
    )
    print(
        f"validation obj: {val_obj:.4e} "
        f"| value mse: {val_vm:.4e} "
        f"| grad mse: {val_gm:.4e} "
        f"| feasibility mse: {val_fm:.4e}"
    )

    if normalize_data:
        val_mse_norm, grad_mse_norm = evaluate_normalized(
            params,
            qva_norm,
            thetava_norm,
            yva_norm,
            gva_norm,
        )
        print(
            f"[test normalized] value mse: {val_mse_norm:.4e} "
            f"| grad mse: {grad_mse_norm:.4e}"
        )

    val_mse, grad_mse = evaluate(params, qva, thetava, yva, gva, normalization)
    print(f"[test original] value mse: {val_mse:.4e} | grad mse: {grad_mse:.4e}")

    save_model(
        prepare_for_export(params, train_metadata, normalization),
        model_path,
    )
