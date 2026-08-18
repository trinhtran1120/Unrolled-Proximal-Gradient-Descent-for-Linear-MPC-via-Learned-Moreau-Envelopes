"""train and export the linear-mpc pcf model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parents[1]
data_dir = project_dir / "data"
model_dir = project_dir / "model"

train_data_path = data_dir / "PGM-rho=0.001_nx=2_N=10-train_fix.npz"
test_data_path = data_dir / "PGM-rho=0.001_nx=2_N=10-test_fix.npz"
model_path = model_dir / "linear-mpc-lpcf001-fix"

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
seed = 0
normalization_eps = 1e-8
normalize_data = False
normalize_gamma = False
normalize_env = True
adaptive_step_size = False

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from pcf import (
    init_lpcf_params,
    loss_fn,
    train_lpcf,
    to_serializable,
    value_and_grad,
)


def _require_fixed_gamma(path: Path, gamma: np.ndarray) -> float:
    """return fixed gamma or raise if the dataset is not fixed-gamma."""
    if gamma.size == 0:
        raise ValueError(f"{path} contains an empty gamma array")

    gamma_ref = float(gamma[0])
    if not np.allclose(gamma, gamma_ref, rtol=1e-8, atol=1e-12):
        raise ValueError(
            f"{path} contains variable gamma values "
            f"(min={np.min(gamma):.4e}, max={np.max(gamma):.4e}). "
            "for the fixed-gamma lpcf experiment, regenerate data with a fixed "
            "pgm step size or set adaptive_step_size=true knowingly."
        )
    return gamma_ref


def load_dataset(path: Path):
    """load mpc moreau-envelope data and split input into q and theta."""
    with np.load(path) as data:
        nx = int(data["nx"])
        nu = int(data["nu"])
        horizon = int(data["N"])
        q_dim = nu * horizon
        x0_dim = nx

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

        gamma = np.asarray(data["gamma"], dtype=float).reshape(-1)
        if gamma.shape[0] != q.shape[0]:
            raise ValueError(
                f"{path} gamma must contain one value per sample: "
                f"got {gamma.shape[0]} for {q.shape[0]} samples"
            )
        if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0.0):
            raise ValueError("gamma values must be finite and strictly positive")

        if adaptive_step_size:
            gamma_fixed = float(np.median(gamma))
            theta = np.hstack((x0, gamma[:, None]))
            theta_dim = x0_dim + 1
        else:
            gamma_fixed = _require_fixed_gamma(path, gamma)
            theta = x0
            theta_dim = x0_dim

        adaptive = bool(np.asarray(data["adaptive"]).item()) if "adaptive" in data else False
        metadata = {
            "nx": nx,
            "nu": nu,
            "horizon": horizon,
            "q_dim": q_dim,
            "theta_dim": theta_dim,
            "x0_dim": x0_dim,
            "rho_initial": float(data["rho_initial"]),
            "gamma_fixed": gamma_fixed,
            "gamma_feature": "gamma" if adaptive_step_size else "none",
            "parameter": "x0_gamma" if adaptive_step_size else "x0",
            "adaptive_data": adaptive,
            "gamma_min": float(np.min(gamma)),
            "gamma_median": float(np.median(gamma)),
            "gamma_mean": float(np.mean(gamma)),
            "gamma_max": float(np.max(gamma)),
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
    elif normalize_gamma and adaptive_step_size:
        gamma_std = float(theta[:, -1].std())
        theta_mean[-1] = float(theta[:, -1].mean())
        theta_std[-1] = 1.0 if gamma_std < normalization_eps else gamma_std

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
        "gamma": metadata["gamma_fixed"],
        "gamma_feature": metadata["gamma_feature"],
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
    params["output_activation"] = params.get("output_activation", "softplus")
    params["gamma_feature"] = metadata["gamma_feature"]
    params["parameter"] = metadata["parameter"]
    params["rho"] = metadata["gamma_fixed"]
    params["rho_initial"] = metadata["rho_initial"]
    params["mpc"] = {
        "nx": metadata["nx"],
        "nu": metadata["nu"],
        "horizon": metadata["horizon"],
        "q_dim": metadata["q_dim"],
        "theta_dim": metadata["theta_dim"],
        "x0_dim": metadata["x0_dim"],
    }
    params["gamma_stats"] = {
        "fixed": metadata["gamma_fixed"],
        "min": metadata["gamma_min"],
        "median": metadata["gamma_median"],
        "mean": metadata["gamma_mean"],
        "max": metadata["gamma_max"],
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
    print(f"default gamma: {train_metadata['gamma_fixed']:.4e}")
    print(f"gamma feature: {train_metadata['gamma_feature']}")
    print(f"adaptive data flag: {train_metadata['adaptive_data']}")
    print(
        "gamma train min/median/mean/max: "
        f"{train_metadata['gamma_min']:.4e} / "
        f"{train_metadata['gamma_median']:.4e} / "
        f"{train_metadata['gamma_mean']:.4e} / "
        f"{train_metadata['gamma_max']:.4e}"
    )
    print(f"normalize data: {normalize_data}")
    print(f"normalize gamma: {normalize_gamma}")
    print(f"normalize env/grad: {normalize_env}")
    print(f"feasibility weight: {feasibility_weight}")
    print(f"learning rate: {learning_rate}")
    print(f"lr decay rate: {lr_decay_rate}")
    print(f"value weight: {value_weight}")
    print(f"grad weight: {grad_weight}")
    print(f"selection metric: {selection_metric}")
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

    # save_model(
    #     prepare_for_export(params, train_metadata, normalization),
    #     model_path,
    # )
