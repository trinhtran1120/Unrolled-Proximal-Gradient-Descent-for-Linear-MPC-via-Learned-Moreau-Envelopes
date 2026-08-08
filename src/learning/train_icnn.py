"""Train and export the linear-MPC Moreau-envelope ICNN model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"

TRAIN_DATA_PATH = DATA_DIR / "PGM-rho=0.1_nx=2_N=10-train.npz"
TEST_DATA_PATH = DATA_DIR / "PGM-rho=0.1_nx=2_N=10-test.npz"
MODEL_PATH = MODEL_DIR / "linear-mpc-icnn-rho=0.1-moreau-conditional"

WIDTHS = [32, 32]
LEARNING_RATE = 1e-3
GRAD_WEIGHT = 5.0
L2_REG = 0.0
BATCH_SIZE = 64
EPOCHS = 5000
SEED = 0
NORMALIZATION_EPS = 1e-8
SCALE_DATA = True
TARGET_MODE = "moreau_envelope"
GAMMA_FEATURE = "gamma"
WEIGHT_MODE = "gamma_squared"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from icnn import (
    act_p,
    batched_forward,
    batched_grad_wrt_x,
    to_serializable,
    train_icnn,
)


def load_dataset(path: Path):
    """Load Moreau-envelope data with the requested supervised target."""
    with np.load(path) as data:
        X = data["input"].T
        env = data["env"]
        env_grad = data["grad"].T

        if "gamma" not in data:
            raise KeyError(
                f"{path} does not contain gamma; regenerate the dataset with "
                "adaptive PGM logging before training this model."
            )

        gamma = np.asarray(data["gamma"], dtype=float).reshape(-1, 1)
        if gamma.shape[0] != X.shape[0]:
            raise ValueError(
                "gamma must contain one value per sample: "
                f"got {gamma.shape[0]} gamma values for {X.shape[0]} samples"
            )
        if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0.0):
            raise ValueError("gamma values must be finite and strictly positive")

        rho_key = "rho_initial" if "rho_initial" in data else "rho"
        adaptive = bool(np.asarray(data["adaptive"]).item()) if "adaptive" in data else False
        metadata = {
            "rho_initial": float(data[rho_key]),
            "rho_default": float(np.median(gamma)),
            "adaptive": adaptive,
            "gamma_min": float(np.min(gamma)),
            "gamma_median": float(np.median(gamma)),
            "gamma_mean": float(np.mean(gamma)),
            "gamma_max": float(np.max(gamma)),
        }
        if TARGET_MODE == "half_squared_distance":
            y = gamma[:, 0] * env
            g = gamma * env_grad
        elif TARGET_MODE == "moreau_envelope":
            X = np.hstack([X, gamma])
            y = env
            g = env_grad
        else:
            raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")

        if WEIGHT_MODE == "none":
            w = np.ones(X.shape[0])
        elif WEIGHT_MODE == "gamma":
            w = gamma[:, 0]
        elif WEIGHT_MODE == "gamma_squared":
            w = gamma[:, 0] ** 2
        else:
            raise ValueError(f"Unsupported WEIGHT_MODE: {WEIGHT_MODE}")

        metadata["weight_mode"] = WEIGHT_MODE
        metadata["weight_min"] = float(np.min(w))
        metadata["weight_mean"] = float(np.mean(w))
        metadata["weight_max"] = float(np.max(w))

        return X, y, g, w, metadata


def fit_normalization(
    X: np.ndarray,
    y: np.ndarray,
    scale_data: bool = SCALE_DATA,
) -> dict[str, np.ndarray | float]:
    """Fit normalization constants from training data only."""
    if not scale_data:
        return {
            "input_mean": np.zeros(X.shape[1]),
            "input_std": np.ones(X.shape[1]),
            "env_scale": 1.0,
        }

    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std = np.where(input_std < NORMALIZATION_EPS, 1.0, input_std)

    # Keep the envelope target nonnegative so it matches the ICNN output activation.
    env_scale = float(y.std())
    if env_scale < NORMALIZATION_EPS:
        env_scale = 1.0

    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "env_scale": env_scale,
    }


def apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    normalization: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize value and gradient labels consistently with x_norm = (x - mean) / std."""
    input_mean = np.asarray(normalization["input_mean"])
    input_std = np.asarray(normalization["input_std"])
    env_scale = float(normalization["env_scale"])

    X_norm = (X - input_mean) / input_std
    y_norm = y / env_scale
    g_norm = g * input_std[: g.shape[1]] / env_scale
    return X_norm, y_norm, g_norm


def evaluate_normalized(params, Xva, yva, gva):
    """Compute value and gradient MSEs in normalized training units."""
    y_pred = batched_forward(params, jnp.asarray(Xva))
    g_pred = batched_grad_wrt_x(params, jnp.asarray(Xva))[:, : gva.shape[1]]

    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    return val_mse, grad_mse


def evaluate(params, Xva, yva, gva, normalization):
    """Compute held-out value and gradient MSEs in original MPC units."""
    Xva_norm, _, _ = apply_normalization(Xva, yva, gva, normalization)
    input_std = np.asarray(normalization["input_std"])
    env_scale = float(normalization["env_scale"])

    y_pred_norm = batched_forward(params, jnp.asarray(Xva_norm))
    g_pred_norm = batched_grad_wrt_x(params, jnp.asarray(Xva_norm))[:, : gva.shape[1]]

    y_pred = y_pred_norm * env_scale
    g_pred = g_pred_norm * env_scale / jnp.asarray(input_std[: gva.shape[1]])

    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    return val_mse, grad_mse


def prepare_for_export(params, metadata, normalization):
    """Apply nonnegative ICNN projections expected by the Julia model loader."""
    params["rho"] = metadata["rho_default"]
    params["rho_initial"] = metadata["rho_initial"]
    params["adaptive_data"] = metadata["adaptive"]
    params["gamma_feature"] = GAMMA_FEATURE if TARGET_MODE == "moreau_envelope" else "none"
    params["gamma_stats"] = {
        "min": metadata["gamma_min"],
        "median": metadata["gamma_median"],
        "mean": metadata["gamma_mean"],
        "max": metadata["gamma_max"],
    }
    params["target"] = TARGET_MODE
    params["weight_mode"] = metadata["weight_mode"]
    params["normalization"] = normalization
    params["v"] = act_p(params["v"])

    for i in range(1, len(params["W"])):
        params["W"][i] = act_p(params["W"][i])

    return params


def save_model(params, path: Path) -> None:
    """Save the trained ICNN in both pickle and JSON formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle_path = Path(f"{path}.pkl")
    json_path = Path(f"{path}.json")

    with open(pickle_path, "wb") as f:
        pickle.dump(params, f)

    with open(json_path, "w") as f:
        json.dump(to_serializable(params), f)


if __name__ == "__main__":
    print(jax.devices())

    Xtr, ytr, gtr, wtr, train_metadata = load_dataset(TRAIN_DATA_PATH)
    Xva, yva, gva, wva, test_metadata = load_dataset(TEST_DATA_PATH)

    n_features, n_samples = Xtr.shape[1], Xtr.shape[0]
    print(f"Number of data: {n_samples}")
    print(f"Input dimension: {n_features}")
    print(f"Gradient target dimension: {gtr.shape[1]}")
    print(f"Target: {TARGET_MODE}")
    print(f"Gamma feature: {GAMMA_FEATURE if TARGET_MODE == 'moreau_envelope' else 'none'}")
    print(f"Weight mode: {WEIGHT_MODE}")
    print(f"Adaptive data: {train_metadata['adaptive']}")
    print(f"rho initial: {train_metadata['rho_initial']:.4e}")
    print(
        "gamma train min/median/mean/max: "
        f"{train_metadata['gamma_min']:.4e} / "
        f"{train_metadata['gamma_median']:.4e} / "
        f"{train_metadata['gamma_mean']:.4e} / "
        f"{train_metadata['gamma_max']:.4e}"
    )
    print(
        "gamma test  min/median/mean/max: "
        f"{test_metadata['gamma_min']:.4e} / "
        f"{test_metadata['gamma_median']:.4e} / "
        f"{test_metadata['gamma_mean']:.4e} / "
        f"{test_metadata['gamma_max']:.4e}"
    )
    print(f"Export default rho: {train_metadata['rho_default']:.4e}")
    print(
        "weight train min/mean/max: "
        f"{train_metadata['weight_min']:.4e} / "
        f"{train_metadata['weight_mean']:.4e} / "
        f"{train_metadata['weight_max']:.4e}"
    )

    print(f"Scale data: {SCALE_DATA}")

    normalization = fit_normalization(Xtr, ytr, SCALE_DATA)
    Xtr_norm, ytr_norm, gtr_norm = apply_normalization(Xtr, ytr, gtr, normalization)
    Xva_norm, yva_norm, gva_norm = apply_normalization(Xva, yva, gva, normalization)

    params = train_icnn(
        Xtr_norm,
        ytr_norm,
        gtr_norm,
        n_in=n_features,
        w=wtr,
        widths=WIDTHS,
        lr=LEARNING_RATE,
        grad_weight=GRAD_WEIGHT,
        l2_reg=L2_REG,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED,
        X_val=Xva_norm,
        y_val=yva_norm,
        g_val=gva_norm,
        w_val=wva,
    )

    norm_val_mse, norm_grad_mse = evaluate_normalized(params, Xva_norm, yva_norm, gva_norm)
    val_mse, grad_mse = evaluate(params, Xva, yva, gva, normalization)
    print(
        f"[TEST {TARGET_MODE}] normalized value MSE: {norm_val_mse:.4e} "
        f"| grad MSE: {norm_grad_mse:.4e}"
    )
    print(
        f"[TEST {TARGET_MODE} original] value MSE: {val_mse:.4e} "
        f"| grad MSE: {grad_mse:.4e}"
    )

    save_model(prepare_for_export(params, train_metadata, normalization), MODEL_PATH)
