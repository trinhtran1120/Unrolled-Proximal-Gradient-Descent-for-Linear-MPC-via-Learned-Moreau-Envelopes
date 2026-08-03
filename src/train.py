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
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"

TRAIN_DATA_PATH = DATA_DIR / "PGM-rho=0.001-train.npz"
TEST_DATA_PATH = DATA_DIR / "PGM-rho=0.001-test.npz"
MODEL_PATH = MODEL_DIR / "linear-mpc-icnn-rho=0.001"

WIDTHS = [16, 16]
LEARNING_RATE = 1e-3
GRAD_WEIGHT = 5.0
L2_REG = 0.0
BATCH_SIZE = 32
EPOCHS = 2000
SEED = 0
NORMALIZATION_EPS = 1e-8

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ICNN import (
    act_p,
    batched_forward,
    batched_grad_wrt_x,
    to_serializable,
    train_icnn,
)


def load_dataset(path: Path):
    """Load a linear-MPC Moreau-envelope dataset in learner-friendly orientation."""
    with np.load(path) as data:
        return (
            data["input"].T,
            data["env"],
            data["grad"].T,
            float(data["rho"]),
        )


def fit_normalization(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit normalization constants from training data only."""
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


def prepare_for_export(params, rho, normalization):
    """Apply nonnegative ICNN projections expected by the Julia model loader."""
    params["rho"] = rho
    params["normalization"] = normalization
    params["v"] = act_p(params["v"])

    for i in range(1, len(params["W"])):
        params["W"][i] = act_p(params["W"][i])

    return params


def save_model(params, path: Path) -> None:
    """Save the trained ICNN in both pickle and JSON formats."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(to_serializable(params), f)


if __name__ == "__main__":
    print(jax.devices())

    Xtr, ytr, gtr, rho = load_dataset(TRAIN_DATA_PATH)
    Xva, yva, gva, _ = load_dataset(TEST_DATA_PATH)

    n_features, n_samples = Xtr.shape[1], Xtr.shape[0]
    print(f"Number of data: {n_samples}")
    print(f"Input dimension: {n_features}")
    print(f"Gradient target dimension: {gtr.shape[1]}")

    normalization = fit_normalization(Xtr, ytr)
    Xtr_norm, ytr_norm, gtr_norm = apply_normalization(Xtr, ytr, gtr, normalization)
    Xva_norm, yva_norm, gva_norm = apply_normalization(Xva, yva, gva, normalization)

    params = train_icnn(
        Xtr_norm,
        ytr_norm,
        gtr_norm,
        n_in=n_features,
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
    )

    norm_val_mse, norm_grad_mse = evaluate_normalized(params, Xva_norm, yva_norm, gva_norm)
    val_mse, grad_mse = evaluate(params, Xva, yva, gva, normalization)
    print(
        f"[TEST normalized] value MSE: {norm_val_mse:.4e} "
        f"| grad MSE: {norm_grad_mse:.4e}"
    )
    print(f"[TEST original] value MSE: {val_mse:.4e} | grad MSE: {grad_mse:.4e}")

    save_model(prepare_for_export(params, rho, normalization), MODEL_PATH)
