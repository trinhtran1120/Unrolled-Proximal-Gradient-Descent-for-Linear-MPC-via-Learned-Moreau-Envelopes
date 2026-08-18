"""Tests for the MPC-specific LPCF/PCF model."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pcf import (  # noqa: E402
    MpcPcf,
    init_lpcf_params,
    loss_fn,
    psi_flat_function,
    unpack_lpcf_tensors,
)


def make_dataset(n_samples: int = 16, q_dim: int = 3, theta_dim: int = 2):
    """Create a tiny convex regression target with an analytic q-gradient."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n_samples, q_dim)).astype(np.float32)
    theta = rng.normal(size=(n_samples, theta_dim)).astype(np.float32)
    shifted = q + 0.1 * theta[:, :1]
    y = (0.5 * np.sum(shifted**2, axis=1) + 0.1).astype(np.float32)
    g = shifted.astype(np.float32)
    w = np.ones(n_samples, dtype=np.float32)
    return q, theta, y, g, w


def test_mpc_pcf_initialization():
    """MpcPcf initializes static metadata and hypernetwork parameters."""
    model = MpcPcf(q_dim=3, theta_dim=2)

    assert model.q_dim == 3
    assert model.theta_dim == 2
    assert model.convex_widths == (32, 32)
    assert model.hyper_widths == (64, 64)
    assert model.seed == 0
    assert model.params["q_dim"] == 3
    assert model.params["theta_dim"] == 2
    assert model.params["layer_dims"] == (3, 32, 32, 1)
    assert set(model.params["hyper"].keys()) == {"W_psi", "V_psi", "omega_psi"}

    custom = MpcPcf(
        q_dim=4,
        theta_dim=3,
        convex_widths=[5, 6],
        hyper_widths=[7],
        seed=11,
    )
    assert custom.convex_widths == (5, 6)
    assert custom.hyper_widths == (7,)
    assert custom.params["layer_dims"] == (4, 5, 6, 1)


def test_mpc_pcf_initialization_validation():
    """Invalid dimensions and widths fail early."""
    with pytest.raises(ValueError):
        MpcPcf(q_dim=0, theta_dim=2)
    with pytest.raises(ValueError):
        MpcPcf(q_dim=3, theta_dim=0)
    with pytest.raises(ValueError):
        MpcPcf(q_dim=3, theta_dim=2, convex_widths=[])
    with pytest.raises(ValueError):
        MpcPcf(q_dim=3, theta_dim=2, hyper_widths=[8, 0])


def test_psi_flat_and_unpack_shapes():
    """psi(theta) emits all W sections, then all V sections, then all offsets."""
    params = init_lpcf_params(
        jax.random.PRNGKey(0),
        q_dim=3,
        theta_dim=2,
        convex_widths=[4, 5],
        hyper_widths=[6],
    )

    theta = jnp.asarray([0.2, -0.5], dtype=jnp.float32)
    emitted = psi_flat_function(theta, params["hyper"])
    assert emitted.shape == (params["shapes"][0][0] * params["shapes"][0][1]
                             + params["shapes"][1][0] * params["shapes"][1][1]
                             + sum(np.prod(shape) for shape in params["shapes"][2:]),)

    tensors = unpack_lpcf_tensors(params, emitted)
    assert [w.shape for w in tensors["W"]] == [(5, 4), (1, 5)]
    assert [v.shape for v in tensors["V"]] == [(4, 3), (5, 3), (1, 3)]
    assert [o.shape for o in tensors["omega"]] == [(4,), (5,), (1,)]


def test_value_grad_predict_and_loss_shapes():
    """Forward, q-gradient, predict, and loss expose stable batch shapes."""
    q, theta, y, g, w = make_dataset(n_samples=10, q_dim=3, theta_dim=2)
    model = MpcPcf(q_dim=3, theta_dim=2, convex_widths=[4, 4], hyper_widths=[5])

    values, grads = model.value_and_grad(q, theta)
    assert values.shape == (10,)
    assert grads.shape == q.shape
    assert model.predict(q, theta).shape == (10,)
    assert model.grad(q, theta).shape == q.shape
    assert np.all(np.asarray(values) > 0.0)

    objective, parts = model.loss(q, theta, y, g, w, grad_weight=2.0)
    value_mse, grad_mse, feasibility_mse = parts
    assert objective.shape == ()
    assert value_mse.shape == ()
    assert grad_mse.shape == ()
    assert feasibility_mse.shape == ()


def test_loss_rejects_bad_shapes():
    """Training-array validation catches inconsistent inputs."""
    q, theta, y, g, w = make_dataset(n_samples=8, q_dim=3, theta_dim=2)
    params = init_lpcf_params(
        jax.random.PRNGKey(0),
        q_dim=3,
        theta_dim=2,
        convex_widths=[4],
        hyper_widths=[5],
    )

    with pytest.raises(ValueError):
        model = MpcPcf(q_dim=3, theta_dim=2, convex_widths=[4], hyper_widths=[5])
        model.fit(q[:, :2], theta, y, g, w, epochs=0)

    objective, parts = loss_fn(
        params,
        jnp.asarray(q),
        jnp.asarray(theta),
        jnp.asarray(y),
        jnp.asarray(g),
        jnp.asarray(w),
    )
    assert np.isfinite(float(objective))
    assert all(np.isfinite(float(part)) for part in parts)


def test_fit_smoke_and_diagnostics(capsys):
    """A short fit runs end-to-end and diagnostics report both networks."""
    q, theta, y, g, w = make_dataset(n_samples=12, q_dim=3, theta_dim=2)
    model = MpcPcf(q_dim=3, theta_dim=2, convex_widths=[4], hyper_widths=[5], seed=3)
    before = model.to_params()["hyper"]["V_psi"][0].copy()

    model.fit(
        q,
        theta,
        y,
        g,
        w,
        lr=1e-3,
        lr_decay_rate=1.0,
        grad_weight=1.0,
        value_weight=1.0,
        batch_size=6,
        epochs=2,
    )

    after = model.to_params()["hyper"]["V_psi"][0]
    assert not np.allclose(np.asarray(before), np.asarray(after))

    model.print_diagnostics(q, theta, y, g, max_samples=4, label="pytest")
    captured = capsys.readouterr()
    assert "[pytest] diagnostics" in captured.out
    assert "[psi]" in captured.out
    assert "[main icnn]" in captured.out
