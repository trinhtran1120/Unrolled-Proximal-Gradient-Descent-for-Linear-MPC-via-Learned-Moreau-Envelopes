"""mpc-specific lpcf model with autodiff gradient in the convex variable."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils_sc import (
    act_p,
    activation,
    as_training_arrays,
    batch_iterator,
    copy_params,
    default_convex_widths,
    default_hyper_widths,
    feasibility_weight,
    glorot_uniform,
    gradient_loss_scale,
    init_scale,
    params_t,
    soft_feasibility_penalty,
    validate_widths,
)


def _lpcf_shapes(
    q_dim: int,
    convex_widths: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int]:
    """plan flat hypernetwork output shapes for lpcf tensors."""
    if q_dim <= 0:
        raise ValueError("q_dim must be positive")

    widths = validate_widths(convex_widths, "convex_widths")
    layer_dims = (q_dim, *widths, 1)
    hidden_dims = layer_dims[1:-1]
    shapes: list[tuple[int, ...]] = []

    for layer_idx, hidden_dim in enumerate(hidden_dims):
        if layer_idx > 0:
            shapes.append((hidden_dim, hidden_dims[layer_idx - 1]))
        shapes.append((hidden_dim, q_dim))
        shapes.append((hidden_dim,))

    shapes.append((1, hidden_dims[-1]))
    shapes.append((1, q_dim))
    shapes.append((1,))

    emit_dim = int(sum(np.prod(shape) for shape in shapes))
    return tuple(shapes), layer_dims, emit_dim


def init_psi_params(
    key: jax.Array,
    theta_dim: int,
    hidden_widths: Sequence[int],
    output_dim: int,
) -> params_t:
    """initialize the LPCF psi network weights W_psi, V_psi, omega_psi."""
    if theta_dim <= 0:
        raise ValueError("theta_dim must be positive")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")

    hidden_widths = validate_widths(hidden_widths, "hidden_widths")
    w_psi = (theta_dim, *hidden_widths, output_dim)
    W_count = len(w_psi) - 2
    V_count = len(w_psi) - 1
    keys = jax.random.split(key, W_count + V_count)

    W_psi: list[jnp.ndarray] = []
    V_psi: list[jnp.ndarray] = []
    omega_psi: list[jnp.ndarray] = []

    key_idx = 0
    for layer_idx in range(1, len(w_psi) - 1):
        scale = 1.0 if layer_idx < len(w_psi) - 2 else init_scale
        W_psi.append(
            scale
            * glorot_uniform(
                keys[key_idx],
                (w_psi[layer_idx + 1], w_psi[layer_idx]),
            )
        )
        key_idx += 1

    for layer_idx in range(len(w_psi) - 1):
        scale = 1.0 if layer_idx < len(w_psi) - 2 else init_scale
        V_psi.append(
            scale * glorot_uniform(keys[key_idx], (w_psi[layer_idx + 1], theta_dim))
        )
        omega_psi.append(jnp.zeros((w_psi[layer_idx + 1],)))
        key_idx += 1

    return {"W_psi": W_psi, "V_psi": V_psi, "omega_psi": omega_psi}


def init_lpcf_params(
    key: jax.Array,
    q_dim: int,
    theta_dim: int,
    convex_widths: Sequence[int] = default_convex_widths,
    hyper_widths: Sequence[int] = default_hyper_widths,
) -> params_t:
    """initialize lpcf parameters."""
    if theta_dim <= 0:
        raise ValueError("theta_dim must be positive")

    convex_widths = validate_widths(convex_widths, "convex_widths")
    hyper_widths = validate_widths(hyper_widths, "hyper_widths")
    shapes, layer_dims, emit_dim = _lpcf_shapes(q_dim, convex_widths)

    return {
        "q_dim": int(q_dim),
        "theta_dim": int(theta_dim),
        "convex_widths": convex_widths,
        "hyper_widths": hyper_widths,
        "layer_dims": layer_dims,
        "shapes": shapes,
        "hyper": init_psi_params(key, theta_dim, hyper_widths, emit_dim),
    }

@jax.jit
def psi_flat_function(theta: jnp.ndarray, hyper_params: params_t) -> jnp.ndarray:
    """evaluate psi(theta) as the flat vector emitted by the psi network."""
    W_psi = hyper_params["W_psi"]
    V_psi = hyper_params["V_psi"]
    omega_psi = hyper_params["omega_psi"]
    out = jnp.dot(V_psi[0], theta) + omega_psi[0]
    for layer_idx in range(1, len(V_psi)):
        out = jax.nn.relu(out)
        out = (
            jnp.dot(W_psi[layer_idx - 1], out)
            + jnp.dot(V_psi[layer_idx], theta)
            + omega_psi[layer_idx]
        )
    return out


def generate_psi_flat(hyper_params: params_t):
    """generate the jitted flat psi(theta) evaluator used by exporters."""

    @jax.jit
    def psi_flat(theta: jnp.ndarray) -> jnp.ndarray:
        return psi_flat_function(theta, hyper_params).squeeze()

    return psi_flat


def unpack_lpcf_tensors(params: params_t, emitted: jnp.ndarray) -> params_t:
    """unpack psi(theta)'s flat output into variable-network tensors."""
    offset = 0
    shape_idx = 0
    shapes = params["shapes"]
    convex_widths = params["convex_widths"]
    W: list[jnp.ndarray] = []
    V: list[jnp.ndarray] = []
    omega: list[jnp.ndarray] = []

    def take(shape: tuple[int, ...]) -> jnp.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        value = emitted[offset : offset + size].reshape(shape)
        offset += size
        return value

    for layer_idx in range(len(convex_widths)):
        if layer_idx > 0:
            W.append(take(shapes[shape_idx]))
            shape_idx += 1
        V.append(take(shapes[shape_idx]))
        shape_idx += 1
        omega.append(take(shapes[shape_idx]))
        shape_idx += 1

    W.append(take(shapes[shape_idx]))
    V.append(take(shapes[shape_idx + 1]))
    omega.append(take(shapes[shape_idx + 2]))
    return {
        "W": W,
        "V": V,
        "omega": omega,
    }


def psi_function(theta: jnp.ndarray, params: params_t) -> params_t:
    """evaluate psi(theta) and unpack its output into W, V, and omega."""
    return unpack_lpcf_tensors(params, psi_flat_function(theta, params["hyper"]))


def lpcf_forward(params: params_t, q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """evaluate scalar f(q, theta)."""
    p = psi_function(theta, params)

    z = jnp.dot(p["V"][0], q) + p["omega"][0]
    for layer_idx in range(1, len(p["V"])):
        z = (
            jnp.dot(act_p(p["W"][layer_idx - 1]), activation(z))
            + jnp.dot(p["V"][layer_idx], q)
            + p["omega"][layer_idx]
        )

    return z[0]

def value_and_grad_wrt_q(
    params: params_t,
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """evaluate f and use jax autodiff for df/dq."""
    value = lpcf_forward(params, q, theta)
    grad_q = jax.grad(lpcf_forward, argnums=1)(params, q, theta)
    return value, grad_q

def grad_wrt_q(params: params_t, q: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """return the autodiff gradient with respect to q."""
    return jax.grad(lpcf_forward, argnums=1)(params, q, theta)


batched_forward = jax.vmap(lpcf_forward, in_axes=(None, 0, 0))
batched_value_and_grad_wrt_q = jax.vmap(value_and_grad_wrt_q, in_axes=(None, 0, 0))
batched_grad_wrt_q = jax.vmap(grad_wrt_q, in_axes=(None, 0, 0))


def _with_hyper(params_template: params_t, hyper_params: params_t) -> params_t:
    """combine static lpcf metadata with trainable hypernetwork parameters."""
    params = dict(params_template)
    params["hyper"] = hyper_params
    return params

def value_and_grad_wrt_q_from_hyper(
    params_template: params_t,
    hyper_params: params_t,
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """evaluate value and autodiff q-gradient when only hyper params are dynamic."""
    return value_and_grad_wrt_q(_with_hyper(params_template, hyper_params), q, theta)

def batched_value_and_grad_wrt_q_from_hyper(
    params_template: params_t,
    hyper_params: params_t,
    q: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """evaluate batched values and autodiff q-gradients."""
    return jax.vmap(
        lambda qi, thetai: value_and_grad_wrt_q_from_hyper(
            params_template,
            hyper_params,
            qi,
            thetai,
        )
    )(q, theta)


def make_batched_value_and_grad(params_template: params_t):
    """build a jitted batched value/gradient evaluator."""

    @jax.jit
    def batched(hyper_params: params_t, q: jnp.ndarray, theta: jnp.ndarray):
        return batched_value_and_grad_wrt_q_from_hyper(
            params_template,
            hyper_params,
            q,
            theta,
        )

    return batched

def loss_fn(
    params: params_t,
    qb: jnp.ndarray,
    thetab: jnp.ndarray,
    yb: jnp.ndarray,
    gb: jnp.ndarray,
    wb: jnp.ndarray,
    grad_weight: float = 1.0,
    feasibility_data: params_t | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """return value, gradient, and soft feasibility loss from paper.md."""
    y_pred, g_pred = batched_value_and_grad_wrt_q(params, qb, thetab)
    alpha = wb / jnp.maximum(jnp.mean(wb), 1e-12)
    gradient_scale = gradient_loss_scale(feasibility_data, qb.dtype)
    value_mse = jnp.mean(alpha * (y_pred - yb) ** 2)
    grad_mse = jnp.mean(
        alpha * jnp.sum(((g_pred - gb) * gradient_scale) ** 2, axis=1)
    )
    feasibility_mse = soft_feasibility_penalty(
        qb,
        thetab,
        g_pred,
        alpha,
        feasibility_data,
    )
    feasibility_penalty_weight = feasibility_weight(feasibility_data)
    loss = value_mse + grad_weight * grad_mse + (
        feasibility_penalty_weight * feasibility_mse
    )
    return loss, (value_mse, grad_mse, feasibility_mse)

def loss_fn_from_hyper(
    params_template: params_t,
    hyper_params: params_t,
    qb: jnp.ndarray,
    thetab: jnp.ndarray,
    yb: jnp.ndarray,
    gb: jnp.ndarray,
    wb: jnp.ndarray,
    grad_weight: float = 1.0,
    feasibility_data: params_t | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """return the lpcf training loss using batched hypernetwork evaluation."""
    y_pred, g_pred = batched_value_and_grad_wrt_q_from_hyper(
        params_template,
        hyper_params,
        qb,
        thetab,
    )
    alpha = wb / jnp.maximum(jnp.mean(wb), 1e-12)
    gradient_scale = gradient_loss_scale(feasibility_data, qb.dtype)
    value_mse = jnp.mean(alpha * (y_pred - yb) ** 2)
    grad_mse = jnp.mean(
        alpha * jnp.sum(((g_pred - gb) * gradient_scale) ** 2, axis=1)
    )
    feasibility_mse = soft_feasibility_penalty(
        qb,
        thetab,
        g_pred,
        alpha,
        feasibility_data,
    )
    feasibility_penalty_weight = feasibility_weight(feasibility_data)
    loss = value_mse + grad_weight * grad_mse + (
        feasibility_penalty_weight * feasibility_mse
    )
    return loss, (value_mse, grad_mse, feasibility_mse)


def make_train_step(
    optimizer: optax.GradientTransformation,
    params_template: params_t,
    feasibility_data: params_t | None,
):
    """build one jitted optimizer step."""

    @jax.jit
    def train_step(
        hyper_params: params_t,
        opt_state: optax.OptState,
        qb: jnp.ndarray,
        thetab: jnp.ndarray,
        yb: jnp.ndarray,
        gb: jnp.ndarray,
        wb: jnp.ndarray,
        grad_weight: float,
    ):
        def hyper_loss(hyper_params: params_t):
            return loss_fn_from_hyper(
                params_template,
                hyper_params,
                qb,
                thetab,
                yb,
                gb,
                wb,
                grad_weight,
                feasibility_data,
            )

        (loss, (vmse, gmse, fmse)), grads = jax.value_and_grad(hyper_loss, has_aux=True)(
            hyper_params
        )
        updates, opt_state = optimizer.update(grads, opt_state, hyper_params)
        hyper_params = optax.apply_updates(hyper_params, updates)
        return hyper_params, opt_state, loss, vmse, gmse, fmse

    return train_step


def make_eval_step(
    params_template: params_t,
    feasibility_data: params_t | None,
):
    """build a jitted validation objective for model selection and logging."""
    batched_eval = make_batched_value_and_grad(params_template)

    @jax.jit
    def eval_step(
        hyper_params: params_t,
        q: jnp.ndarray,
        theta: jnp.ndarray,
        y: jnp.ndarray,
        g: jnp.ndarray,
        w: jnp.ndarray,
        grad_weight: float,
    ):
        y_pred, g_pred = batched_eval(hyper_params, q, theta)
        alpha = w / jnp.maximum(jnp.mean(w), 1e-12)
        gradient_scale = gradient_loss_scale(feasibility_data, q.dtype)
        value_mse = jnp.mean(alpha * (y_pred - y) ** 2)
        grad_mse = jnp.mean(
            alpha * jnp.sum(((g_pred - g) * gradient_scale) ** 2, axis=1)
        )
        feasibility_mse = soft_feasibility_penalty(
            q,
            theta,
            g_pred,
            alpha,
            feasibility_data,
        )
        feasibility_penalty_weight = feasibility_weight(feasibility_data)
        selection_obj = value_mse + grad_weight * grad_mse + (
            feasibility_penalty_weight * feasibility_mse
        )
        return selection_obj, value_mse, grad_mse, feasibility_mse

    return eval_step


class MpcPcf:
    """object-oriented wrapper for the mpc-specific lpcf model."""

    def __init__(
        self,
        q_dim: int,
        theta_dim: int,
        convex_widths: Sequence[int] = default_convex_widths,
        hyper_widths: Sequence[int] = default_hyper_widths,
        seed: int = 0,
        params: params_t | None = None,
    ) -> None:
        self.q_dim = int(q_dim)
        self.theta_dim = int(theta_dim)
        self.convex_widths = validate_widths(convex_widths, "convex_widths")
        self.hyper_widths = validate_widths(hyper_widths, "hyper_widths")
        self.seed = int(seed)

        if params is None:
            key = jax.random.PRNGKey(self.seed)
            params = init_lpcf_params(
                key,
                self.q_dim,
                self.theta_dim,
                self.convex_widths,
                self.hyper_widths,
            )
        self.set_params(params)

    def _make_params_template(self) -> params_t:
        params_template = dict(self.params)
        params_template.pop("hyper")
        return params_template

    def set_params(self, params: params_t) -> None:
        """set model parameters and rebuild compiled helpers."""
        self.params = params
        self.params_template = self._make_params_template()
        self.batched_eval = make_batched_value_and_grad(self.params_template)

    def fit(
        self,
        q: np.ndarray,
        theta: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        w: np.ndarray | None = None,
        lr: float = 1e-3,
        grad_weight: float = 1.0,
        l2_reg: float = 0.0,
        batch_size: int = 128,
        epochs: int = 200,
        lr_decay_rate: float = 1.0,
        lr_decay_steps: int | None = None,
        q_val: np.ndarray | None = None,
        theta_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        g_val: np.ndarray | None = None,
        w_val: np.ndarray | None = None,
        feasibility_data: params_t | None = None,
    ) -> "MpcPcf":
        """train the model in place and return self."""
        self.set_params(
            train_lpcf(
                q,
                theta,
                y,
                g,
                self.q_dim,
                self.theta_dim,
                w,
                self.convex_widths,
                self.hyper_widths,
                lr,
                grad_weight,
                l2_reg,
                batch_size,
                epochs,
                lr_decay_rate,
                lr_decay_steps,
                self.seed,
                q_val,
                theta_val,
                y_val,
                g_val,
                w_val,
                feasibility_data,
            )
        )
        return self

    def value_and_grad(
        self,
        q: np.ndarray | jnp.ndarray,
        theta: np.ndarray | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """return batched f(q, theta) and autodiff gradients with respect to q."""
        return self.batched_eval(
            self.params["hyper"],
            jnp.asarray(q, dtype=jnp.float32),
            jnp.asarray(theta, dtype=jnp.float32),
        )

    def predict(
        self,
        q: np.ndarray | jnp.ndarray,
        theta: np.ndarray | jnp.ndarray,
    ) -> jnp.ndarray:
        """return batched scalar predictions f(q, theta)."""
        values, _grads = self.value_and_grad(q, theta)
        return values

    def grad(
        self,
        q: np.ndarray | jnp.ndarray,
        theta: np.ndarray | jnp.ndarray,
    ) -> jnp.ndarray:
        """return batched autodiff gradients with respect to q."""
        _values, grads = self.value_and_grad(q, theta)
        return grads

    def loss(
        self,
        q: np.ndarray | jnp.ndarray,
        theta: np.ndarray | jnp.ndarray,
        y: np.ndarray | jnp.ndarray,
        g: np.ndarray | jnp.ndarray,
        w: np.ndarray | jnp.ndarray | None = None,
        grad_weight: float = 1.0,
        feasibility_data: params_t | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """evaluate the current value, gradient, and feasibility loss."""
        qj = jnp.asarray(q, dtype=jnp.float32)
        if w is None:
            w = np.ones(qj.shape[0])
        return loss_fn_from_hyper(
            self.params_template,
            self.params["hyper"],
            qj,
            jnp.asarray(theta, dtype=jnp.float32),
            jnp.asarray(y, dtype=jnp.float32),
            jnp.asarray(g, dtype=jnp.float32),
            jnp.asarray(w, dtype=jnp.float32),
            grad_weight,
            feasibility_data,
        )

    def to_params(self) -> params_t:
        """return the underlying parameter dictionary."""
        return self.params


def train_lpcf(
    q: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    q_dim: int,
    theta_dim: int,
    w: np.ndarray | None = None,
    convex_widths: Sequence[int] = default_convex_widths,
    hyper_widths: Sequence[int] = default_hyper_widths,
    lr: float = 1e-3,
    grad_weight: float = 1.0,
    l2_reg: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    lr_decay_rate: float = 1.0,
    lr_decay_steps: int | None = None,
    seed: int = 0,
    q_val: np.ndarray | None = None,
    theta_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    feasibility_data: params_t | None = None,
) -> params_t:
    """train an lpcf with value and autodiff q-gradient supervision."""
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if lr_decay_rate <= 0.0 or lr_decay_rate > 1.0:
        raise ValueError("lr_decay_rate must be in (0, 1]")
    if lr_decay_steps is not None and lr_decay_steps <= 0:
        raise ValueError("lr_decay_steps must be positive")
    if w is None:
        w = np.ones(q.shape[0])

    key = jax.random.PRNGKey(seed)
    qj, thetaj, yj, gj, wj = as_training_arrays(
        q,
        theta,
        y,
        g,
        q_dim,
        theta_dim,
        w,
    )

    has_validation = (
        q_val is not None
        or theta_val is not None
        or y_val is not None
        or g_val is not None
        or w_val is not None
    )
    if has_validation:
        if q_val is None or theta_val is None or y_val is None or g_val is None:
            raise ValueError("q_val, theta_val, y_val, and g_val must be provided together")
        if w_val is None:
            w_val = np.ones(q_val.shape[0])
        qvj, thetavj, yvj, gvj, wvj = as_training_arrays(
            q_val, theta_val, y_val, g_val, q_dim, theta_dim, w_val
        )

    params = init_lpcf_params(key, q_dim, theta_dim, convex_widths, hyper_widths)
    best_params = params
    best_val = float("inf")

    # if lr_decay_rate < 1.0:
    #     steps_per_epoch = max(1, int(np.ceil(qj.shape[0] / batch_size)))
    #     decay_steps = lr_decay_steps or steps_per_epoch
    learning_rate = optax.exponential_decay(
        init_value=lr,
        transition_steps=600,
        decay_rate=lr_decay_rate,
        )
    # else:
    # learning_rate = lr

    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=l2_reg)
    opt_state = optimizer.init(params["hyper"])
    params_template = dict(params)
    params_template.pop("hyper")
    train_step = make_train_step(optimizer, params_template, feasibility_data)
    eval_step = make_eval_step(params_template, feasibility_data)

    for epoch in range(1, epochs + 1):
        batch_key, key = jax.random.split(key)
        for qb, thetab, yb, gb, wb in batch_iterator(
            qj, thetaj, yj, gj, wj, batch_size, batch_key
        ):
            hyper_params, opt_state, _loss, _vmse, _gmse, _fmse = train_step(
                params["hyper"],
                opt_state,
                qb,
                thetab,
                yb,
                gb,
                wb,
                grad_weight,
            )
            params = dict(params)
            params["hyper"] = hyper_params

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            if has_validation:
                eval_q, eval_theta, eval_y, eval_g, eval_w = qvj, thetavj, yvj, gvj, wvj
                metric_label = "val"
            else:
                eval_q, eval_theta, eval_y, eval_g, eval_w = (
                    qj[:1024],
                    thetaj[:1024],
                    yj[:1024],
                    gj[:1024],
                    wj[:1024],
                )
                metric_label = "train"

            selection_obj, vm, gm, fm = eval_step(
                params["hyper"],
                eval_q,
                eval_theta,
                eval_y,
                eval_g,
                eval_w,
                grad_weight,
            )

            if float(selection_obj) < best_val:
                best_val = float(selection_obj)
                best_params = copy_params(params)
            print(
                f"epoch {epoch:4d} | {metric_label} value mse: {vm:.4e} "
                f"| {metric_label} grad mse: {gm:.4e} "
                f"| {metric_label} feasibility mse: {fm:.4e}"
            )

    return best_params
