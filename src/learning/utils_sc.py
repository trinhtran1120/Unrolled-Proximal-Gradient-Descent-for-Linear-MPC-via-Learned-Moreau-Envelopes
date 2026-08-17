"""
Utility functions for fitting a parametric convex function to data
and exporting it to cvxpy etc.

M. Schaller, A. Bemporad, March 19, 2025
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import warnings

try:
    from jax_sysid.utils import compute_scores
except ModuleNotFoundError:
    compute_scores = None


params_t = dict[str, Any]
batch_t = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

default_convex_widths = (32, 32)
default_hyper_widths = (64, 64)
init_scale = 0.05


@dataclass
class Activation:
    jax: Callable
    cvxpy: Callable
    convex_increasing: bool
    
    
@dataclass
class MakePositive:
    jax: Callable
    cvxpy: Callable


@dataclass
class Ind:
    start: int = 0
    end: int = 0


@dataclass
class WeightInd:
    w: Ind
    v: Ind
    o: Ind


@dataclass
class Section:
    start: int = 0
    end: int = 0
    shape: tuple = (0, 0)
    

def _rand(m, n):
    """Compute a ramdom array of shape (m, n) and entries in [-0.5, 0.5]"""
    return np.random.rand(m, n) - 0.5


def _unsqueeze(x):
    """If array has shape (n,), turn into array of shape (n, 1)"""
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _map_matmul(A, b):
    """Map matrix multiplication with JAX"""
    
    return jax.vmap(jnp.matmul)(A, b)


def _append_section(section, offset, shape, size=None):
    """Add section in psi output p, such that p[start:end].reshape(shape)
    is a matrix or vector in main network; returns offset for next section"""
    
    if size is None:
        size = np.prod(shape)
    section.append(Section(start=offset, end=offset+size, shape=shape))
    return offset+size


def _compute_r2(Y, Yhat, return_msg=False):
    """Compute R2 score for true outputs Y and predicted outputs Yhat"""
    if compute_scores is None:
        residual = np.sum((Y - Yhat) ** 2, axis=0)
        centered = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        r2 = np.mean(1.0 - residual / np.maximum(centered, np.finfo(float).eps))
        msg = f"R2: {r2:.4f}"
    else:
        r2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')
        r2 = np.mean(r2)
    if return_msg:
        return r2, msg
    else:
        return r2


def _compute_acc(Y, Yhat, return_msg=False):
    """Compute accuracy for true labls Y and predicted labels Yhat"""
    if compute_scores is None:
        acc = np.mean((Y == -1) == (Yhat <= 0))
        msg = f"Accuracy: {acc:.4f}"
    else:
        acc, _, msg = compute_scores(Y==-1, Yhat<=0, None, None, fit='Accuracy')
        acc = np.mean(acc)
    if return_msg:
        return acc, msg
    else:
        return acc


def _extract_activations(activation_registry, activation, activation_psi):
    """Extract activation functions from activation and activation_psi options"""
    
    # check that main activation is convex and increasing
    activation = activation.lower()
    if not activation_registry[activation].convex_increasing:
        raise ValueError('Activation function for variable network must'
                            'be convex and increasing.')
        
    # if not specified, make psi activation equal to main activation
    if activation_psi is None:
        activation_psi = activation
    else:
        activation_psi = activation_psi.lower()
    
    # extract main activations
    act_jax = activation_registry[activation].jax
    act_cvxpy = activation_registry[activation].cvxpy
    
    # extract psi activations
    act_psi_jax = activation_registry[activation_psi].jax
    act_psi_cvxpy = activation_registry[activation_psi].cvxpy
    
    return act_jax, act_cvxpy, act_psi_jax, act_psi_cvxpy


def _extract_monotonicity(increasing, decreasing):
    """Extract monotonicity multiplier from increasing/decreasing options"""
    
    if increasing and decreasing:
        warnings.warn("\033[1mFunction enforced to be both increasing and decreasing.\033[0m")
        return 0
    elif increasing:
        return 1
    elif decreasing:
        return -1
    else:
        return None


def act_p(x: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained weights to the nonnegative orthant."""
    return jax.nn.softplus(x)


def activation(x: jnp.ndarray) -> jnp.ndarray:
    """Convex nondecreasing activation for the variable network."""
    return jax.nn.softplus(x)


def activation_derivative(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of softplus for manual q-gradient propagation."""
    return jax.nn.sigmoid(x)


def glorot_uniform(key: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    """Initialize a dense layer with Glorot-uniform weights."""
    fan_in, fan_out = shape[1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def validate_widths(widths: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate hidden-layer widths."""
    widths = tuple(int(width) for width in widths)
    if not widths:
        raise ValueError(f"{name} must contain at least one hidden layer")
    if any(width <= 0 for width in widths):
        raise ValueError(f"all {name} entries must be positive")
    return widths


def copy_params(params: params_t) -> params_t:
    """Copy a JAX pytree of model parameters."""
    return jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        params,
    )


def to_serializable(obj: Any) -> Any:
    """Recursively convert JAX/NumPy arrays to JSON-serializable values."""
    if isinstance(obj, (jax.Array, jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj
