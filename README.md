# Learned Moreau Envelopes for Linear MPC

Code and supplementary material for **Unrolled Proximal Gradient Descent for
Linear MPC via Learned Moreau Envelopes**.

The repository compares three open-loop linear MPC solution alogrithms:

- a JuMP-based baseline solver, currently configured for OSQP/Ipopt/Gurobi/Mosek;
- an exact proximal-gradient method (PGM) that uses a Moreau envelope of the MPC
  constraints;
- a learned PGM variant that replaces the exact Moreau-envelope gradient with an
  input-convex neural network (ICNN).

## Repository Layout

```text
src/mpc/              Julia MPC problem definitions and solvers
src/utils/            Julia model-loading and preprocessing utilities
src/scripts/          Dataset generation and benchmark scripts
src/learning/         Python/JAX ICNN training code
data/                 Generated `.npz` training and test datasets
model/                Exported ICNN model artifacts
test/                 Julia tests for the solver stack
Project.toml          Julia environment
Manifest.toml         Julia package manifest
```

## Requirements

Julia dependencies are managed by `Project.toml` and `Manifest.toml`. The scripts
use JuMP with solver backends including OSQP, Ipopt, Gurobi, and MosekTools. OSQP
is the default in the included data and benchmark scripts.

The ICNN training code is Python-based and imports:

- `jax`
- `jax.numpy`
- `numpy`
- `optax`

Install these in your preferred Python environment before running the training
script.

## Setup

Instantiate the Julia environment from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the Julia tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Generate Data

The dataset script solves the linear MPC problem, runs exact PGM, and logs
projection queries, initial states, and exact projection labels for neural
projection training.

```bash
julia --project=. src/scripts/get_data.jl
```

This writes files like:

```text
data/PGM-mu=0.001_gamma=0.001_nx=2_N=10-train_adaptive.npz
data/PGM-mu=0.001_gamma=0.001_nx=2_N=10-test_adaptive.npz
```

## Train the Projection MLP

Train and export the learned projection model:

```bash
python src/learning/train_nn.py
```

The training script currently reads the `mu=0.001, gamma=0.001` datasets and writes:

```text
model/linear-mpc-projection-mlp_tanh.pkl
model/linear-mpc-projection-mlp_tanh.json
```

The projection MLP learns the map from `(U_query, x0)` to `V_star`. At runtime,
the learned PGM forms the Moreau-envelope value and gradient from that learned
projection and the positive Moreau coefficient.

## Run the Benchmark

Compare the JuMP baseline, exact PGM, and learned PGM:

```bash
julia --project=. src/scripts/get_benchmark.jl
```

The benchmark reports objective values, iteration counts, relative objective
gaps, control/state differences from the baseline solution, and repeated timing
statistics.

## MPC Example

The default problem is defined in `src/mpc/problem.jl`:

- state dimension `nx = 2`;
- input dimension `nu = 1`;
- horizon `N = 10`;
- box constraints on state and input;
- quadratic stage cost with matrices `Q` and `R`.

`src/mpc/solver.jl` builds the baseline JuMP model. `src/mpc/pgm.jl` contains
the exact PGM implementation and dataset logging hooks. `src/mpc/learned_pgm.jl`
loads the exported ICNN and uses its gradient inside the learned PGM loop.
