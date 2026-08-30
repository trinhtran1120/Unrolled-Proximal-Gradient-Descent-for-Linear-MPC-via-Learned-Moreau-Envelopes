# Coding Guide: Learned Three-Operator Splitting for Single-Shooting Linear MPC

## 1. Purpose and scope

This document specifies how to extend the current proximal-gradient linear MPC implementation into a Davis--Yin three-operator splitting (TOS) solver and then replace the exact state-constraint Moreau-gradient evaluation with the neural projection model developed in the current ACC paper.

The implementation should be developed in three stages:

1. **Exact TOS baseline:** use an exact QP projection to evaluate the Moreau-envelope gradient.
2. **Projection-model training:** collect exact projection labels and train the neural projection model with the affine equality-correction layer.
3. **Learned TOS inference:** replace only the exact state projection in TOS with the learned projection. Keep the input-box projection and quadratic-cost proximal map exact.

Do not begin performance claims from the learned solver alone. The exact TOS baseline is required to distinguish errors caused by the operator splitting from errors caused by the learned Moreau gradient.

---

## 2. Linear MPC and single-shooting condensation

Consider the discrete-time linear system

$$
x_{j+1}=Ax_j+Bu_j,
\qquad j=0,\ldots,N-1,
$$

where

$$
x_j\in\mathbb R^{n_x},
\qquad
u_j\in\mathbb R^{n_u},
\qquad
x_0\in\mathbb R^{n_x}.
$$

Stack the control inputs as

$$
U=\operatorname{col}(u_0,\ldots,u_{N-1})\in\mathbb R^{n_U},
\qquad n_U=Nn_u,
$$

and the predicted states as

$$
X=\operatorname{col}(x_1,\ldots,x_N)\in\mathbb R^{n_X},
\qquad n_X=Nn_x.
$$

Repeated substitution of the dynamics gives

$$
X=Gx_0+HU,
$$

with

$$
G=
\begin{bmatrix}
A\\
A^2\\
\vdots\\
A^N
\end{bmatrix}
\in\mathbb R^{n_X\times n_x},
$$

and

$$
H=
\begin{bmatrix}
B & 0 & \cdots & 0\\
AB & B & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
A^{N-1}B & A^{N-2}B & \cdots & B
\end{bmatrix}
\in\mathbb R^{n_X\times n_U}.
$$

In code, avoid using the same symbol for the condensed dynamics matrix and the quadratic cost Hessian. Recommended names are:

- `Sx` for $G$;
- `Su` for $H$;
- `Qbar` and `Rbar` for horizon cost matrices;
- `Hcost` for the condensed cost Hessian;
- `Gx` for the condensed state-inequality matrix.

### 2.1 Input constraints

The stagewise input bounds are

$$
\underline u\le u_j\le\overline u.
$$

After stacking,

$$
\mathcal U=
\left\{
U\in\mathbb R^{n_U}:\underline U\le U\le\overline U
\right\},
$$

where

$$
\underline U=\mathbf 1_N\otimes\underline u,
\qquad
\overline U=\mathbf 1_N\otimes\overline u.
$$

The Euclidean projection is componentwise clipping:

$$
\Pi_{\mathcal U}(W)
=
\min\{\max\{W,\underline U\},\overline U\}.
$$

This operation must remain exact in both the classical and learned solvers.

### 2.2 State constraints

The state bounds are

$$
\underline X\le X\le\overline X.
$$

Substitution of $X=Gx_0+HU$ gives

$$
G_xU\le b_x(x_0),
$$

where

$$
G_x=
\begin{bmatrix}
H\\
-H
\end{bmatrix}
\in\mathbb R^{m\times n_U},
\qquad
m=2Nn_x,
$$

and

$$
b_x(x_0)=
\begin{bmatrix}
\overline X-Gx_0\\
-\underline X+Gx_0
\end{bmatrix}
\in\mathbb R^m.
$$

Define the state-feasible control set

$$
\mathcal C(x_0)=
\left\{
U\in\mathbb R^{n_U}:G_xU\le b_x(x_0)
\right\}.
$$

The set depends on the measured initial state $x_0$, whereas $G_x$ is constant when the system and horizon are fixed.

### 2.3 Condensed quadratic cost

For a standard quadratic tracking objective, write the condensed cost using one consistent convention:

$$
\ell(U;x_0)
=
\frac12U^\top H_cU+q(x_0)^\top U+c(x_0),
$$

where

$$
H_c\in\mathbb R^{n_U\times n_U},
\qquad H_c\succeq0,
\qquad q(x_0)\in\mathbb R^{n_U}.
$$

The implementation below assumes the factor $1/2$ is present. If the existing code uses $U^\top H_cU$ instead, adjust every gradient and proximal formula consistently.

The gradient is

$$
\nabla_U\ell(U;x_0)=H_cU+q(x_0).
$$

The proximal map required by TOS is

$$
\operatorname{prox}_{\gamma\ell(\cdot;x_0)}(W)
=
(I+\gamma H_c)^{-1}\bigl(W-\gamma q(x_0)\bigr).
$$

For a fixed $\gamma$, factorize $I+\gamma H_c$ once offline. Do not form its matrix inverse explicitly.

---

## 3. Moreau smoothing of the state constraints

The exact condensed MPC objective is

$$
\ell(U;x_0)
+\delta_{\mathcal U}(U)
+\delta_{\mathcal C(x_0)}(U).
$$

This implementation uses three distinct coefficients:

- $\mu>0$ is the Moreau smoothing coefficient;
- $\gamma>0$ is the Davis--Yin operator-splitting stepsize;
- $\lambda>0$ is the fixed relaxation parameter.

Only $\gamma$ is adapted. The Moreau coefficient $\mu$ and relaxation
$\lambda$ remain fixed. Replace the state-set indicator with its Moreau
envelope:

$$
M_\mu^{\mathcal C}(U;x_0)
=
\min_V
\left\{
\delta_{\mathcal C(x_0)}(V)
+\frac{1}{2\mu}\|V-U\|^2
\right\}.
$$

Equivalently,

$$
M_\mu^{\mathcal C}(U;x_0)
=
\frac{1}{2\mu}
\operatorname{dist}^2\bigl(U,\mathcal C(x_0)\bigr).
$$

Let

$$
V^\star(U,x_0)
=
\Pi_{\mathcal C(x_0)}(U)
=
\arg\min_V\frac12\|V-U\|^2
\quad\text{subject to}\quad
G_xV\le b_x(x_0).
$$

Then

$$
\nabla_UM_\mu^{\mathcal C}(U;x_0)
=
\frac1\mu\left(U-V^\star(U,x_0)\right).
$$

The exact Moreau gradient is $1/\mu$-Lipschitz. Hence it is
$\mu$-cocoercive.

The smoothed optimization problem is

$$
\min_U
\ell(U;x_0)
+\delta_{\mathcal U}(U)
+M_\mu^{\mathcal C}(U;x_0).
$$

Important: for finite $\mu$, this is a penalized problem. Its minimizer is not guaranteed to satisfy $G_xU\le b_x(x_0)$. Moreau smoothing and TOS do not by themselves provide hard state feasibility.

---

## 4. Three-operator splitting formulation

Use the decomposition

$$
f(U)=\ell(U;x_0),
\qquad
g(U)=\delta_{\mathcal U}(U),
\qquad
h(U)=M_\mu^{\mathcal C}(U;x_0).
$$

This assignment is recommended because:

- $\operatorname{prox}_{\gamma f}$ is a structured linear solve;
- $\operatorname{prox}_{\gamma g}$ is componentwise clipping;
- $\nabla h$ is the only expensive operation and is therefore the component to learn.

The corresponding monotone inclusion is

$$
0\in
\partial f(U)+\partial g(U)+\nabla h(U).
$$

### 4.1 Exact Davis--Yin iteration

Let $Z^k\in\mathbb R^{n_U}$ be the fixed-point variable. One iteration is

$$
X^k=\operatorname{prox}_{\gamma g}(Z^k)
=\Pi_{\mathcal U}(Z^k),
$$

$$
D^k=\nabla h(X^k)
=\frac1\mu
\left(
X^k-\Pi_{\mathcal C(x_0)}(X^k)
\right),
$$

$$
R^k=2X^k-Z^k-\gamma D^k,
$$

$$
Y^k=\operatorname{prox}_{\gamma f}(R^k)
=(I+\gamma H_c)^{-1}
\left(R^k-\gamma q(x_0)\right),
$$

and

$$
Z^{k+1}=Z^k+\lambda_k(Y^k-X^k).
$$

The natural fixed-point residual is

$$
r_{\mathrm{TOS}}^k=Y^k-X^k.
$$

Possible normalized stopping criteria are

$$
\frac{\|Y^k-X^k\|_2}{\max\{1,\|X^k\|_2\}}
\le\varepsilon_{\mathrm{TOS}}
$$

or

$$
\|Z^{k+1}-Z^k\|_2
\le\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{rel}}\|Z^k\|_2.
$$

For the exact Moreau gradient, the standard Davis--Yin stepsize condition is

$$
0<\gamma<2\mu.
$$

The relaxation condition becomes

$$
0<\lambda_k<2-\frac{\gamma}{2\mu}.
$$

The simple fixed choice $\lambda_k=1$ is admissible whenever $0<\gamma<2\mu$.

### 4.2 Which iterate is the solver output?

At a fixed point,

$$
X^\star=Y^\star
$$

and this common value solves the smoothed problem. At a finite iteration:

- $X^k$ satisfies the input box exactly;
- $Y^k$ need not satisfy the input box;
- neither point is guaranteed to satisfy the original state constraints.

Use $X^K$ as the default finite-iteration output because it is exactly input feasible. Apply the first control block

$$
u_0=\left[X^K\right]_{1:n_u}.
$$

If hard state feasibility is required, pass $X^K$ through a separately specified exact feasibility-correction or safety layer before applying $u_0$.

### 4.3 Exact TOS pseudocode with Algorithm 3 line search

```text
Inputs:
    x0                  measured state, shape (nx,)
    Z0                  initial fixed-point variable, shape (nU,)
    Hcost               condensed cost Hessian, shape (nU, nU)
    q(x0)               condensed linear cost term, shape (nU,)
    lower_U, upper_U    input bounds, each shape (nU,)
    Gx                  state-constraint matrix, shape (m, nU)
    bx(x0)              state-constraint right-hand side, shape (m,)
    mu                  Moreau smoothing coefficient, fixed
    gamma, lambda       algorithm parameters; gamma is the splitting stepsize, lambda is fixed relaxation
    gamma_min           smallest allowed splitting gamma
    gamma_reduce        factor in (0, 1) used by the line search
    gamma_increase      optional factor above 1 tried before the line search
    K                   maximum number of iterations
    tolerance           stopping tolerance

Z = Z0
for k = 0, ..., K - 1:
    gamma_candidate = gamma * gamma_increase

    repeat:
        X = clip(Z, lower_U, upper_U)
        Vstar = solve_projection_qp(X, Gx, bx(x0))
        hX = (1 / mu) * 0.5 * norm(X - Vstar)^2
        D = (X - Vstar) / mu

        R = 2 * X - Z - gamma_candidate * D
        Y = solve(I + gamma_candidate * Hcost, R - gamma_candidate * q(x0))

        VY = solve_projection_qp(Y, Gx, bx(x0))
        hY = (1 / mu) * 0.5 * norm(Y - VY)^2

        residual = Y - X
        upper_model =
            hX
            + dot(residual, D)
            + (1 / (2 * gamma_candidate)) * norm(residual)^2

        if hY <= upper_model or gamma_candidate <= gamma_min:
            gamma = gamma_candidate
            break

        gamma_candidate = max(gamma_min, gamma_reduce * gamma_candidate)

    residual = Y - X
    Z = Z + lambda * residual
    if norm(residual) / max(1, norm(X)) <= tolerance:
        break
return X, Z, diagnostics
```

The line-search test is the one from Algorithm 3 in
`three-operatro_splitting.pdf`, specialized to
$h(U)=M_\mu^{\mathcal C}(U;x_0)$:

$$
h(Y)\le h(X)+\langle Y-X,\nabla h(X)\rangle
+\frac{1}{2\gamma_{\mathrm{candidate}}}\|Y-X\|^2.
$$

This is not a residual-monotonicity test. The Moreau envelope $h$ is evaluated
with the fixed smoothing parameter $\mu$ throughout the line search. Only the
splitting stepsize $\gamma$ changes, through the candidate
$\gamma_{\mathrm{candidate}}$. The relaxation parameter remains fixed.

### 4.4 Recommended exact-baseline function interface

```julia
result = solve_tos_exact(problem, x0;
    Z0=nothing,
    gamma,
    relaxation=1.0,
    max_iter,
    atol,
    rtol,
    warm_start=nothing,
)
```

Recommended result fields:

```text
result.U                  final X iterate, shape (nU,)
result.u0                 first control input, shape (nu,)
result.Z                  final fixed-point variable, shape (nU,)
result.iterations         number of iterations
result.converged          Boolean
result.fixed_point_norm   ||Y - X||
result.objective_smooth   ell(U;x0) + M_mu^C(U;x0)
result.input_violation    input-bound violation
result.state_violation    ||[Gx U - bx(x0)]_+||
result.projection_time    accumulated exact projection time
result.total_time         total solver time
```

---

## 5. Data collection for the neural projection

The network approximates the projection

$$
(U,x_0)\longmapsto V^\star(U,x_0)
=\Pi_{\mathcal C(x_0)}(U).
$$

Because TOS evaluates the Moreau gradient at

$$
X^k=\Pi_{\mathcal U}(Z^k),
$$

the most important training query points lie inside the input box. Training only on arbitrary Gaussian $U$ values can create a severe inference-distribution mismatch.

### 5.1 One supervised sample

For sample $i$, generate an initial state and projection query

$$
x_{0,i}\in\mathbb R^{n_x},
\qquad
U_i\in\mathbb R^{n_U}.
$$

Compute

$$
b_i=b_x(x_{0,i})\in\mathbb R^m.
$$

Solve the exact projection QP

$$
V_i^\star
=
\arg\min_V\frac12\|V-U_i\|^2
\quad\text{subject to}\quad
G_xV\le b_i.
$$

For the projection network, store only the query, parameter, and exact
projection label. The slack, distance, distance gradient, solver status, and
iteration diagnostics can be recomputed later if a different loss needs them.

The exact slack is

$$
s_i^\star=b_i-G_xV_i^\star\in\mathbb R^m,
\qquad s_i^\star\ge0,
$$

the exact half squared distance

$$
d_i^\star=\frac{1}{2}\|U_i-V_i^\star\|^2,
$$

and the exact distance gradient

$$
\nabla d_i^\star=U_i-V_i^\star\in\mathbb R^{n_U}.
$$

Store the following fields:

| Field | Shape | Role |
|---|---:|---|
| `x0` | `(nx,)` | parameter defining the state-feasible set |
| `U_query` | `(nU,)` | point to be projected; later corresponds to $X^k$ |
| `V_star` | `(nU,)` | exact projection target |

### 5.2 Sampling initial states

Sample $x_0$ from the intended closed-loop operating region, not merely from a broad unconstrained Gaussian. Use a mixture of:

1. states encountered in nominal closed-loop MPC simulations;
2. states near state-constraint boundaries;
3. perturbed states around nominal trajectories;
4. states from multiple references if the reference varies online.

Before labeling projection queries, verify that $\mathcal C(x_0)$ is nonempty. If the research problem intentionally includes infeasible MPC instances, assign them to a separate dataset because the projection onto an empty set is undefined.

The train/validation/test split must be performed by trajectory or by initial state before generating multiple $U$ queries. Randomly splitting individual $(U,x_0)$ pairs can leak nearly identical states across splits.

### 5.3 Sampling projection queries

Use a mixture rather than one sampling rule.

#### A. Broad box samples

Sample

$$
U_i\sim\operatorname{Uniform}(\underline U,\overline U).
$$

This gives global coverage of the domain at which TOS evaluates the learned gradient.

#### B. Perturbed expert solutions

First solve the exact MPC or exact smoothed problem and obtain $U_i^{\mathrm{expert}}$. Generate

$$
U_i=
\Pi_{\mathcal U}
\left(
U_i^{\mathrm{expert}}+\sigma\varepsilon_i
\right),
\qquad
\varepsilon_i\sim\mathcal N(0,I).
$$

Use several noise scales $\sigma$. These samples emphasize the region relevant near convergence.

#### C. Boundary-focused samples

Generate points near active state constraints by perturbing feasible expert controls or projection solutions. These samples are important because the Moreau gradient changes most significantly when active constraints change.

#### D. On-policy TOS samples

Run the exact TOS solver for sampled $x_0$ and record

$$
U_i=X^k=\Pi_{\mathcal U}(Z^k)
$$

at multiple iterations $k$. Solve the exact projection QP for each unique recorded query and append the corresponding labels.

This is the most important dataset for matching deployment behavior.

### 5.4 Recommended staged data collection

Use dataset aggregation:

1. **Dataset 0:** broad box samples, expert perturbations, and boundary-focused samples.
2. Train an initial projection network.
3. Run learned TOS on training-region initial states.
4. Record its actual $X^k$ queries, especially samples with large prediction error, high state violation, or large fixed-point residual.
5. Relabel these queries using the exact projection QP.
6. Append them to the dataset and retrain.
7. Repeat until the rollout distribution and performance stabilize.

This procedure is preferable to increasing the number of independent random samples without checking where the learned solver actually operates.

### 5.5 Label generation pseudocode

```text
Inputs:
    sampled initial states {x0_j}
    query generator
    constant Gx
    exact QP projection solver

dataset = []
for each x0_j:
    b = bx(x0_j)
    if C(x0_j) is empty:
        mark or skip according to dataset policy
        continue

    queries = generate_broad_and_expert_queries(x0_j)
    queries += collect_exact_TOS_queries(x0_j)

    for U in unique(queries):
        Vstar = solve min 0.5 * ||V - U||^2 subject to Gx * V <= b
        require acceptable solver status and KKT residual
        append (x0_j, U, Vstar)
return dataset
```

### 5.6 Exact projection solver details

The QP Hessian is the identity and $G_x$ is constant. Reuse the solver model and update only:

- the linear objective term $-U_i$;
- the right-hand side $b_x(x_{0,i})$.

Warm-start successive queries for the same trajectory or nearby $U_i$. Record both solver time and KKT residual. A failed or inaccurate QP solve must not silently become a training target.

---

## 6. Neural projection model

### 6.1 Model input

The primary network input is

$$
\xi_i=
\operatorname{col}(U_i,x_{0,i})
\in\mathbb R^{n_U+n_x}.
$$

Thus the input dimension is

$$
n_{\mathrm{in}}=Nn_u+n_x.
$$

Use normalized quantities:

$$
\bar U=(U-c_U)\oslash s_U,
\qquad
\bar x_0=(x_0-c_x)\oslash s_x,
$$

where $\oslash$ denotes elementwise division. Store the normalization constants with the trained model.

An alternative input is $\operatorname{col}(U,b_x(x_0))$. This directly exposes the changing polyhedron but increases the input dimension from $n_x$ to $m$. Do not mix the two interfaces in one experiment. Start with $(U,x_0)$ to match the current paper, then test $(U,b_x(x_0))$ as an ablation.

### 6.2 Raw model output

The network predicts both a projected control and a slack vector:

$$
\widehat z_i
=h_\theta(U_i,x_{0,i})
=
\begin{bmatrix}
\widehat V_i\\
\widehat s_i
\end{bmatrix}
\in\mathbb R^{n_U+m}.
$$

The output dimension is

$$
n_{\mathrm{out}}
=n_U+m
=Nn_u+2Nn_x.
$$

Do not describe $\widehat V$ as feasible. It is only a raw network prediction.

### 6.3 Affine equality-correction layer

The slack reformulation is

$$
G_xV+s=b_x(x_0),
\qquad s\ge0.
$$

Define

$$
z=
\begin{bmatrix}
V\\s
\end{bmatrix},
\qquad
E=
\begin{bmatrix}
G_x&I_m
\end{bmatrix}
\in\mathbb R^{m\times(n_U+m)}.
$$

The affine layer computes

$$
\widetilde z
=
\widehat z
-E^\top(EE^\top)^{-1}
\left(E\widehat z-b_x(x_0)\right).
$$

Write

$$
\widetilde z=
\begin{bmatrix}
\widetilde V\\
\widetilde s
\end{bmatrix}.
$$

Because

$$
EE^\top=G_xG_x^\top+I_m\succ0,
$$

factorize $EE^\top$ once. Never explicitly form $(EE^\top)^{-1}$.

The affine layer guarantees

$$
G_x\widetilde V+\widetilde s=b_x(x_0)
$$

up to numerical precision. It does **not** guarantee $\widetilde s\ge0$. Therefore it does not guarantee $G_x\widetilde V\le b_x(x_0)$.

### 6.4 Learned Moreau-gradient output

The component used by TOS is

$$
\widehat D_\theta(U,x_0)
=
\frac1\mu
\left(U-\widetilde V_\theta(U,x_0)\right)
\in\mathbb R^{n_U}.
$$

The complete learned module therefore has the interface

$$
(U,x_0)
\longmapsto
(\widetilde V,\widetilde s,\widehat D).
$$

At deployment, TOS consumes only $\widehat D$, but return $\widetilde V$ and $\widetilde s$ for diagnostics and training.

### 6.5 Training loss

The current paper uses a raw-output loss of the form

$$
\mathcal L_{\mathrm{raw}}(\theta)
=
\frac1{N_s}\sum_i
\left(
\|\widehat V_i-V_i^\star\|^2
+\lambda_{\mathrm{eq}}
\|G_x\widehat V_i+\widehat s_i-b_i\|^2
+\lambda_+
\|[-\widehat s_i]_+\|^2
\right).
$$

For the TOS implementation, train through the affine correction layer and evaluate the quantities actually used at inference. A recommended loss is

$$
\mathcal L(\theta)
=
\lambda_V\mathcal L_V
+\lambda_D\mathcal L_D
+\lambda_s\mathcal L_s
+\lambda_+\mathcal L_+,
$$

where

$$
\mathcal L_V
=
\frac1{N_s}\sum_i
\|\widetilde V_i-V_i^\star\|^2,
$$

$$
\mathcal L_D
=
\frac1{N_s}\sum_i
\left\|
\frac{U_i-\widetilde V_i}{\gamma}-D_i^\star
\right\|^2,
$$

$$
\mathcal L_s
=
\frac1{N_s}\sum_i
\|\widetilde s_i-s_i^\star\|^2,
$$

and

$$
\mathcal L_+
=
\frac1{N_s}\sum_i
\|[-\widetilde s_i]_+\|^2.
$$

Since

$$
\nabla d_i^\star=U_i-V_i^\star,
$$

$\mathcal L_V$ and the distance-gradient loss contain related information. Begin with $\mathcal L_V+\lambda_+\mathcal L_+$ and then ablate gradient and slack supervision rather than assuming all terms are necessary.

The post-layer equality residual

$$
\|G_x\widetilde V_i+\widetilde s_i-b_i\|
$$

should be monitored as a numerical assertion, not used as a main loss, because the affine layer already enforces it.

### 6.6 Batch tensor shapes

For batch size $B_s$:

| Tensor | Shape |
|---|---:|
| `U_query` | `(nU, Bs)` or `(Bs, nU)` |
| `x0` | `(nx, Bs)` or `(Bs, nx)` |
| `network_input` | `(nU + nx, Bs)` or `(Bs, nU + nx)` |
| `V_hat`, `V_tilde`, `V_star` | `(nU, Bs)` or `(Bs, nU)` |
| `s_hat`, `s_tilde` | `(m, Bs)` or `(Bs, m)` |

Choose one batch convention and use it everywhere. Julia/Flux often uses features-by-batch, whereas many Python frameworks use batch-by-features.

---

## 7. Learned three-operator splitting

Replace only the exact Moreau-gradient evaluation:

$$
\frac1\mu
\left(
X^k-\Pi_{\mathcal C(x_0)}(X^k)
\right)
$$

with

$$
\widehat D^k
=
\frac1\mu
\left(
X^k-\widetilde V_\theta(X^k,x_0)
\right).
$$

The learned iteration is

$$
X^k=\Pi_{\mathcal U}(Z^k),
$$

$$
(\widetilde V^k,\widetilde s^k,\widehat D^k)
=
\operatorname{LearnedProjection}_\theta(X^k,x_0),
$$

$$
R^k=2X^k-Z^k-\gamma\widehat D^k,
$$

$$
Y^k=(I+\gamma H_c)^{-1}
\left(R^k-\gamma q(x_0)\right),
$$

$$
Z^{k+1}=Z^k+\lambda_k(Y^k-X^k).
$$

### 7.1 Learned TOS pseudocode

```text
Inputs:
    x0, Z0
    exact MPC matrices and bounds
    trained model and normalization statistics
    mu, gamma, lambda
    maximum iteration count K

Offline:
    factorize I + gamma * Hcost
    factorize Gx * Gx' + I

Z = Z0
for k = 0, ..., K - 1:
    # Exact box proximal step
    X = clip(Z, lower_U, upper_U)

    # Learned state-projection module
    network_input = normalize(concatenate(X, x0))
    (Vhat, shat) = neural_network(network_input)
    b = bx(x0)
    residual_eq = Gx * Vhat + shat - b
    w = solve_factorized(Gx * Gx' + I, residual_eq)
    Vtilde = Vhat - Gx' * w
    stilde = shat - w
    Dhat = (X - Vtilde) / mu

    # Exact quadratic-cost proximal step
    R = 2 * X - Z - gamma * Dhat
    Y = solve_factorized(I + gamma * Hcost, R - gamma * q(x0))

    # Davis--Yin relaxation
    fixed_point_residual = Y - X
    Z = Z + lambda * fixed_point_residual

    record diagnostics
    if stopping test is satisfied:
        break

return X, Z, diagnostics
```

The two corrected outputs can be implemented without assembling $E$:

$$
w=(G_xG_x^\top+I)^{-1}
\left(G_x\widehat V+\widehat s-b_x(x_0)\right),
$$

$$
\widetilde V=\widehat V-G_x^\top w,
\qquad
\widetilde s=\widehat s-w.
$$

### 7.2 Recommended model interface

```julia
Vtilde, stilde, Dhat, model_diag = learned_moreau_gradient(
    model,
    X,
    x0,
    problem,
    affine_factorization,
    normalization,
    gamma,
)
```

Recommended diagnostics:

```text
model_diag.raw_equality_residual
model_diag.corrected_equality_residual
model_diag.minimum_corrected_slack
model_diag.predicted_state_violation
model_diag.gradient_norm
```

### 7.3 Warm start

In receding-horizon MPC, shift the previous solution:

$$
U_{mathrm{shift}}
=
\operatorname{col}
(u_1^{\mathrm{prev}},\ldots,u_{N-1}^{\mathrm{prev}},u_{N-1}^{\mathrm{prev}}).
$$

Because $Z^k$ is not itself the primal control, distinguish between:

- warm-starting the primal point $X^0$;
- warm-starting the fixed-point variable $Z^0$.

The simplest first implementation uses

$$
Z^0=U_{\mathrm{shift}}.
$$

Later, compare it with shifting the previous final $Z$ directly. Report which warm-start rule is used.

### 7.4 Algorithm 3 line search and fixed relaxation

The implementation fixes the relaxation parameter $\lambda$ and the Moreau
smoothing parameter $\mu$. The Algorithm 3 line search varies only the
splitting stepsize $\gamma$, through the candidate
$\gamma_{\mathrm{candidate}}$, used in

$$
J_{\gamma_{\mathrm{candidate}}A}
$$

and in the forward/reflection argument. The smooth term remains

$$
h(U)=M_\mu^{\mathcal C}(U;x_0),
$$

so changing $\gamma$ does not change the smoothed objective itself. Each
accepted candidate stepsize requires solving with

$$
I+\gamma_{\mathrm{candidate}}H_c.
$$

Do not adapt or learn $\lambda_k$ in this implementation. Keep relaxation
fixed.

### 7.5 Convergence statement for the learned solver

The exact Davis--Yin convergence result assumes a cocoercive forward operator. A neural approximation

$$
\widehat D_\theta(U,x_0)
$$

is not automatically the gradient of a convex smooth function and is not automatically cocoercive. Consequently, the exact TOS theorem cannot be claimed for the learned iteration without additional assumptions or an inexact-operator analysis.

Treat the learned method initially as a finite-depth unrolled solver. Possible later safeguards include:

- periodically replacing the learned projection with an exact projection;
- accepting a learned step only when a verified merit or residual decreases;
- bounding the learned-gradient error;
- training a convex smooth scalar Moreau model whose gradient is used by TOS;
- applying an exact final feasibility-correction layer.

These are research extensions, not assumptions that should be silently inserted into the first code version.

---

## 8. Suggested code organization

Adapt these names to the existing repository rather than duplicating already implemented functions.

```text
src/
    problem.jl
        system and MPC data structures
        single-shooting matrices
        condensed cost
        state-constraint right-hand side

    tos/
        operators.jl
            box_projection
            quadratic_cost_prox
            exact_state_projection
            exact_moreau_gradient

        exact_tos.jl
            exact Davis--Yin solver

        learned_tos.jl
            learned Davis--Yin solver

    learning/
        projection_model.jl
            neural architecture
            normalization

        affine_layer.jl
            factorization of Gx*Gx' + I
            equality correction

        losses.jl
            projection, gradient, slack, and rollout losses

        dataset.jl
            sample schema
            serialization
            train/validation/test split

scripts/
    generate_projection_dataset.jl
    collect_on_policy_queries.jl
    train_projection_model.jl
    evaluate_projection_model.jl
    benchmark_exact_tos.jl
    benchmark_learned_tos.jl

test/
    test_condensing.jl
    test_cost_prox.jl
    test_affine_layer.jl
    test_exact_moreau_gradient.jl
    test_exact_tos.jl
    test_learned_tos.jl
```

---

## 9. Required unit tests

### 9.1 Single-shooting equivalence

For random $(x_0,U)$, compare recursively simulated states with

$$
X=Gx_0+HU.
$$

Require agreement near machine precision.

### 9.2 State-constraint construction

Check that

$$
G_xU\le b_x(x_0)
$$

is equivalent to

$$
\underline X\le Gx_0+HU\le\overline X.
$$

### 9.3 Quadratic proximal map

For random $W$, verify

$$
(I+\gamma H_c)Y=W-\gamma q(x_0)
$$

and compare $Y$ with a generic convex solver.

### 9.4 Exact Moreau gradient

Compare

$$
D=\frac1\mu(U-V^\star)
$$

with finite differences of

$$
M_\mu^{\mathcal C}(U;x_0)
=\frac1{2\mu}\|U-V^\star\|^2.
$$

Avoid finite-difference points exactly at active-set transitions.

### 9.5 Affine correction

For random raw network outputs, require

$$
\|G_x\widetilde V+\widetilde s-b_x(x_0)\|
$$

to be near numerical precision. Separately report negative slack; do not confuse equality satisfaction with inequality feasibility.

### 9.6 Exact TOS solution

Compare the converged exact TOS solution against a trusted solver for the same **Moreau-smoothed** problem. Do not compare it only against the original hard-constrained MPC problem because they are different optimization problems.

### 9.7 Learned module accuracy

On the held-out test set, report:

$$
\|\widetilde V-V^\star\|,
\qquad
\|\widehat D-D^\star\|,
\qquad
\|[G_x\widetilde V-b_x(x_0)]_+\|,
$$

and

$$
\min_j\widetilde s_j.
$$

### 9.8 Closed-loop test

Run exact MPC, exact PGM, exact TOS, learned PGM, and learned TOS from identical initial conditions and references. Use the same warm-start policy and stopping budget whenever possible.

---

## 10. Benchmark protocol

At minimum, compare:

1. trusted QP MPC solver for the original hard-constrained problem;
2. exact PGM for the Moreau-smoothed problem;
3. exact TOS for the Moreau-smoothed problem;
4. learned PGM using the same learned projection module;
5. learned TOS using the same learned projection module.

The learned PGM and learned TOS comparison must use the same training data and model whenever possible. Otherwise an apparent algorithmic improvement may only reflect a better network.

Report:

- total wall-clock time;
- per-iteration time;
- iteration count;
- number and cost of exact projection calls;
- neural inference time;
- quadratic proximal-solve time;
- objective gap relative to the appropriate exact reference;
- fixed-point residual;
- input violation;
- maximum and normed state violation;
- correction-layer time, if used;
- final corrected objective and feasibility;
- closed-loop cumulative cost;
- closed-loop constraint violations;
- failure rate over test instances.

Run timing experiments after warm-up and separate CPU and GPU results. Small dense linear algebra and a small network may be slower on a GPU because of transfer and kernel-launch overhead.

### 10.1 Essential ablations

Perform the following ablations before increasing model complexity:

1. exact versus learned Moreau gradient;
2. PGM versus TOS with exact gradients;
3. PGM versus TOS with the same learned gradient;
4. broad-only training data versus broad plus on-policy data;
5. raw network output versus affine-corrected output;
6. projection loss alone versus added gradient/slack losses;
7. several initial/adaptive $\gamma$ schedules;
8. several fixed relaxation values;
9. cold start versus receding-horizon warm start;
10. performance before and after the final feasibility correction.

---

## 11. Implementation sequence and acceptance gates

### Phase 1: exact operators

- Reuse and verify the current single-shooting matrices.
- Implement the quadratic-cost proximal map.
- Reuse the exact state-projection QP.
- Implement exact Moreau-gradient evaluation.
- Implement exact TOS.

Acceptance gate: exact TOS converges to the trusted solution of the smoothed problem and all operator unit tests pass.

### Phase 2: dataset

- Generate broad, expert-perturbed, boundary, and exact-TOS on-policy queries.
- Store only `U_query`, `x0`, and `V_star`.
- Split by initial state or trajectory.

Acceptance gate: all retained projection labels are valid and no infeasible $\mathcal C(x_0)$ instance is silently labeled.

### Phase 3: learned projection

- Reuse the current $(U,x_0)\mapsto(\widehat V,\widehat s)$ network.
- Implement the differentiable affine correction using a factorized solve.
- Train and evaluate the corrected outputs.

Acceptance gate: corrected equality residuals are numerical, held-out projection/gradient errors meet a declared threshold, and negative-slack behavior is quantified.

### Phase 4: learned TOS

- Replace only the exact Moreau-gradient call.
- Keep box projection and quadratic proximal solve exact.
- Add per-iteration diagnostics.
- Collect learned-rollout queries and perform dataset aggregation.

Acceptance gate: learned TOS remains numerically stable across the complete test distribution and improves an end-to-end metric relative to learned PGM.

### Phase 5: hard-feasibility handling

- Add or reuse an exact final correction if hard state feasibility is required.
- Measure correction cost and objective degradation.

Acceptance gate: the final applied control sequence satisfies the stated feasibility tolerance, and the full pipeline remains competitive in runtime.

---

## 12. Key cautions

1. **Do not claim that Moreau smoothing preserves hard feasibility.** It replaces the state indicator with a finite smooth penalty.
2. **Do not claim the affine layer enforces the inequality.** It enforces $G_x\widetilde V+\widetilde s=b$; feasibility additionally requires $\widetilde s\ge0$.
3. **Do not claim exact Davis--Yin convergence for an arbitrary learned gradient.** The learned operator may not be cocoercive.
4. **Do not vary all parameters immediately.** Keep relaxation fixed and vary only $\gamma$.
5. **Do not train only on random points.** TOS evaluates the network on its own box-projected iterate distribution.
6. **Do not compare different problems as if they were identical.** Exact TOS solves the Moreau-smoothed problem; the trusted hard-constrained QP solves the original MPC problem.
7. **Do not form matrix inverses.** Use Cholesky or another appropriate factorization for both $I+\gamma H_c$ and $G_xG_x^\top+I$.
8. **Do not report neural prediction error alone.** The relevant outcomes are solver convergence, objective quality, feasibility, correction cost, runtime, and closed-loop behavior.

---

## 13. Minimal end-to-end data flow

### Offline training

```text
sample x0 and U queries
        |
        v
solve exact projection QP
        |
        v
store (U, x0, V*, s*, M*, D*)
        |
        v
train h_theta(U, x0) -> (Vhat, shat)
        |
        v
apply affine equality correction
        |
        v
evaluate (Vtilde, stilde, Dhat)
```

### Online learned TOS

```text
(x0, Zk)
    |
    v
Xk = box_projection(Zk)
    |
    v
learned projection + affine correction
    |
    v
Dhat_k = (Xk - Vtilde_k) / mu
    |
    v
quadratic_cost_prox(2Xk - Zk - gamma Dhat_k)
    |
    v
Yk and fixed-point residual Yk - Xk
    |
    v
Z{k+1} = Zk + lambda (Yk - Xk)
```

The first implementation should follow this flow exactly: fixed relaxation and
adaptive $\gamma$. Additional learned relaxation networks, quasi-Newton
directions, or safeguards should be introduced only after this baseline is
verified.
