::: IEEEkeywords
:::

# Problem Formulation {#sec:problem_formulation}

## Linear MPC

Consider the discrete-time linear system $$x_{k+1}=Ax_k+Bu_k,
    \qquad k=0,\ldots,N-1,
    \label{eq:dynamics}$$ where $x_k\in\mathbb{R}^{n_x}$,
$u_k\in\mathbb{R}^{n_u}$, and the initial state is $x_0=x_t$ with the
current state $x_t$.

The finite-horizon MPC problem is $$\label{eq:original_mpc}
\begin{align}
    \underset{\{x_k,u_k\}}{\operatorname{minimize}}\quad
    &\sum_{k=0}^{N-1} \ell_k(x_k,u_k)+\ell_N(x_N)
    \label{eq:original_objective}\\
    \operatorname{subject\ to}\quad
    &x_0=x_t, \\
    &x_{k+1}=Ax_k+Bu_k,\quad k=0,\ldots,N-1,\\
    &\underline{x}\leq x_k\leq\overline{x},
    \quad k=1,\ldots,N,\\
    &\underline{u}\leq u_k\leq\overline{u},
    \quad k=0,\ldots,N-1.
\end{align}$$

The functions $\ell_k$ and $\ell_N$ represent the smooth stage and
terminal costs, respectively.

## State elimination

Repeated substitution of
[\[eq:dynamics\]](#eq:dynamics){reference-type="eqref"
reference="eq:dynamics"} gives
$$x_k=A^kx_0+\sum_{j=0}^{k-1}A^{k-1-j}Bu_j,
    \qquad k=1,\ldots,N.
    \label{eq:state_expansion}$$ Define the stacked input and
predicted-state vectors $$U=
    \begin{bmatrix}
        u_0\\u_1\\\vdots\\u_{N-1}
    \end{bmatrix}
    \in\mathbb{R}^{Nn_u},
    \qquad
    X=
    \begin{bmatrix}
        x_1\\x_2\\\vdots\\x_N
    \end{bmatrix}
    \in\mathbb{R}^{Nn_x}.
    \label{eq:stacked_vectors}$$

Let denote $$S_k=
    \begin{bmatrix}
        A^{k-1}B&A^{k-2}B&\cdots&B&
        0&\cdots&0
    \end{bmatrix}
    \in\mathbb{R}^{n_x\times Nn_u},
    \label{eq:Sk}$$ where the first $k$ block columns are nonzero. Then
$$x_k=A^kx_0+S_kU.
    \label{eq:single__state}$$ For convenience, define $S_0=0$, so that
$x_0=A^0x_0+S_0U$.

Equivalently, the stacked predicted states satisfy
$$X=\mathcal{A}x_0+\mathcal{B}U,
    \label{eq:stacked_dynamics}$$ with $$\mathcal{A}=
    \begin{bmatrix}
        A\\A^2\\\vdots\\A^N
    \end{bmatrix},
    \qquad
    \mathcal{B}=
    \begin{bmatrix}
        B&0&\cdots&0\\
        AB&B&\cdots&0\\
        \vdots&\vdots&\ddots&\vdots\\
        A^{N-1}B&A^{N-2}B&\cdots&B
    \end{bmatrix}.
    \label{eq:prediction_matrices}$$

The nominal cost is rewritten as $$\ell(U;x_0)
    =
    \sum_{k=0}^{N-1}
    \ell_k\!\left(A^kx_0+S_kU,u_k\right)
    +
    \ell_N\!\left(A^Nx_0+S_NU\right).$$

The MPC
problem [\[eq:original_mpc\]](#eq:original_mpc){reference-type="eqref"
reference="eq:original_mpc"} can be written in the dense form
$$\begin{align}
        \underset{U}{\operatorname{minimize}}
        \quad &
        \ell(U;x_0)\\
        \text{subject to}
        \quad &
        \underline{X}
        \leq
        \mathcal{A}x_0+\mathcal{B}U
        \leq
        \overline{X},\\
        &
        \underline{U}
        \leq U
        \leq
        \overline{U}.
    \end{align}
    \label{eq:_mpc}$$ Define the admissible input set as $$\mathcal{U}
    :=
    \left\{
        U\in\mathbb{R}^{Nn_u}
        \,\middle|\,
        \underline{U}\leq U\leq\overline{U}
    \right\}.
    \label{eq:admissible_input_set}$$

The input constraints are represented by the extended-value constraint
function $$g(U)
    :=
    \delta_{\mathcal U}(U)
    =
    \begin{cases}
        0,
        & U\in\mathcal U,\\
        +\infty,
        & U\notin\mathcal U.
    \end{cases}
    \label{eq:input_constraint_function}$$

Because $\mathcal U$ is a nonempty, closed, and convex box, $g$ is
proper, closed, and convex.

Separately, define the state-feasible input set $$\mathcal{C}_x(x_0)
    :=
    \left\{
        U\in\mathbb{R}^{Nn_u}
        \,\middle|\,
        \underline{X}
        \leq
        \mathcal{A}x_0+\mathcal{B}U
        \leq
        \overline{X}
    \right\}.
    \label{eq:state_feasible_set}$$

The MPC problem can then be written as $$\label{eq:composite__mpc}
\begin{align}
    \underset{U}{\operatorname{minimize}}
    \quad&
    \ell(U;x_0)+g(U)
    \label{eq:composite__objective}\\
    \operatorname{subject\ to}
    \quad&
    U\in\mathcal{C}_x(x_0).
    \label{eq:composite_state_constraint}
\end{align}$$

The state constraints can equivalently be written as
$$G_xU\leq b_x(x_0),
    \label{eq:state_inequality}$$ where $$G_x=
    \begin{bmatrix}
        \mathcal{B}\\
        -\mathcal{B}
    \end{bmatrix},
    \qquad
    b_x(x_0)=
    \begin{bmatrix}
        \overline{X}-\mathcal{A}x_0\\
        -\underline{X}+\mathcal{A}x_0
    \end{bmatrix}.
    \label{eq:state_constraint_matrices}$$

Consequently, $$\mathcal{C}_x(x_0)
    =
    \left\{
        U
        \,\middle|\,
        G_xU\leq b_x(x_0)
    \right\}.
    \label{eq:state_feasible_polyhedron}$$

## Constrained proximal-gradient iteration

Define the complete nonsmooth constrained term as $$\phi(U;x_0)
    :=
    g(U)+\delta_{\mathcal{C}_x(x_0)}(U).
    \label{eq:complete_proximal_function}$$

The roles of the two terms
in [\[eq:complete_proximal_function\]](#eq:complete_proximal_function){reference-type="eqref"
reference="eq:complete_proximal_function"} are different:
$g(U) = \delta_{\mathcal U}(U)$ represents the input constraints;
$\delta_{\mathcal{C}_x(x_0)}(U)$ represents the state constraints.

Their sum is the indicator of the complete feasible set: $$\phi(U;x_0)
    =
    \delta_{\mathcal U\cap\mathcal{C}_x(x_0)}(U).
    \label{eq:combined_indicator}$$

The last condition ensures that $\phi(\cdot;x_0)$ is proper. Because
both sets are closed and convex, $\phi(\cdot;x_0)$ is proper, closed,
and convex.

For a step size $\alpha>0$, the proximal-gradient iteration is $$U^{k+1}
    =
    \operatorname{prox}_{\alpha\phi}
    \left(
        U^k-\alpha\nabla_U\ell(U^k;x_0)
    \right).
    \label{eq:proximal_gradient_iteration}$$

Since $\phi$ is the indicator of the complete feasible set,
[\[eq:proximal_gradient_iteration\]](#eq:proximal_gradient_iteration){reference-type="eqref"
reference="eq:proximal_gradient_iteration"} becomes $$U^{k+1}
    =
    \Pi_{\mathcal U\cap\mathcal{C}_x(x_0)}
    \left(
        U^k-\alpha\nabla_U\ell(U^k;x_0)
    \right).
    \label{eq:projected_gradient_iteration}$$

Thus, the proximal step projects the gradient update onto the set
satisfying both the input and state constraints.

## Moreau-envelope representation

The Moreau envelope of $\phi(\cdot;x_0)$ with parameter $\alpha>0$ is
$$\begin{aligned}
    \phi^\alpha(W;x_0)
    :={}&
    \underset{Y\in\mathbb{R}^{Nn_u}}
    {\operatorname{min}}
    \left\{
        \phi(Y;x_0)
        +
        \frac{1}{2\alpha}
        \left\|Y-W\right\|_2^2
    \right\}
    \label{eq:moreau_envelope_definition}\\
    ={}&
    \underset{
        Y\in\mathcal U\cap\mathcal{C}_x(x_0)
    }{\operatorname{min}}
    \frac{1}{2\alpha}
    \left\|Y-W\right\|_2^2.
    \label{eq:moreau_projection_problem}
\end{aligned}$$

Because $\phi$ is the indicator of the feasible set, its Moreau envelope
is the scaled squared distance to that set: $$\phi^\alpha(W;x_0)
    =
    \frac{1}{2\alpha}
    \operatorname{dist}^2_{
        \mathcal U\cap\mathcal{C}_x(x_0)
    }(W).
    \label{eq:moreau_distance}$$

Since $\phi(\cdot;x_0)$ is proper, closed, and convex, its Moreau
envelope is continuously differentiable, with
$$\nabla_W\phi^\alpha(W;x_0)
    =
    \frac{1}{\alpha}
    \left[
        W-
        \operatorname{prox}_{\alpha\phi(\cdot;x_0)}(W)
    \right].
    \label{eq:moreau_gradient_identity}$$

Therefore, $$\operatorname{prox}_{\alpha\phi(\cdot;x_0)}(W)
    =
    W-\alpha\nabla_W\phi^\alpha(W;x_0).
    \label{eq:proximal_from_moreau}$$

Substituting [\[eq:proximal_from_moreau\]](#eq:proximal_from_moreau){reference-type="eqref"
reference="eq:proximal_from_moreau"} into
[\[eq:proximal_gradient_iteration\]](#eq:proximal_gradient_iteration){reference-type="eqref"
reference="eq:proximal_gradient_iteration"} gives $$\begin{aligned}
    U^{k+1}
    ={}&
    U^k-\alpha\nabla_U\ell(U^k;x_0)
    \nonumber\\
    &-
    \alpha\nabla_W\phi^\alpha
    \left(
        U^k-\alpha\nabla_U\ell(U^k;x_0);
        x_0
    \right).
    \label{eq:moreau_proximal_gradient}
\end{aligned}$$

:::: algorithm
::: algorithmic
Initial state $x_0$, input sequence
$U=\operatorname{col}(u_0,\ldots,u_{N-1})$, matrices $A$ and $B$

Cost $\ell(U;x_0)$ and gradient $\nabla_U\ell(U;x_0)$

$\ell(U;x_0)\gets 0$

$\ell(U;x_0)
    \gets
    \ell(U;x_0)+\ell_k(x_k,u_k)$

$x_{k+1}\gets Ax_k+Bu_k$

$\ell(U;x_0)
\gets
\ell(U;x_0)+\ell_N(x_N)$

$p_N\gets\nabla_{x_N}\ell_N(x_N)$

$\nabla_{u_k}\ell(U;x_0)
    \gets
    \nabla_{u_k}\ell_k(x_k,u_k)
    +B^\top p_{k+1}$

$p_k
    \gets
    \nabla_{x_k}\ell_k(x_k,u_k)
    +A^\top p_{k+1}$

$\nabla_U\ell(U;x_0)
\gets
\operatorname{col}\!\left(
\nabla_{u_0}\ell(U;x_0),\ldots,
\nabla_{u_{N-1}}\ell(U;x_0)
\right)$

**return** $\ell(U;x_0)$ and $\nabla_U\ell(U;x_0)$
:::
::::

# Learning the Moreau-Envelope Gradient {#sec:learning_moreau}

This section presents a learning-based implementation of the constrained
proximal-gradient iterations developed in
Section [1](#sec:problem_formulation){reference-type="ref"
reference="sec:problem_formulation"}. The proposed architecture learns
the Moreau envelope of the constrained term using an input-convex neural
network (ICNN) and obtains the corresponding gradient through automatic
differentiation. The resulting learned gradient is embedded in a
finite-depth unrolled proximal-gradient architecture.

## Motivation {#subsec:learning_motivation}

Recall that the MPC problem is written as
$$\underset{U\in\mathbb{R}^{Nn_u}}{\operatorname{minimize}}
    \quad
    \ell(U;x_0)+\phi(U;x_0),
    \label{eq:learning_composite_problem}$$ where $\ell(\cdot;x_0)$ is
the smooth MPC cost and $$\phi(U;x_0)
    =
    g(U)+\delta_{\mathcal{C}_x(x_0)}(U)
    =
    \delta_{\mathcal{F}(x_0)}(U),
    \label{eq:learning_phi}$$ with $$\mathcal{F}(x_0)
    :=
    \mathcal{U}\cap\mathcal{C}_x(x_0)$$ denoting the complete feasible
set.

A standard proximal-gradient iteration for
[\[eq:learning_composite_problem\]](#eq:learning_composite_problem){reference-type="eqref"
reference="eq:learning_composite_problem"} is
$$\label{eq:exact_pg_decomposition}
\begin{align}
    W^k
    &=
    U^k
    -
    \alpha_k
    \nabla_U\ell(U^k;x_0),
    \label{eq:exact_forward_step}\\
    U^{k+1}
    &=
    \operatorname{prox}_{
        \alpha_k\phi(\cdot;x_0)
    }(W^k),
    \label{eq:exact_backward_step}
\end{align}$$ where $\alpha_k>0$ is the step size.

Forward--backward splitting is attractive because each iteration uses
only the gradient of the smooth cost and the proximal mapping of the
nonsmooth term. Nevertheless, as discussed in the PANOC, first-order
forward--backward iterations can exhibit slow tail convergence,
particularly for ill-conditioned optimal-control problems and long
prediction horizons. Consequently, a large number of iterations may be
required to obtain an accurate solution.

In addition, the proximal mapping in
[\[eq:exact_backward_step\]](#eq:exact_backward_step){reference-type="eqref"
reference="eq:exact_backward_step"} is not merely a projection onto the
input box.

Evaluating the proximal mapping
$\operatorname{prox}_{\alpha_k\phi(\cdot;x_0)}(W^k)$ generally requires
solving a state-dependent constrained quadratic program. This
computation must be repeated because $W^k$ changes across iterations and
the feasible set $\mathcal{F}(x_0)$ changes with the initial state. The
total computational burden can therefore be enormous.

The purpose of the proposed learning architecture is to reduce this
burden by replacing the repeated calculation with evaluations of a
trained neural model. The method does not assume that proximal-gradient
iterations fail to converge.

## Conditional input-convex approximation {#subsec:conditional_icnn}

Let $$\widehat{\phi}_{\theta_k}^{\alpha_k}(W;x_0)
    \approx
    \phi^{\alpha_k}(W;x_0)
    \label{eq:learned_envelope_approximation}$$ denote the learned
Moreau envelope at iteration $k$, where $\theta_k$ collects the
trainable network parameters.

The network is constructed to be convex with respect to $W$, while $x_0$
is treated as a contextual parameter. A conditional ICNN layer can be
expressed as $$\label{eq:conditional_icnn_layers}
\begin{align}
    &z_{k,0}
    =
    \sigma_{k,0}
    \left(
        A_{k,0}W
        +
        C_{k,0}x_0
        +
        b_{k,0}
    \right),
    \label{eq:icnn_first_layer}\\
    &z_{k,j+1}
    =
    \sigma_{k,j+1}
    \left(
        Z_{k,j}z_{k,j}
        +
        A_{k,j+1}W
        +
        C_{k,j+1}x_0
        +
        b_{k,j+1}
    \right),
    \label{eq:icnn_hidden_layer}\\
    &\widehat{\phi}_{\theta_k}^{\alpha_k}(W;x_0)
    =
    z_{k,J}.
    \label{eq:icnn_output}
\end{align}$$

Convexity with respect to $W$ is preserved by imposing $$Z_{k,j}\geq 0
    \quad\text{elementwise},
    \qquad j=0,\ldots,J-1,
    \label{eq:icnn_nonnegative_weights}$$ and selecting convex,
nondecreasing activation functions $\sigma_{k,j}$. Smooth activations,
such as the softplus function, are used so that the learned envelope is
differentiable with respect to $W$.

The learned Moreau-envelope gradient is obtained through automatic
differentiation: $$\widehat{d}_{\theta_k}(W;x_0)
    :=
    \nabla_W
    \widehat{\phi}_{\theta_k}^{\alpha_k}(W;x_0).
    \label{eq:learned_moreau_gradient}$$

The corresponding learned proximal mapping is
$$\widehat{\mathcal{P}}_{\theta_k}(W;x_0)
    :=
    W
    -
    \alpha_k
    \widehat{d}_{\theta_k}(W;x_0).
    \label{eq:learned_proximal_mapping}$$

## Unrolled iteration-dependent architecture {#subsec:unrolled_architecture}

The proposed architecture unrolls $K$ proximal-gradient iterations. Each
unrolled layer contains three operations:

1.  evaluation of the smooth MPC gradient;

2.  computation of the forward point $W^k$;

3.  evaluation of an iteration-dependent learned Moreau-envelope
    gradient.

Specifically, layer $k$ performs $$\label{eq:learned_unrolled_layer}
\begin{align}
    D^k
    &=
    \nabla_U\ell(U^k;x_0),
    \label{eq:layer_smooth_gradient}\\
    W^k
    &=
    U^k-\alpha_kD^k,
    \label{eq:layer_forward_point}\\
    \widehat{d}^k
    &=
    \nabla_W
    \widehat{\phi}_{\theta_k}^{\alpha_k}(W^k;x_0),
    \label{eq:layer_envelope_gradient}\\
    U^{k+1}
    &=
    W^k-\alpha_k\widehat{d}^k.
    \label{eq:layer_learned_update}
\end{align}$$

Equivalently, $$\begin{aligned}
    U^{k+1}
    ={}&
    U^k
    -
    \alpha_k\nabla_U\ell(U^k;x_0)
    \nonumber\\
    &-
    \alpha_k
    \nabla_W
    \widehat{\phi}_{\theta_k}^{\alpha_k}
    \left(
        U^k
        -
        \alpha_k\nabla_U\ell(U^k;x_0);
        x_0
    \right).
    \label{eq:complete_learned_update}
\end{aligned}$$
