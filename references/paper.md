\section{Problem Formulation}
\subsection{Linear MPC and single shooting}
Consider the discrete-time linear system
\begin{equation}
x_{k+1}=Ax_k+Bu_k,\qquad k=0,\ldots,N-1,
\end{equation}
where $x_k\in\mathbb{R}^{n_x}$, $u_k\in\mathbb{R}^{n_u}$, and $x_0=x_t$ is the measured state. The finite-horizon problem is
\begin{subequations}\label{eq:mpc}
\begin{align}
\minimize_{\{x_k,u_k\}}\quad &\sum_{k=0}^{N-1}\ell_k(x_k,u_k)+\ell_N(x_N),\\
\text{s.t.}\quad &x_{k+1}=Ax_k+Bu_k,\\
&\underline{x}\le x_k\le\bar{x},\quad k=1,\ldots,N,\\
&\underline{u}\le u_k\le\bar{u},\quad k=0,\ldots,N-1.
\end{align}
\end{subequations}
The functions \(\ell_k\) and \(\ell_N\) denote the smooth stage and terminal costs.

Repeated substitution gives
\begin{equation}
x_k=A^kx_0+\sum_{j=0}^{k-1}A^{k-1-j}Bu_j.
\end{equation}
Define $U=[u_0^\top,\ldots,u_{N-1}^\top]^\top$ and $X=[x_1^\top,\ldots,x_N^\top]^\top$. Then
\begin{equation}\label{eq:shooting}
X=G x_0+H U,
\end{equation}
where
\begin{equation}
G=\begin{bmatrix}A\\A^2\\\vdots\\A^N\end{bmatrix},\quad
H=\begin{bmatrix}
B&0&\cdots&0\\AB&B&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\A^{N-1}B&A^{N-2}B&\cdots&B
\end{bmatrix}.
\end{equation}
The condensed cost is
\begin{equation}
\ell(U;x_0)=\sum_{k=0}^{N-1}\ell_k(A^kx_0+S_kU,u_k)+\ell_N(A^Nx_0+S_NU),
\end{equation}
where $S_k$ is the appropriate block row of $H$. We assume $\ell(\cdot;x_0)$ is continuously differentiable with a Lipschitz gradient.

\subsection{Separate state and input constraint sets}
The input box is
\begin{equation}
\mathcal U=\{U\in\mathbb{R}^{Nn_u}\mid\underline U\le U\le\bar U\},\qquad g(U)=\delta_{\mathcal U}(U).
\end{equation}
Using \eqref{eq:shooting}, the state-feasible input set is
\begin{equation}\label{eq:C}
\mathcal C(x_0)=\{U\mid G_xU\le b_x(x_0)\},
\end{equation}
with
\begin{equation}
G_x=\begin{bmatrix}H\\-H\end{bmatrix},\qquad
b_x(x_0)=\begin{bmatrix}\bar X-G x_0\\-\underline X+G x_0\end{bmatrix}.
\end{equation}
Thus the exact condensed MPC problem is
\begin{equation}\label{eq:exact}
\min_U\;\ell(U;x_0)+\delta_{\mathcal C(x_0)}(U)+\delta_{\mathcal U}(U).
\end{equation}
Unlike a unified projection onto $\mathcal U\cap\mathcal C(x_0)$, we retain $\delta_{\mathcal U}$ because its proximal mapping is the inexpensive componentwise clipping operator.


\section{Methodology}

\subsection{Moreau-Smoothed State Constraints}

For $\mu>0$, the Moreau envelope of the state-constraint
indicator is defined as
\begin{align}
M_\mu^{\mathcal C}(U;x_0)
&=
\min_V
\left\{
\delta_{\mathcal C(x_0)}(V)
+
\frac{1}{2\mu}\|V-U\|^2
\right\} \\
&=
\frac{1}{2\mu}
\operatorname{dist}^2
\bigl(U,\mathcal C(x_0)\bigr).
\end{align}
Since $\mathcal C(x_0)$ is nonempty, closed, and convex,
$M_\mu^{\mathcal C}(\cdot;x_0)$ is continuously differentiable,
with gradient
\begin{equation}
\label{eq:MEgrad}
\nabla_U M_\mu^{\mathcal C}(U;x_0)
=
\frac{1}{\mu}
\left[
U-\mathcal P_{\mathcal C(x_0)}(U)
\right],
\end{equation}
where $\mathcal P_{\mathcal C(x_0)}(U)$ denotes the Euclidean
projection of $U$ onto $\mathcal C(x_0)$.

We then consider the smoothed composite problem
\begin{equation}
\label{eq:smoothprob}
\min_U
\;
f_\mu(U;x_0)+g(U),
\qquad
f_\mu
=
\ell+M_\mu^{\mathcal C},
\qquad
g
=
\delta_{\mathcal U}.
\end{equation}
For a stepsize $\gamma_k>0$, the corresponding proximal-gradient
iteration is
\begin{subequations}
\label{eq:pgm}
\begin{align}
Y^k
&=
U^k
-
\gamma_k
\left[
\nabla_U\ell(U^k;x_0)
+
\nabla_U M_\mu^{\mathcal C}(U^k;x_0)
\right],
\label{eq:pgm_forward}
\\
U^{k+1}
&=
\operatorname{prox}_{\gamma_k g}(Y^k)
=
\mathcal P_{\mathcal U}(Y^k).
\label{eq:pgm_backward}
\end{align}
\end{subequations}
Because $\mathcal U$ is a box, the backward step
\eqref{eq:pgm_backward} is evaluated by componentwise clipping.
The main computational bottleneck is therefore the evaluation of
\eqref{eq:MEgrad}, which requires solving
\begin{equation}
\label{eq:projection}
V^\star(U,x_0)
=
\underset{V}{\operatorname{argmin}}
\;
\frac{1}{2}\|V-U\|^2
\quad
\text{subject to}
\;
G_xV\le b_x(x_0).
\end{equation}

\subsection{Learned Neural Projection}

\subsubsection{Slack-variable reformulation}

To represent the inequality constraints, introduce a slack variable
$s\in\mathbb R^{2Nn_x}$. Problem \eqref{eq:projection} can then be
written equivalently as
\begin{subequations}
\label{eq:slackqp}
\begin{align}
\underset{V,s}{\operatorname{minimize}}
\quad&
\frac{1}{2}\|V-U\|^2,
\label{eq:slackqp_obj}
\\
\operatorname{subject\ to}
\quad&
G_xV+s=b_x(x_0),
\label{eq:slackqp_eq}
\\
&
s\ge0.
\label{eq:slackqp_nonneg}
\end{align}
\end{subequations}
For each training sample $(U_i,x_{0,i})$, the exact projection
$V_i^\star$ is obtained by solving \eqref{eq:projection}. Its
associated optimal slack is
\begin{equation}
\label{eq:optimal_slack}
s_i^\star
=
b_x(x_{0,i})-G_xV_i^\star
\ge0.
\end{equation}

\subsubsection{Neural approximation}

Let $h_\theta$ be a neural network parameterized by $\theta$.
Given the projection point $U$ and the initial state $x_0$, the
network predicts both the projected input and its associated slack:
\begin{equation}
\label{eq:nn_prediction}
\hat z
=
h_\theta(U,x_0)
=
\begin{bmatrix}
\hat V\\
\hat s
\end{bmatrix}.
\end{equation}
The raw network output is not required to satisfy the equality
constraint exactly. We therefore train the network using the loss
\begin{align}
\mathcal L(\theta)
=
\frac{1}{N_s}
\sum_{i=1}^{N_s}
\Bigl(
&
\|\hat V_i-V_i^\star\|^2
\nonumber\\
&
+
\lambda_{\mathrm{eq}}
\|
G_x\hat V_i+\hat s_i-b_x(x_{0,i})
\|^2
\nonumber\\
&
+
\lambda_+
\|[-\hat s_i]_+\|^2
\Bigr),
\label{eq:learning_loss}
\end{align}
where $\lambda_{\mathrm{eq}}\ge0$ and $\lambda_+\ge0$ are penalty
parameters.

The first term trains the network to approximate the exact Euclidean
projection. The second term promotes consistency with the
slack-variable equality, whereas the third term penalizes negative
slack components. Importantly, these penalty terms encourage, but do
not guarantee, satisfaction of the corresponding constraints.

% If the optimal slack labels \eqref{eq:optimal_slack} are retained, an
% additional supervised slack loss may be included:
% \begin{equation}
% \label{eq:slack_loss}
% \mathcal L_s(\theta)
% =
% \frac{\lambda_s}{N_s}
% \sum_{i=1}^{N_s}
% \|\hat s_i-s_i^\star\|^2.
% \end{equation}
% This term is optional because the quantity required by the
% proximal-gradient iteration is $V^\star$. Nevertheless, it may improve
% the accuracy of the predicted slack and reduce violations after the
% subsequent affine projection.

\subsubsection{Closed-form affine projection layer}

Although the equality residual is included in
\eqref{eq:learning_loss}, a finite penalty parameter cannot enforce
the equality exactly. We therefore append a closed-form affine
projection layer to the neural-network output.

Define
\begin{equation}
\label{eq:z_E_b}
z
=
\begin{bmatrix}
V\\
s
\end{bmatrix},
\qquad
E
=
\begin{bmatrix}
G_x & I
\end{bmatrix},
\qquad
b=b_x(x_0).
\end{equation}
The equality constraint \eqref{eq:slackqp_eq} is equivalently written
as
\begin{equation}
\label{eq:affine_constraint}
Ez=b.
\end{equation}
Given the raw prediction $\hat z$, the affine projection layer solves
\begin{equation}
\label{eq:affine_projection_problem}
\tilde z
=
\underset{z}{\operatorname{argmin}}
\;
\frac{1}{2}\|z-\hat z\|^2
\quad
\text{subject to}
\quad
Ez=b.
\end{equation}
Since $E$ contains an identity block, it has full row rank and
\begin{equation}
\label{eq:EET}
EE^\top
=
G_xG_x^\top+I
\succ0.
\end{equation}
Consequently, \eqref{eq:affine_projection_problem} admits the
closed-form solution
\begin{equation}
\label{eq:affproj}
\tilde z
=
\hat z
-
E^\top
(EE^\top)^{-1}
(E\hat z-b).
\end{equation}
Writing
\begin{equation}
\tilde z
=
\begin{bmatrix}
\tilde V\\
\tilde s
\end{bmatrix},
\end{equation}
the output of the affine projection layer satisfies
\begin{equation}
\label{eq:exact_equality}
G_x\tilde V+\tilde s=b_x(x_0)
\end{equation}
up to numerical precision.

For a fixed system model and prediction horizon, $E$ is constant.
Therefore, a factorization of $EE^\top$ can be computed offline and
reused for all MPC instances. Moreover, the inverse in
\eqref{eq:affproj} need not be formed explicitly; the affine layer can
be evaluated by solving a linear system with the precomputed
factorization.

\subsubsection{Learned Moreau-envelope gradient}

The corrected primal component $\tilde V$ is used as an approximation
of the exact projection:
\begin{equation}
\mathcal P_{\mathcal C(x_0)}(U)
\approx
\tilde V.
\end{equation}
The resulting learned approximation of the Moreau-envelope gradient
is
\begin{equation}
\label{eq:learnedgrad}
\nabla_U \hat{M}_\mu^{\mathcal C}(U;x_0)
=
\frac{1}{\mu}
\left(
U-\tilde V
\right).
\end{equation}
This approximation replaces the exact gradient
\eqref{eq:MEgrad} in the forward step \eqref{eq:pgm_forward}.
