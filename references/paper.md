\documentclass[journal,twoside,web]{ieeecolor}
\usepackage{xcolor}
\usepackage{algorithm, algpseudocode}
\usepackage{lcsys}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage[matmatrix, opt, reviewmark]{icpslab}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\markboth{\journalname, VOL. XX, NO. XX, XXXX 2017}
{Author \MakeLowercase{\textit{et al.}}: Learned Dual-Reduced Projections for
Safeguarded Proximal-Gradient MPC}
\begin{document}
\title{Learned Dual-Reduced Projections for Safeguarded
Proximal-Gradient Model Predictive Control}
\author{First A. Author, Second B. Author, and Third C. Author}

\maketitle

\begin{abstract}
% We propose a learning-augmented proximal-gradient method for linear
% model predictive control (MPC) in which the computationally expensive
% constrained proximal mapping is reduced, via Lagrangian duality, to an
% orthant-constrained problem in the multiplier variable. The reduced
% problem admits matrix-free evaluation through forward rollouts and
% backward adjoint sweeps, and its associated dual value function is
% convex and coordinate-wise nondecreasing in the constraint-violation
% vector. A monotone input-convex neural network (ICNN) is trained to
% approximate this dual value function, and its gradient supplies a
% nonnegative multiplier initialization that warm-starts a small number
% of dual projected-gradient iterations. The resulting inexact proximal
% mapping is embedded in an outer forward--backward envelope (FBE)
% safeguarded iteration, guaranteeing global convergence irrespective of
% the learned model's accuracy while allowing the learned initialization
% to reduce the number of inner iterations required in practice.
\end{abstract}

\begin{IEEEkeywords}
Model predictive control, proximal-gradient methods, Moreau envelope,
input-convex neural networks, learning to optimize.
\end{IEEEkeywords}

\section{Problem Formulation}\label{sec:problem_formulation}

\subsection{Linear MPC}

Consider the discrete-time linear system
\begin{equation}
    x_{k+1}=Ax_k+Bu_k,
    \qquad k=0,\ldots,N-1,
    \label{eq:dynamics}
\end{equation}
where \(x_k\in\mathbb{R}^{n_x}\), \(u_k\in\mathbb{R}^{n_u}\), and the
initial state is \(x_0=x_t\) with the current measured state $x_t$.
The finite-horizon MPC problem is
\begin{subequations}
\label{eq:original_mpc}
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
\end{align}
\end{subequations}
The functions \(\ell_k\) and \(\ell_N\) denote the smooth stage and
terminal costs.

\subsection{State elimination and single shooting}

Repeated substitution of \eqref{eq:dynamics} gives
\begin{equation}
    x_k=A^kx_0+\sum_{j=0}^{k-1}A^{k-1-j}Bu_j,
    \qquad k=1,\ldots,N.
    \label{eq:state_expansion}
\end{equation}
Define the stacked input and state vectors
\begin{equation}
    U=
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
    \label{eq:stacked_vectors}
\end{equation}
Then
\begin{equation}
    X=\mathcal{A}x_0+\mathcal{B}U,
    \label{eq:stacked_dynamics}
\end{equation}
with
\begin{equation}
    \mathcal{A}=
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
    \label{eq:prediction_matrices}
\end{equation}
The nominal cost is
\begin{equation}
    \ell(U;x_0)
    =
    \sum_{k=0}^{N-1}
    \ell_k\!\left(A^kx_0+S_kU,u_k\right)
    +
    \ell_N\!\left(A^Nx_0+S_NU\right),
\end{equation}
where $S_k$ denotes the $k$-th block-row of $\mathcal B$.
Throughout, $\ell(\cdot\,;x_0)$ is assumed continuously differentiable
with $L_\ell$-Lipschitz gradient, uniformly in $x_0$ over the region of
interest.

\subsection{Constraint sets}

The admissible input box is
\begin{equation}
    \mathcal{U}
    :=
    \left\{
        U\in\mathbb{R}^{Nn_u}
        \,\middle|\,
        \underline{U}\leq U\leq\overline{U}
    \right\},
    \qquad
    g(U):=\delta_{\mathcal U}(U).
    \label{eq:admissible_input_set}
\end{equation}
The state-feasible input set is
\begin{equation}
    \mathcal{C}_x(x_0)
    :=
    \left\{
        U
        \,\middle|\,
        G_xU\leq b_x(x_0)
    \right\},
    \label{eq:state_feasible_set}
\end{equation}
with
\begin{equation}
    G_x=
    \begin{bmatrix}
        \mathcal{B}\\-\mathcal{B}
    \end{bmatrix},
    \qquad
    b_x(x_0)=
    \begin{bmatrix}
        \overline{X}-\mathcal{A}x_0\\
        -\underline{X}+\mathcal{A}x_0
    \end{bmatrix}.
    \label{eq:state_constraint_matrices}
\end{equation}
The complete feasible set is $\mathcal F(x_0):=\mathcal U\cap\mathcal
C_x(x_0)$, and it is convenient, for the dual derivation below, to
represent it in the single unified form
\begin{equation}
    \mathcal F(x_0)=\{U\mid GU\leq b(x_0)\},
    \;
    G=\begin{bmatrix}G_x\\I\\-I\end{bmatrix},
    \;
    b(x_0)=\begin{bmatrix}b_x(x_0)\\\overline U\\-\underline U\end{bmatrix}.
    \label{eq:unified_polyhedron}
\end{equation}
The nonsmooth term is
$\phi(U;x_0):=g(U)+\delta_{\mathcal C_x(x_0)}(U)=\delta_{\mathcal
F(x_0)}(U)$, which, by \eqref{eq:unified_polyhedron}, is itself an
indicator of a polyhedron $\{GU\le b(x_0)\}$. 
% This unification is a
% notational convenience only: Sections~\ref{sec:dual_reduction}
% and~\ref{sec:learning_dual} treat $G$ generically and remain valid if
% the box is instead retained in closed form and handled separately
% (Remark~\ref{rem:box_separate}).

\subsection{Constrained proximal-gradient iteration}

Because $\mathcal F(x_0)$ is nonempty, closed, and convex whenever the
MPC problem is feasible, $\phi(\cdot\,;x_0)$ is proper, closed, and
convex. For a step size $\alpha>0$, the proximal-gradient iteration is
\begin{align}
\label{eq:projected_gradient_iteration}
    U^{k+1}
    &=\Pi_{\mathcal F(x_0)}\!\left(U^k-\alpha\nabla_U\ell(U^k;x_0)\right)
     \nonumber\\
    &=\operatorname{prox}_{\gamma\phi(\cdot\,;x_0)}
    \left(
        U^k-\alpha\nabla_U\ell(U^k;x_0)
    \right).
\end{align}

\subsection{Moreau-envelope representation}

The Moreau envelope of $\phi(\cdot\,;x_0)$ with parameter $\gamma>0$ is
\begin{equation}
    \phi^\gamma(q;x_0)
    =
    \frac{1}{2\gamma}
    \operatorname{dist}^2_{\mathcal F(x_0)}(q),
    \label{eq:moreau_distance}
\end{equation}
which is continuously differentiable with
\begin{subequations}
\begin{align}
    \nabla_q\phi^\gamma(q;x_0)
    &=
    \frac{1}{\gamma}
    \left[
        q-\Pi_{\mathcal F(x_0)}(q)
    \right],
    \\
    \operatorname{prox}_{\gamma\phi(\cdot\,;x_0)}(q)
    &=
    q-\gamma\nabla_q\phi^\gamma(q;x_0).
\end{align}
    \label{eq:moreau_gradient_identity}
\end{subequations}
Every constrained proximal-gradient iteration therefore requires
evaluating $\Pi_{\mathcal F(x_0)}(q)$ for the current forward point
$q=U^k-\alpha\nabla_U\ell(U^k;x_0)$. 

\section{Discussion}
The bottleneck in every PGM iteration is $\Pi_{\mathcal F(x_0)}(q)$, i.e. evaluating $\phi_\gamma$ and its gradient. 
Fig.~\ref{fig:pgm_time} showcase that the execution time of forward step is smaller than backward step in PGM.
Note $\phi_\gamma$ is convex in $q$ for fixed $\gamma$, and (as an indicator's envelope) monotonically \emph{nonincreasing} in $\gamma$.
Also, the minimizer $\Pi_{\mathcal F(x_0)}(q)$ itself does not depend on $\gamma$ (only the envelope value/curvature does). These two structural facts (convexity in $q$, monotonicity in $\gamma$) are what we intend to exploit in the learned model.
\begin{figure}
    \centering\includegraphics[width=\linewidth]{figs/pgm_time.png}
    \caption{Caption}
    \label{fig:pgm_time}
\end{figure}


We learn $\phi_\gamma(q;x_0)$ with a network that is \emph{convex in $q$ by construction}, and obtain the learned proximal step by differentiating through the network:
\[
\widehat{U} \;=\; q - \alpha\,\nabla_q \widehat\phi_\gamma(q; x_0).
\]
Since convexity and (if desired) monotonicity in $\gamma$ are architectural guarantees, the learned surrogate is easier to make well-behaved, and its gradient is cheap compared to solving the QP-like projection exactly.

\subsection{Architecture: parametric convex function network}
We use the neural network architecture from~\cite{schaller2025learning}: for input $q$ and parameter $\theta$,
\begin{align}
z^0 &= q,\\ 
z^l &= \varphi\big(W_l(\theta) z^{l-1} + V_l(\theta)q + \omega_l(\theta)\big),\, l=1,\dots,L-1, \\
\widehat\phi_\gamma &= W_L(\theta) z^{L-1} + V_L(\theta)q + \omega_L(\theta),
\end{align}
with $\varphi$ convex nondecreasing (ReLU/softplus) and $W_l(\theta) \ge 0$ enforced by construction, guaranteeing convexity of $\widehat\phi_\gamma$ in $q$ for every $\theta$. In our setting:
\begin{itemize}
  \item The variable is $q$ (the PGM forward point); the parameter is $\theta = x_0$ (fixed $\gamma$) or $\theta = (x_0,\gamma)$ (adaptive $\gamma$, see \S\ref{sec:gamma}).
  \item Weights $(W_l,V_l,\omega_l)$ are produced by a hypernetwork $\psi(\theta)$, as in~\cite{schaller2025learning}.
\end{itemize}

\textbf{Loss function}: Beyond a regression term on $\phi_\gamma$ and $\nabla_W\phi_\gamma$, we add a soft-constraint penalty on the induced estimate $\widehat U = q - \alpha\nabla_q\widehat\phi_\gamma(q;x_0)$ so that training steers $\widehat U$ toward feasible set, not just toward matching envelope values:
\begin{align}
\mathcal{L} &= \underbrace{(\widehat\phi_\gamma(q;x_0) - \nabla_q\phi_\gamma(q;x_0)\big)^2}_{\text{ME-matching}} \nonumber\\
&+\underbrace{\big\|\nabla_q\widehat\phi_\gamma(q;x_0) - \nabla_q\phi_\gamma(q;x_0)\big\|^2}_{\text{gradient-matching}} \nonumber\\
&+\lambda \underbrace{\big\|\mathrm{ReLU}(G\widehat U - b(x_0))\big\|^2}_{\text{soft feasibility penalty}}.
\end{align}
This penalty does not guarantee feasibility, hence the correction layer is used as below.

\subsection{Guaranteeing feasibility: HardNet-Aff correction}
Since the learned $\widehat U$ can violate $GU \le b(x_0)$, we append the closed-form affine correction layer from HardNet-Aff~\cite{min2024hardnet}:
\[
P(\widehat U) = \widehat U + G^{+}\big[\mathrm{ReLU}(b(x_0) - G\widehat U)\big]_{-},
\]
where $G^+=G^\top(GG^\top)^−1$.

\paragraph{Two placements to test.}
\begin{enumerate}[nosep]
  \item \textbf{Per-iteration correction:} $U^{k+1} = P\big(W^k - \alpha\nabla_W\widehat\phi_\gamma(W^k;x_0)\big)$ applied at every unrolled PGM step. Keeps every iterate feasible; more calls to $P$, and interacts with the learned gradient at every step (may change the effective dynamics the network was trained under).
  \item \textbf{Final-iteration correction only:} run $K$ unrolled steps with the raw learned update, apply $P$ once at the end. Cheaper, but intermediate iterates are not feasible and the learned surrogate must stay accurate over more steps without a feasibility anchor.
\end{enumerate}
Both are cheap to implement and should be benchmarked for solve time, feasibility violation (should be $\approx 0$ for both, by construction, at least at reported points), and solution accuracy vs. exact PGM/OSQP.

\subsection{Step-size: fixed vs.\ adaptive}
\label{sec:gamma}

\subsubsection{Fixed $\gamma$}
$\gamma$ is a hyperparameter, not an input to the network beyond implicitly shaping the training data. Intuition: small $\gamma$ makes $\phi_\gamma$ sharper (larger gradients near the boundary, closer to the true indicator), which is more informative for the outer PGM iteration but harder for the network to fit smoothly; large $\gamma$ gives a smoother, easier-to-learn envelope but a weaker/slower proximal signal. \textbf{Plan:} sweep small vs.\ large $\gamma$ and measure learnability (fit error) vs.\ downstream PGM convergence speed to characterize this tradeoff empirically.

\subsubsection{Adaptive $\gamma$}
If we let $\gamma$ vary (e.g., across outer iterations or problem instances), we fold it into the parameter: $\theta = (x_0, \gamma)$, playing the same role as $\theta$ in LPCF. This requires:
\begin{itemize}[nosep]
  \item Enforcing convexity in $W$ as before (unaffected by adding $\gamma$ to $\theta$).
  \item \textbf{Open problem:} enforcing monotonicity of $\widehat\phi_\gamma$ with respect to $\gamma$. LPCF's monotonicity extension (\S3.2 of that paper) enforces monotonicity of $f$ \emph{with respect to the input $x$} (nonnegative/nonpositive $V_l$ feeding $x$ into each layer). It does \emph{not} directly give monotonicity with respect to the parameter $\theta$, since $\theta$ enters only through the hypernetwork $\psi(\theta)$ that generates $(W_l, V_l, \omega_l)$, not through a dedicated feedforward path with sign-constrained weights. We need a construction that isolates a $\gamma$-dependent path in $\psi$ (or a separate scalar gate) with a sign constraint analogous to LPCF's $V_l \ge 0$ trick, so that $\widehat\phi_\gamma$ is provably nonincreasing in $\gamma$ for every $W, x_0$. This is a concrete architecture design task, not yet solved.
\end{itemize}

\section{Open questions / next steps}
\begin{itemize}[nosep]
  \item Design and verify the $\gamma$-monotonic architecture extension described above.
  \item Decide the training target: match $\phi_\gamma$ itself, its gradient, or both (gradient is what we actually use downstream).
  \item Empirically compare per-iteration vs.\ final-iteration HardNet-Aff correction on speed, accuracy, and feasibility margin.
  \item Sweep fixed $\gamma$ (small vs.\ large) to characterize the learnability/convergence-speed tradeoff before committing to adaptive $\gamma$.
  \item Exploit single-shooting structure ($G$, $b(x_0)$) for a matrix-free HardNet-Aff correction, to keep the correction layer cheap relative to $\phi_\gamma$ evaluation.
  \item Benchmark end-to-end solver (learned gradient + correction, unrolled) against OSQP: solve time, feasibility, optimality gap, generalization across MPC instances (different $x_0$, possibly different horizons/dynamics).
\end{itemize}
\nocite{*}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}


\begin{algorithm}[t]
\caption{Backward automatic differentiation of $\ell(U;x_0)$}
\label{alg:backward_ad_cost}
\begin{algorithmic}[1]
\Require $x_0$, $U=\operatorname{col}(u_0,\ldots,u_{N-1})$, $A$, $B$
\Ensure $\ell(U;x_0)$, $\nabla_U\ell(U;x_0)$
\State $\ell(U;x_0)\gets 0$
\For{$k=0,\ldots,N-1$}
    \State $\ell(U;x_0)\gets\ell(U;x_0)+\ell_k(x_k,u_k)$
    \State $x_{k+1}\gets Ax_k+Bu_k$
\EndFor
\State $\ell(U;x_0)\gets\ell(U;x_0)+\ell_N(x_N)$
\State $p_N\gets\nabla_{x_N}\ell_N(x_N)$
\For{$k=N-1,\ldots,0$}
    \State $\nabla_{u_k}\ell(U;x_0)\gets\nabla_{u_k}\ell_k(x_k,u_k)+B^\top p_{k+1}$
    \State $p_k\gets\nabla_{x_k}\ell_k(x_k,u_k)+A^\top p_{k+1}$
\EndFor
\State \Return $\ell(U;x_0)$, $\nabla_U\ell(U;x_0)$
\end{algorithmic}
\end{algorithm}


\end{document}
