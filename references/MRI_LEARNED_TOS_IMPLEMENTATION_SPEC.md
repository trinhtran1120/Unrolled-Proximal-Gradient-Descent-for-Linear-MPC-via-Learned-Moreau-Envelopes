# Large-Scale MRI Reconstruction with Classical and Learned Three-Operator Splitting

## 1. Purpose

Build a reproducible Python/PyTorch project for large-scale compressive-sensing MRI reconstruction using Davis-Yin three-operator splitting (TOS). The project must implement:

1. classical TOS with exact operators;
2. an ICNN that learns the Moreau envelope of the transform-consistency indicator;
3. learned TOS using the gradient of the learned envelope;
4. safeguarded learned TOS followed by exact classical iterations;
5. training, validation, in-distribution testing, out-of-distribution testing, profiling, and unit tests.

The main reference problem is the setup in *Deep ADMM-Net for Compressive Sensing MRI* (Yang et al., NeurIPS 2016). The original experiment uses real-valued brain and chest MR images, pseudo-radial k-space masks, 100 training images and 50 testing images per dataset, and eight local DCT filters of size 3 by 3. Images are 256 by 256 in the released implementation and extended paper.

This is a large-scale structured inverse problem. Fourier and convolution operators must be applied implicitly. Never instantiate an image-sized Fourier matrix or an image-sized convolution matrix.

## 2. Non-negotiable mathematical conventions

Use the following notation consistently throughout code, configuration, comments, tests, plots, and documentation.

- `w`: stacked optimization variable.
- `y_k`: internal TOS fixed-point iterate.
- `u_k`: output of the proximal operator of `G`.
- `q_k`: reflected forward-gradient point.
- `v_k`: output of the proximal operator of `F`, equivalently the exact projection.
- `v_hat_k`: ICNN approximation of `v_k`.
- `gamma`: fixed TOS forward step size. It must not vary with iteration.
- `lambda_k`: iteration-dependent TOS relaxation parameter. This is the only TOS parameter learned per stage in the initial implementation.
- `tau_l`: sparsity regularization weight for transform `l`. Do not use `lambda_l` for regularization because `lambda_k` is reserved for relaxation.
- `mu`: fixed Moreau-envelope parameter.

Do not expand the mathematical algorithm into separate `x` and `z` recursions in the paper-facing API. Internally, `w` may be stored as structured tensors for memory efficiency.

## 3. MRI reconstruction problem

Let the image be

```math
x\in\mathbb C^{N}, \qquad N=H W.
```

For the main experiments,

```math
H=W=256, \qquad N=65{,}536.
```

Let

```math
\mathcal A = P_\Omega\mathcal F
```

be the undersampled Fourier operator, where `mathcal F` is the unitary two-dimensional Fourier transform and `P_Omega` is the k-space sampling mask. Let `D_l` be fixed local analysis filters, with `l=1,...,L` and `L=8` in the main setup.

Start from

```math
\min_x\;\frac12\|\mathcal A x-b\|_2^2
+\sum_{\ell=1}^{L}\tau_\ell\|D_\ell x\|_1,
```

where `b` denotes undersampled k-space measurements. The symbol `b` is used for measurements so that `y_k` remains exclusively the TOS iterate.

Introduce transform variables `z_l=D_l x` and define the stacked variable

```math
w=(x,z_1,\ldots,z_L).
```

Define the transform-consistency subspace

```math
\mathcal V
=
\{w:\ z_\ell=D_\ell x,\ \ell=1,\ldots,L\}.
```

The final composite formulation is

```math
\min_w\;F(w)+G(w)+H(w),
```

with

```math
F(w)=\delta_{\mathcal V}(w),
```

```math
G(w)=\sum_{\ell=1}^{L}\tau_\ell\|z_\ell\|_1,
```

and

```math
H(w)=\frac12\|\mathcal A x-b\|_2^2.
```

The transforms `D_l` are fixed for a given optimization problem. Do not make them stage-dependent in the first version. Stage-dependent transforms would change `F` across iterations and would no longer correspond to classical TOS applied to one fixed objective.

## 4. Structured representation of the stacked variable

Mathematically, all variables are stacked vectors. In code, use a structured container to avoid concatenating large arrays repeatedly.

```python
@dataclass
class StackedMRI:
    image: torch.Tensor       # [batch, 2, H, W]
    coeffs: torch.Tensor      # [batch, L, 2, H, W]
```

Use two real channels for complex values:

```math
x_{\mathbb R}=(\operatorname{Re}x,\operatorname{Im}x).
```

The container must support, without flattening:

- addition and subtraction;
- multiplication by a scalar or batch of scalars;
- inner products and squared norms over all components;
- detached copies;
- conversion to and from native complex tensors where FFT operations benefit from it;
- batch slicing;
- device and dtype transfers.

All norms in losses and residuals must include both the image and coefficient blocks unless a metric explicitly states otherwise.

## 5. MRI operators

Create an abstract operator interface:

```python
class MRIOperators(Protocol):
    def A(self, x, mask): ...
    def AH(self, kspace, mask): ...
    def D(self, x): ...
    def DH(self, z): ...
    def normal_A(self, x, mask): ...
    def frame_normal(self, x): ...
```

### 5.1 Fourier measurement operator

Use an orthonormal FFT:

```python
torch.fft.fft2(x, norm="ortho")
torch.fft.ifft2(k, norm="ortho")
```

Implement

```math
\mathcal A x=P_\Omega\mathcal F x,
```

and

```math
\mathcal A^*b=\mathcal F^*P_\Omega^*b.
```

The adjoint test must satisfy

```math
\langle \mathcal A x,b\rangle
=
\langle x,\mathcal A^*b\rangle
```

to numerical precision.

### 5.2 Analysis filters

Initialize the main model with eight nonconstant 3 by 3 DCT filters, matching the Deep ADMM-Net setup. Remove the constant/average DCT basis. Store filters as convolution kernels, not expanded matrices.

Define one boundary convention and use it everywhere. The recommended initial choice is circular padding because it permits an exact FFT-domain projection. Add reflect-padding filters only as a later ablation.

Implement `D` and its exact adjoint `DH`. Verify the adjoint relation numerically. Do not assume that a flipped convolution is the adjoint until the padding and indexing conventions are tested.

### 5.3 Gradient of the smooth term

For a stacked variable `w`,

```math
\nabla H(w)
=
(\mathcal A^*(\mathcal A x-b),0,\ldots,0).
```

The implementation returns a `StackedMRI` object with a zero coefficient block.

For a unitary FFT and a binary mask,

```math
L_H=\|\mathcal A^*\mathcal A\|_2=1.
```

Still implement power iteration to estimate `L_H` and use it in tests. This catches normalization errors.

## 6. Exact proximal operators

### 6.1 Proximal operator of `G`

The image block is unchanged. Apply complex soft thresholding to each coefficient block:

```math
\mathcal S_\alpha(c)
=
\max\left(1-\frac{\alpha}{|c|},0\right)c.
```

Use a safe denominator such as `clamp_min(eps)` while explicitly returning zero below the threshold. The threshold for block `l` is

```math
\gamma\tau_\ell.
```

The API is

```python
u_k = prox_G(y_k, gamma, tau)
```

### 6.2 Exact proximal operator of `F`

Because `F` is an indicator,

```math
\operatorname{prox}_{\gamma F}(q)=\Pi_{\mathcal V}(q),
```

independently of `gamma`.

For `q=(q_x,q_{z_1},...,q_{z_L})`, solve

```math
\bar x
=
\left(I+\sum_{\ell=1}^{L}D_\ell^*D_\ell\right)^{-1}
\left(q_x+\sum_{\ell=1}^{L}D_\ell^*q_{z_\ell}\right),
```

and return

```math
\Pi_{\mathcal V}(q)
=
(\bar x,D_1\bar x,\ldots,D_L\bar x).
```

Implement two exact projection backends:

1. `fft_exact`: circular convolution, FFT diagonalization, and elementwise division;
2. `cg_exact`: matrix-free conjugate gradient for boundary conventions not diagonalized by FFT.

The FFT backend is the primary classical baseline. Precompute the frequency-domain denominator once per fixed filter bank:

```math
1+\sum_{\ell=1}^{L}|\widehat D_\ell|^2.
```

Do not call a generic dense linear solver.

The projection tests must verify:

- membership: `D(v.image) == v.coeffs`;
- idempotence: `Pi(Pi(q)) == Pi(q)`;
- orthogonality: `q-Pi(q)` is orthogonal to sampled vectors in `V`;
- FFT and high-accuracy CG projections agree.

## 7. Classical Davis-Yin TOS

Use the final agreed iteration:

```math
u^k=\operatorname{prox}_{\gamma G}(y^k),
```

```math
q^k=2u^k-y^k-\gamma\nabla H(u^k),
```

```math
v^k=\operatorname{prox}_{\gamma F}(q^k)=\Pi_{\mathcal V}(q^k),
```

```math
y^{k+1}=y^k+\lambda_k(v^k-u^k).
```

`gamma` is fixed for the whole run. Only `lambda_k` varies with iteration.

For `L_H`-Lipschitz gradient, require

```math
0<\gamma<\frac{2}{L_H}
```

and constrain relaxation to

```math
0<\lambda_k<2-\frac{\gamma L_H}{2}.
```

Use `lambda_k=1` as the default classical baseline.

Define the unscaled fixed-point residual

```math
r^k=v^k-u^k
```

and the normalized stopping residual

```math
R_k
=
\frac{\|v^k-u^k\|_2}
{\max\{1,\|u^k\|_2\}}.
```

Return `v_k.image` as the iteration reconstruction because `v_k` lies exactly in `V`. At termination, also compute the objective using `v_k.image` and its consistent coefficients.

Suggested initialization:

```math
x^0=\mathcal A^*b,
```

```math
y^0=(x^0,D_1x^0,\ldots,D_Lx^0).
```

Support a zero initialization for ablation.

## 8. Moreau envelope of the consistency indicator

For fixed `mu>0`,

```math
M_\mu^F(q)
=
\min_v\left\{F(v)+\frac{1}{2\mu}\|v-q\|_2^2\right\}
=
\frac{1}{2\mu}\operatorname{dist}^2(q,\mathcal V).
```

Its exact gradient is

```math
\nabla M_\mu^F(q)
=
\frac1\mu\left(q-\Pi_{\mathcal V}(q)\right).
```

Therefore,

```math
\Pi_{\mathcal V}(q)
=
q-\mu\nabla M_\mu^F(q).
```

The exact Moreau envelope does not accelerate projection because its target gradient contains the exact projection. Acceleration is attempted by replacing it with a learned scalar potential.

For this fixed linear subspace, the true envelope is convex, nonnegative, smooth, and quadratic. The project must explicitly acknowledge this fact. The ICNN is an approximation experiment, not evidence that the exact envelope is intrinsically nonlinear.

## 9. ICNN Moreau model

Define a scalar input-convex neural network

```math
\widehat M_{\mu,\theta}^{F}(q)\in\mathbb R
```

and compute

```math
g_\theta(q)=\nabla_q\widehat M_{\mu,\theta}^{F}(q)
```

using PyTorch automatic differentiation.

The learned projection is

```math
\widehat\Pi_{\mathcal V,\theta}(q)
=
q-\mu g_\theta(q).
```

### 9.1 Architecture requirements

A fully connected ICNN over the flattened 256 by 256 stacked variable is forbidden. Use a convolutional ICNN with local operations and a scalar nonnegative spatial aggregation.

A generic ICNN block is

```math
h_{j+1}
=
\sigma\left(W_j^{(h)}*h_j+W_j^{(q)}*q+b_j\right),
```

where:

- `*` denotes convolution;
- all weights in `W_j^(h)` are constrained nonnegative;
- skip convolutions `W_j^(q)` are unconstrained;
- `sigma` is convex and nondecreasing, preferably softplus;
- the final aggregation returns one scalar per sample;
- the model must be convex with respect to `q`.

Parameterize nonnegative recurrent weights using softplus:

```python
weight_positive = F.softplus(weight_raw)
```

The ICNN input contains all stacked blocks as channels. Keep a fixed channel order and record it in checkpoints.

### 9.2 Smoothness requirement

Convexity alone is insufficient. The true envelope satisfies

```math
0\preceq\nabla^2M_\mu^F(q)\preceq\frac1\mu I.
```

The learned gradient should therefore be approximately `1/mu`-Lipschitz. Implement at least one of:

- spectral normalization with a conservative global bound;
- gradient-Lipschitz penalty estimated from pairs of nearby samples;
- Hessian-vector-product penalty on sampled directions.

The initial implementation should combine spectral normalization and an empirical pairwise penalty. Log the observed ratio

```math
\frac{\|g_\theta(q_1)-g_\theta(q_2)\|_2}
{\|q_1-q_2\|_2}
```

and flag ratios larger than `1/mu` plus tolerance.

### 9.3 Important memory constraint

Training through `g_theta(q)` requires mixed second derivatives with respect to network parameters. Use:

- `create_graph=True` only during training;
- gradient checkpointing across ICNN blocks and TOS stages;
- small batches with gradient accumulation;
- optional truncated unrolling;
- per-sample scalar outputs before summing for autograd;
- profiling of peak allocated and reserved CUDA memory.

Do not silently detach `q` or `g_theta(q)` in the learned TOS training path.

## 10. Learned TOS

Use

```math
u^k=\operatorname{prox}_{\gamma G}(y^k),
```

```math
q^k=2u^k-y^k-\gamma\nabla H(u^k),
```

```math
\widehat v^k
=
q^k-\mu\nabla_q\widehat M_{\mu,\theta}^{F}(q^k),
```

```math
y^{k+1}
=
y^k+\lambda_k(\widehat v^k-u^k).
```

The ICNN parameters are shared across stages in the primary model. Add stage-specific ICNNs only as an ablation because they multiply memory and weaken the interpretation as one learned Moreau envelope.

### 10.1 Learned relaxation

Learn one scalar `lambda_k` per unrolled stage. Constrain it to the valid interval:

```math
\lambda_k
=
\lambda_{\min}
+
(\lambda_{\max}-\lambda_{\min})\operatorname{sigmoid}(a_k),
```

where

```math
0<\lambda_{\min}<\lambda_{\max}
<2-\frac{\gamma L_H}{2}.
```

Initialize all `lambda_k` near 1. Save both the unconstrained values `a_k` and transformed values `lambda_k`.

Do not learn `gamma` in the initial project.

## 11. Training data generation

Training the envelope requires points representative of actual TOS trajectories. Random Gaussian stacked tensors alone are insufficient.

### 11.1 MRI data

Support two data routes:

1. released Deep ADMM-Net brain/chest data for reproduction;
2. a public MRI dataset such as fastMRI for scaling experiments.

The first milestone uses 256 by 256 single-coil images. Normalize every image using one documented rule, such as division by the 99th percentile magnitude or by its maximum magnitude. Save the scale factor for every sample.

Split by patient or volume, never by adjacent slices, when metadata permits. Ensure no leakage between training, validation, and test sets.

### 11.2 Masks and measurements

Implement deterministic pseudo-radial masks matching the reference setup at sampling rates 20, 30, 40, and 50 percent. Also support Cartesian variable-density masks for OOD testing.

Generate measurements as

```math
b=P_\Omega\mathcal Fx_{\mathrm{gt}}+\varepsilon.
```

Start with noiseless data and later add complex Gaussian noise at configured SNR levels.

All masks, seeds, normalization constants, and noise realizations must be saved.

### 11.3 Moreau training samples

For each training measurement:

1. initialize `y_0`;
2. run exact classical TOS for `K_collect` iterations;
3. save selected `q_k` values;
4. compute `v_k=Pi_V(q_k)` with the exact projection;
5. compute the exact value and gradient targets:

```math
M_i^*
=
\frac{1}{2\mu}\|q_i-v_i\|_2^2,
```

```math
g_i^*
=
\frac1\mu(q_i-v_i).
```

Collect early, middle, and late trajectory points. Balance the dataset so that late points near `V` are not overwhelmed by large-residual early points.

Add controlled perturbations around trajectory points:

```math
q_i^{\mathrm{aug}}=q_i+\sigma_i\epsilon_i,
```

using several relative noise scales. Recompute exact targets for every perturbed point.

Store samples in chunked HDF5 or Zarr files. Do not keep the complete dataset in RAM.

## 12. Loss functions

### 12.1 Envelope pretraining

Use a weighted combination:

```math
\mathcal L_{\mathrm{env}}
=
\omega_M\mathcal L_M
+\omega_g\mathcal L_g
+\omega_v\mathcal L_v
+\omega_0\mathcal L_0
+\omega_L\mathcal L_L.
```

Value matching:

```math
\mathcal L_M
=
\frac1B\sum_i
\left|\widehat M_{\mu,\theta}^F(q_i)-M_i^*\right|^2.
```

Gradient matching:

```math
\mathcal L_g
=
\frac1B\sum_i
\|g_\theta(q_i)-g_i^*\|_2^2.
```

Projection matching:

```math
\mathcal L_v
=
\frac1B\sum_i
\|q_i-\mu g_\theta(q_i)-v_i\|_2^2.
```

Subspace anchoring uses exact points `p_i in V`:

```math
\mathcal L_0
=
\frac1B\sum_i
\left(
|\widehat M_{\mu,\theta}^F(p_i)|^2
+\|g_\theta(p_i)\|_2^2
\right).
```

Smoothness penalty:

```math
\mathcal L_L
=
\frac1B\sum_i
\left[
\max\left{
\frac{\|g_\theta(q_i')-g_\theta(q_i)\|_2}
{\|q_i'-q_i\|_2+\epsilon}
-\frac1\mu,
0
\right}
\right]^2.
```

Normalize value, gradient, and projection losses so their magnitude does not scale uncontrollably with image size.

### 12.2 End-to-end unrolled training

After envelope pretraining, unroll `K_learned` stages. Use:

- reconstruction loss against ground truth;
- supervised solution loss against a high-accuracy reference reconstruction;
- objective loss;
- fixed-point residual loss;
- transform-consistency loss;
- envelope regularization loss retained on sampled `q_k`.

Recommended reconstruction loss:

```math
\mathcal L_{\mathrm{recon}}
=
\frac{\|\widehat x-x_{\mathrm{gt}}\|_2^2}
{\|x_{\mathrm{gt}}\|_2^2+\epsilon}.
```

Consistency loss:

```math
\mathcal L_{\mathrm{cons}}
=
\frac{\sum_\ell\|\widehat v_{z_\ell}-D_\ell\widehat v_x\|_2^2}
{\max\{1,\|\widehat v\|_2^2\}}.
```

The primary model should first freeze the pretrained ICNN and learn `lambda_k`, then jointly fine-tune with a smaller learning rate. Compare against learning `lambda_k` alone while keeping the exact projection.

## 13. Safeguarded inference

The ICNN output is not guaranteed to lie exactly in `V`. Implement the following modes.

### 13.1 Learned-only

Run `K_learned` learned stages and return the learned reconstruction. This measures raw speed and approximation quality but provides no exact feasibility guarantee.

### 13.2 Learned warm start plus exact TOS

Run `K_learned` learned stages, then switch to the exact projection and continue classical TOS until

```math
R_k\le\varepsilon
```

or `K_exact_max` is reached. This is the primary safeguarded method.

### 13.3 Periodic exact correction

Every `J` learned stages, replace the learned projection with `Pi_V(q_k)`. Treat this as an ablation.

### 13.4 Acceptance safeguard

Optionally compute an exact candidate only when a learned step appears unreliable. Reject the learned candidate if any configured condition holds:

- normalized consistency violation increases beyond a factor;
- objective evaluated at an exact projection increases excessively;
- approximate residual is nonfinite;
- ICNN gradient norm exceeds a configured bound;
- observed local Lipschitz ratio exceeds tolerance.

If rejected, use the exact `v_k=Pi_V(q_k)` for that stage. Report the number and fraction of rejected learned steps.

## 14. Reference solutions

Use at least two high-accuracy references:

1. exact classical TOS run to a strict fixed-point tolerance;
2. a convex reference solver formulation, such as CVXPY with Clarabel or another reliable conic solver, on reduced-size images.

Full 256 by 256 conic solves may be too expensive. Solver agreement tests may use 16 by 16 and 32 by 32 images, while full-resolution reference results use converged exact TOS and, where practical, an independent ADMM implementation.

Never label a finite-iteration solution as ground truth without recording its residual and tolerance.

## 15. Experimental protocol

### 15.1 Methods

Compare:

1. zero-filled inverse FFT;
2. classical TOS with exact FFT projection;
3. classical TOS with exact CG projection;
4. exact TOS with learned `lambda_k` only;
5. learned ICNN-TOS;
6. safeguarded ICNN-TOS;
7. a conventional ADMM baseline;
8. Deep ADMM-Net results or released model where reproduction is possible.

### 15.2 Data regimes

Report:

- training performance;
- validation performance;
- in-distribution test performance;
- unseen anatomy, if available;
- unseen sampling rate;
- unseen mask family;
- unseen noise level;
- resolution transfer, if the convolutional ICNN permits it.

### 15.3 Metrics

For every method, report:

- NMSE;
- PSNR;
- optional SSIM;
- objective value;
- objective gap relative to the reference;
- transform-consistency violation;
- exact fixed-point residual, evaluated using the exact projection;
- learned approximate residual;
- iterations or stages;
- total wall-clock time;
- time to specified accuracy levels;
- per-stage time;
- time spent in FFT, filters, exact projection, ICNN forward, and ICNN input-gradient computation;
- peak CPU and GPU memory.

Use synchronized GPU timing:

```python
torch.cuda.synchronize()
```

before and after timed regions. Separate cold-start compilation/allocation time from steady-state time. Include data transfer only in explicitly labeled end-to-end measurements.

The main claim must be based on time-to-accuracy. Do not assume that an ICNN stage is faster than the exact FFT projection.

## 16. Precision and scaling

Use consistent preprocessing, scaling, FFT normalization, and dtype during data collection, training, and inference.

Default correctness mode:

- `torch.float64` for real tensors;
- `torch.complex128` for complex FFT tensors.

GPU throughput mode may use float32/complex64 only as a clearly labeled experiment. Compare its residuals and reconstruction metrics with double precision before adopting it.

Save:

- image normalization rule and sample scales;
- FFT convention;
- filter normalization;
- `gamma`, `mu`, all `tau_l`, and all `lambda_k`;
- random seeds;
- masks and mask seeds;
- dtype and device;
- software versions;
- checkpoint version and channel ordering.

## 17. Repository layout

```text
mri_learned_tos/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── classical_tos.yaml
│   ├── icnn.yaml
│   ├── train_envelope.yaml
│   ├── train_unrolled.yaml
│   └── benchmark.yaml
├── src/mri_tos/
│   ├── __init__.py
│   ├── types.py
│   ├── complex_ops.py
│   ├── fft_ops.py
│   ├── filters.py
│   ├── stacked_ops.py
│   ├── objectives.py
│   ├── prox.py
│   ├── projection.py
│   ├── tos.py
│   ├── icnn.py
│   ├── learned_tos.py
│   ├── safeguards.py
│   ├── losses.py
│   ├── metrics.py
│   ├── datasets.py
│   ├── masks.py
│   ├── references.py
│   ├── checkpoints.py
│   └── profiling.py
├── scripts/
│   ├── prepare_data.py
│   ├── generate_masks.py
│   ├── collect_moreau_data.py
│   ├── solve_references.py
│   ├── train_envelope.py
│   ├── train_unrolled.py
│   ├── infer.py
│   └── benchmark.py
├── tests/
│   ├── test_complex_ops.py
│   ├── test_fft_adjoint.py
│   ├── test_filter_adjoint.py
│   ├── test_soft_threshold.py
│   ├── test_projection.py
│   ├── test_gradient_h.py
│   ├── test_tos_step.py
│   ├── test_moreau_identity.py
│   ├── test_icnn_convexity.py
│   ├── test_relaxation_bounds.py
│   ├── test_scaling_consistency.py
│   ├── test_checkpoint_roundtrip.py
│   └── test_solver_agreement.py
└── outputs/
    ├── checkpoints/
    ├── metrics/
    ├── profiles/
    └── figures/
```

Do not commit raw medical data, generated training tensors, or large checkpoints. Include download/preparation instructions and checksums where licensing permits.

## 18. Required APIs

```python
def prox_G(y_k: StackedMRI, gamma: float, tau: Tensor) -> StackedMRI:
    ...

def grad_H(
    w: StackedMRI,
    measurement: Tensor,
    mask: Tensor,
    operators: MRIOperators,
) -> StackedMRI:
    ...

def project_V_exact(
    q: StackedMRI,
    operators: MRIOperators,
    backend: str = "fft_exact",
) -> StackedMRI:
    ...

def classical_tos_step(
    y_k: StackedMRI,
    measurement: Tensor,
    mask: Tensor,
    gamma: float,
    lambda_k: Tensor,
    tau: Tensor,
    operators: MRIOperators,
) -> tuple[StackedMRI, dict]:
    ...

def learned_projection(
    q: StackedMRI,
    icnn: nn.Module,
    mu: float,
    create_graph: bool,
) -> tuple[StackedMRI, Tensor, StackedMRI]:
    ...

def learned_tos_step(...):
    ...

def exact_fixed_point_residual(...):
    ...
```

Each TOS step returns diagnostics containing `u_k`, `q_k`, `v_k` or `v_hat_k`, residuals, objective components, and timing when profiling is enabled.

## 19. Unit and integration tests

All tests must run on CPU. GPU tests should be conditionally enabled.

Required tests:

1. complex real-channel round trip;
2. FFT forward-adjoint identity;
3. filter forward-adjoint identity;
4. complex soft-thresholding against hand-computed examples;
5. gradient of `H` against finite differences;
6. exact projection membership, idempotence, and orthogonality;
7. FFT projection agreement with CG;
8. one TOS iteration against a direct small-vector implementation;
9. Moreau value-gradient identity using exact projection;
10. ICNN convexity checked on random line segments;
11. transformed `lambda_k` always lies in its safe interval;
12. training and inference preprocessing produce identical tensors;
13. checkpoint save-load reproduces identical outputs;
14. small-problem solution agrees with Clarabel/CVXPY;
15. learned-to-exact safeguard reduces the exact residual;
16. no NaN or infinity under all configured sampling rates.

Use deterministic seeds and strict tolerances appropriate to dtype.

## 20. Implementation order

Implement in the following milestones.

### Milestone 1: exact small problem

- structured variable container;
- FFT and filter operators;
- exact proximal operators;
- exact projection;
- classical TOS;
- unit tests on 16 by 16 and 32 by 32 images;
- agreement with a reference solver.

### Milestone 2: full-resolution classical baseline

- 256 by 256 data pipeline;
- pseudo-radial masks;
- exact FFT projection;
- residual, objective, NMSE, and PSNR reporting;
- CPU and GPU profiling.

Do not proceed to learned projection until this baseline is correct and timed.

### Milestone 3: Moreau dataset

- collect real TOS trajectory points;
- compute exact envelope values and gradients;
- save chunked datasets and metadata;
- inspect target norms across iteration and sampling rate.

### Milestone 4: ICNN pretraining

- convolutional scalar ICNN;
- value, gradient, projection, anchoring, and smoothness losses;
- convexity and local-Lipschitz diagnostics;
- checkpointing and validation.

### Milestone 5: learned and safeguarded TOS

- learned projection insertion;
- safe learned `lambda_k`;
- frozen-ICNN training, then optional joint fine-tuning;
- exact continuation and periodic correction;
- ID and OOD evaluation.

### Milestone 6: final benchmark

- repeated synchronized timings;
- time-to-accuracy plots;
- ablations;
- reproducibility report;
- failure-case analysis.

## 21. Ablations

At minimum, compare:

- exact projection versus ICNN projection;
- exact projection with fixed `lambda=1` versus learned `lambda_k`;
- shared ICNN versus stage-specific ICNN on a reduced problem;
- value-only versus gradient-only versus combined envelope training;
- with and without smoothness regularization;
- learned-only versus exact continuation;
- trajectory samples versus random samples;
- float64/complex128 versus float32/complex64;
- 3 by 3 versus 5 by 5 filters;
- fixed sampling rate versus mixed sampling rates.

## 22. Acceptance criteria

The implementation is complete only when:

1. exact classical TOS converges on small and full-resolution cases;
2. the exact projection passes all geometric tests;
3. the objective and residual use consistent scaling;
4. ICNN outputs are convex to numerical tests on sampled line segments;
5. learned relaxation remains in its valid range;
6. learned projection error is reported separately from reconstruction error;
7. safeguarded inference reaches the requested exact residual tolerance;
8. all reference comparisons identify solver tolerance and dtype;
9. all benchmark tables separate cold-start, steady-state, and end-to-end time;
10. every result can be reproduced from saved configuration, seed, normalization metadata, and checkpoint.

## 23. Claims and interpretation

Use careful language in reports and papers.

- The MRI problem is large-scale and contains an implicitly dense partial Fourier operator.
- Its computational implementation is structured and matrix-free, not a generic dense-matrix QP.
- The exact consistency projection can exploit FFT structure and is a strong baseline.
- The learned ICNN projection is approximate and does not guarantee exact membership in `V`.
- Exact post-processing restores the classical convergence mechanism.
- Acceleration must be demonstrated by measured time to equal accuracy, not inferred from fewer iterations.
- If exact FFT projection is faster, report that outcome and reposition the ICNN as a learned warm-start or iteration-reduction mechanism.

## 24. Reproducibility checklist

Every experiment directory must contain:

- resolved YAML configuration;
- Git commit hash;
- Python, PyTorch, CUDA, cuDNN, and GPU information;
- all random seeds;
- dataset split identifiers;
- masks or mask seeds;
- normalization statistics;
- filter coefficients;
- `gamma`, `mu`, `tau_l`, and transformed `lambda_k`;
- model checkpoint;
- training and validation curves;
- raw per-instance metrics;
- aggregate tables;
- timing methodology;
- failure log.

## 25. References and implementation resources

1. D. Davis and W. Yin, *A Three-Operator Splitting Scheme and its Optimization Applications*, 2015.
2. Y. Yang, J. Sun, H. Li, and Z. Xu, *Deep ADMM-Net for Compressive Sensing MRI*, NeurIPS 2016.
3. Official Deep ADMM-Net code: <https://github.com/yangyan92/Deep-ADMM-Net>
4. fastMRI dataset and tools: <https://fastmri.med.nyu.edu/>
5. PyTorch FFT documentation: <https://pytorch.org/docs/stable/fft.html>

The implementation must begin with Milestone 1 and preserve the notation and algorithm specified in Sections 2 and 7.
