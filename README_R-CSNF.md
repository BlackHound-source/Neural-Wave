# Recurrent Continuous Schrödinger Neural Field (R-CSNF)

*A unitary, continuous-time recurrent reservoir framework for chaotic, multi-scale, and non-integrable time-series prediction.*

## Overview

The **Recurrent Continuous Schrödinger Neural Field (R-CSNF)** is a proposed recurrent reservoir architecture that represents its hidden state as a coupled, two-component complex-valued field on a one-dimensional periodic domain, \(S^1 \cong [0,2\pi)\).

Its design combines Hamiltonian-style field evolution, delay-informed wave-packet injection, persistent recurrent state, and a regularized linear readout. The framework is motivated by challenges in long-horizon time-series prediction, including unstable optimization, excessive numerical dissipation, and ambiguity when trajectories cross in observed coordinates.

> **Scientific status:** The architecture and benchmark values below are presented as supplied. They should be treated as reported experimental results, not independently verified results. Claims about stability, conservation, and performance depend on the exact implementation and experimental protocol.

## 1. Theoretical formulation

The two-component field is

\[
\mathbf{\Psi}(x,t)=
\begin{bmatrix}
\psi(x,t)\\
\phi(x,t)
\end{bmatrix}.
\]

It evolves according to a time-dependent Schrödinger-like equation,

\[
i\tau\frac{\partial\mathbf{\Psi}}{\partial t}
=\hat{H}(t)\mathbf{\Psi},
\qquad
\hat{H}(t)=
\begin{bmatrix}
\hat{H}_{\psi} & \hat{J}(x,t)\\
\hat{J}^{\dagger}(x,t) & \hat{H}_{\phi}
\end{bmatrix}.
\]

The field operators are defined as

\[
\hat{H}_{\psi}=-D_{\psi}\nabla^2+V_{\psi}(x),
\qquad
\hat{H}_{\phi}=-D_{\phi}\nabla^2+V_{\phi}(x),
\]

with multi-mode inter-field coupling

\[
\hat{J}(x,t)=a(x)\sum_m w_m
\exp\!\left(i\left(\theta(x)-\omega_m t\right)\right).
\]

Here, \(D_\psi,D_\phi\) control dispersion; \(V_\psi,V_\phi\) are potential landscapes; and \(\hat J\) couples the signal and carrier fields.

## 2. Architecture

### 2.1 Cayley / Crank–Nicolson evolution

The proposed midpoint time step is

\[
\begin{aligned}
\mathbf{\Psi}(t+\Delta t)
={}&\left(\mathbf{I}+\frac{i\Delta t}{2\tau}
\hat{H}\left(t+\frac{\Delta t}{2}\right)\right)^{-1}\\
&\left(\mathbf{I}-\frac{i\Delta t}{2\tau}
\hat{H}\left(t+\frac{\Delta t}{2}\right)\right)
\mathbf{\Psi}(t).
\end{aligned}
\]

For a Hermitian midpoint Hamiltonian and an accurately solved linear system, the Cayley step is unitary in exact arithmetic. In finite-precision computation, norm conservation is approximate and should be measured. This property does **not** by itself guarantee the absence of exploding or vanishing gradients in every training pipeline, nor does it imply that all forms of numerical error vanish over arbitrarily long sequences.

### 2.2 Delay embedding through wave-packet injection

Lagged inputs \((u_t,u_{t-1},\ldots,u_{t-K+1})\) are encoded as a superposition of Gaussian packets with phase winding:

\[
\phi_{\mathrm{inj}}(x)=
\sum_{k=0}^{K-1} w_k
\exp\!\left(-\frac{(x-c_k(u_{t-k}))^2}{2\sigma^2}\right)
\exp\!\left(i\frac{v_t(k+1)\pi x}{L}\right).
\]

The angular difference is calculated with a shortest-arc metric,

\[
v_t=\operatorname{atan2}\!\left(
\sin(u_t-u_{t-1}),\,
\cos(u_t-u_{t-1})
\right).
\]

This wrapping is appropriate for angular or periodic variables measured in radians. For ordinary non-periodic scalar data, use a velocity definition suitable to that variable rather than applying angular wrapping automatically.

### 2.3 Persistent recurrent state

The terminal field from the previous input step is blended with the new injection:

\[
\mathbf{\Psi}_0^{(t)}
=(1-\alpha)\mathbf{\Psi}_T^{(t-1)}
+\alpha\mathbf{\Psi}_{\mathrm{inj}}^{(t)},
\]

followed by normalization,

\[
\mathbf{\Psi}_0^{(t)}
\leftarrow
\frac{\mathbf{\Psi}_0^{(t)}}
{\|\mathbf{\Psi}_0^{(t)}\|}.
\]

The retention coefficient is \(1-\alpha\). Normalization makes this state update nonlinear and can change the norm independently of the unitary evolution step. Therefore, norm conservation should be evaluated separately for (1) the propagation operator and (2) the complete recurrent update.

### 2.4 Centered SVD–Tikhonov readout

Let \(F\) be the feature matrix and \(y\) the target vector. Center them using training-set statistics:

\[
\widetilde F=F-\mu_F,
\qquad
\widetilde y=y-\mu_y.
\]

With the singular value decomposition

\[
\widetilde F=USV^\mathsf{T},
\]

the regularized readout is

\[
W=VS_{\mathrm{filt}}U^\mathsf{T}\widetilde y,
\]

where

\[
S_{\mathrm{filt},i}=
\begin{cases}
\dfrac{S_i}{S_i^2+\lambda},
& S_i\geq S_0\,\mathrm{rcond},\\[6pt]
0,&\text{otherwise}.
\end{cases}
\]

Predictions restore the target mean:

\[
\widehat y=\widetilde F W+\mu_y.
\]

Use training-only means and scales to prevent leakage. The truncated SVD threshold and Tikhonov regularization address different issues: truncation removes sufficiently small singular directions, while \(\lambda\) shrinks retained directions.

**Important mathematical qualification:** constant probability or norm conservation does not, by itself, establish that appending a bias column necessarily creates an exact zero eigenvalue. Analytical centering and bias restoration are a sound readout design choice, but the claimed eigenvalue consequence requires a precise derivation for the particular feature matrix and solver.

## 3. Empirical benchmarks

The supplied benchmark description specifies chronological, leakage-controlled partitions of **60% training / 20% validation / 20% test** and uses normalized root mean squared error:

\[
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\sigma_y}.
\]

The reference baseline is persistence, \(\widehat y_{t+1}=y_t\), described as cold-start persistence.

| Benchmark | Dynamical regime | Grid and lags | Diffusion \((D_\psi,D_\phi)\) | Test NRMSE | Persistence NRMSE | Reported advantage |
|---|---|---:|---:|---:|---:|---:|
| Lorenz-63 (\(x\)) | Continuous chaos (ODE) | \(N=64,\ K=16\) | \((0.30,0.25)\) | **0.0330** | 0.1572 | **4.76× better** |
| Logistic map (\(r=3.9\)) | Discrete chaos (1D map) | \(N=48,\ K=8\) | \((0.25,0.25)\) | **0.0947** | 1.7833 | **18.8× better** |
| Mackey–Glass (\(\tau=17\)) | Delay differential equation | \(N=64,\ K=8\) | \((0.25,0.40)\) | **0.0533** | 0.1387 | **2.60× better** |
| Driven pendulum (\(\theta\)) | Angular chaos (ODE) | \(N=32,\ K=8\) | \((0.10,0.25)\) | **0.4125** | 0.4231 | **1.03× better** |
| Lotka–Volterra prey | Smooth limit cycle | \(N=48,\ K=8\) | \((0.20,0.25)\) | **0.0548** | 0.0611 | **1.11× better** |
| Sunspot-like activity | Stochastic quasi-periodic | \(N=256,\ K=128\) | \((0.001,0.001)\) | **1.0506** | 1.2666 | **1.21× better** |

The advantage column is approximately the ratio of persistence NRMSE to R-CSNF test NRMSE. It is a relative comparison against the listed baseline, not a claim of universal superiority. In particular, the driven-pendulum result indicates only a small improvement, and the sunspot-like test NRMSE above 1.0 indicates substantial residual prediction error.

For reproducibility, publish dataset-generation details, sample counts, exact chronological split boundaries, normalization rules, random seeds, model-selection procedure, baseline implementation, and per-seed results. Confirm that all hyperparameters were selected without using the final test partition.

## 4. Domain-specific parameter guidelines

The following are **proposed starting ranges**, not universal physical laws. Tune them on validation data and report sensitivity.

| Regime | Packet width \(\sigma\) | Diffusion \((D_\psi,D_\phi)\) | Frequencies \(\omega\) | Memory retention \(1-\alpha\) | Readout \(\lambda\) |
|---|---|---|---|---|---|
| Continuous chaos (Lorenz) | 0.20–0.35 (narrow) | 0.10–0.30 (moderate) | \([0.5,1.25,2.75]\) | 30–40% | \(10^{-6}\)–\(10^{-4}\) |
| Discrete maps (logistic) | 0.30–0.40 (medium) | 0.20–0.25 (moderate) | \([0.5,1.25,2.75]\) | 15–25% | \(10^{-4}\) |
| Delay DDEs (Mackey–Glass) | 0.25–0.35 (narrow) | 0.25–0.40 (active) | \([0.5,1.25,2.75]\) | 40–50% | \(10^{-6}\) |
| Smooth orbits (Lotka–Volterra) | 0.40–0.60 (broad) | 0.20–0.30 (standing-wave regime) | Approximately \(\omega_{\mathrm{orbit}}\); \([0.6,1.2,1.8]\) | 40–50% | \(10^{-3}\)–\(10^{-2}\) |
| Stochastic bursts (sunspots) | 0.05–0.10 (sharp) | 0.001 (minimal dispersion) | \([12.0,20.0,28.0]\) (fast) | 10–25% | \(10^{-2}\)–\(10^{-1}\) |

### Interpreting the guidelines

- **Continuous chaos:** narrower packets and sufficient grid resolution may help distinguish nearby regions of the attractor. Low regularization should be used only when validation and conditioning diagnostics support it.
- **Discrete maps:** stronger injection may reduce unwanted recurrent inertia, while moderate diffusion may produce useful spatial features.
- **Delay systems:** persistent state and adequate resolution may preserve information related to delayed dependencies; a recurrent field is not automatically an exact delay line.
- **Smooth limit cycles:** matching carrier timescales to the dominant orbital frequency may reduce unnecessary oscillatory structure; moderate diffusion can support smoother representations.
- **Spiky series:** low diffusion may localize bursts, while low memory retention can limit stale-noise persistence. High carrier frequencies are a hypothesis to test, not a guarantee of improved forecasting.

## 5. Recommended validation protocol

1. **Chronological split:** keep training, validation, and test intervals ordered in time. Never use test outcomes for parameter selection.
2. **Training-only preprocessing:** calculate feature and target normalization statistics from training data only.
3. **Baselines:** compare against persistence and at least one suitable classical or machine-learning baseline.
4. **Multiple seeds / windows:** report mean, standard deviation or confidence intervals, and per-run outcomes.
5. **Numerical diagnostics:** measure propagation norm error, solver residuals, feature condition numbers, readout weight norms, and angular phase error where relevant.
6. **Ablations:** test the effects of packet width, grid size, diffusion, carrier frequencies, memory retention, centering, SVD cutoff, and \(\lambda\).
7. **Long-horizon evaluation:** report one-step metrics separately from multi-step rollout performance. Small one-step error does not establish long-horizon stability.
8. **Reproducibility:** record code version, dependencies, solver tolerances, discretization, random seeds, and exact dataset splits.

## 6. Notation

| Symbol | Meaning |
|---|---|
| \(\mathbf{\Psi}\) | Coupled signal and carrier field |
| \(\psi,\phi\) | Signal and carrier components |
| \(\tau\) | Time-scale parameter in the evolution equation |
| \(D_\psi,D_\phi\) | Field dispersion coefficients |
| \(V_\psi,V_\phi\) | Spatial potential landscapes |
| \(\hat J\) | Inter-field coupling operator |
| \(N\) | Spatial grid resolution |
| \(K\) | Number of input lags |
| \(\sigma\) | Gaussian packet width |
| \(\omega_m\) | Carrier frequencies |
| \(\alpha\) | Injection fraction; \(1-\alpha\) is nominal memory retention |
| \(\lambda\) | Tikhonov regularization parameter |
| \(\mathrm{rcond}\) | Relative singular-value cutoff |
| NRMSE | RMSE normalized by target standard deviation |

## 7. Summary

R-CSNF proposes a continuous wave-field reservoir with Hermitian Hamiltonian coupling, Cayley-style propagation, delay-informed input injection, persistent recurrent state, and a centered SVD–Tikhonov readout. The supplied benchmark table reports improvements over a persistence baseline across six dynamical regimes, with gains ranging from marginal to substantial. The next step toward a rigorous research contribution is a reproducible evaluation that verifies the numerical invariants, validates the claimed mechanisms through ablations, and compares against well-tuned baselines under identical chronological splits.
