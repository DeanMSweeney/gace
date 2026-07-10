"""
forge_sampler.py
----------------
Step functions for forge-variant ODE sampling.

Continuous modalities use a deterministic endpoint-matching Euler step.
Discrete atom types use a continuous-time Markov chain (CTMC) step,
adapted from zatom/flow/interpolants.py::DiscreteInterpolant.step().
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def langevin_schedule(t_scalar: float, cutoff: float = 0.9, eps: float = 1e-2) -> float:
    """Score/noise weighting g(t), following zatom's SampleNoiseSchedule.

    g(t) = 1 / (t + eps) for t < cutoff, else 0.0. Correction and noise are
    strongest early and switch off past `cutoff`, letting the endpoint commit
    deterministically over the final steps.

    Args:
        t_scalar: Current time as a Python float.
        cutoff: Times at or above this return 0.0 (no score/noise).
        eps: Numerical stabiliser in the denominator.

    Returns:
        Scalar weight g(t) >= 0.
    """
    if t_scalar >= cutoff:
        return 0.0
    return 1.0 / (t_scalar + eps)


def euler_step(
    x_t: Tensor,
    x1_pred: Tensor,
    t_scalar: float,
    dt: float,
    pbc_aware: bool = False,
    noise_scale: float = 0.0,
    langevin_gamma: float = 0.0,
    langevin_cutoff: float = 0.9,
) -> Tensor:
    """Endpoint-matching Euler step for a continuous modality.

    Computes the velocity as (x1_pred - x_t) / (1 - t) and advances by dt.
    Works for any shape tensor (N, D), (B, D), or (B, D1, D2).

    Two independent stochastic mechanisms can be layered on top of the drift:

    1. Langevin score correction (Algorithm 2 EUCLIDEAN STEP; zatom's
       SDEMetricInterpolant.step), enabled by langevin_gamma > 0:

           v_t   = (x1_pred - x_t) / (1 - t)
           s_t   = g(t) * (t*v_t - x_t) / (1 - t)          # score → pulls to manifold
           dW_t  = sqrt(2 * langevin_gamma * g(t)) * N(0, I)
           x_new = x_t + (v_t + s_t + dW_t) * dt

       g(t) is langevin_schedule(t, langevin_cutoff). The score s_t is only the
       true score under a standard-Gaussian prior x_0 ~ N(0, I); it is NOT valid
       for uniform/informed priors. langevin_gamma is zatom's per-modality
       white_noise_sampling_scale (~50 for raw Cartesian, ~0.01 for
       fractional/normalised coords).

    2. Prior-agnostic Brownian noise (Euler-Maruyama), enabled by
       noise_scale > 0: adds noise_scale * sqrt(dt) * N(0, I). Valid for any
       initial distribution.

    Args:
        x_t: Current state at time t.
        x1_pred: Model's endpoint prediction.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.
        pbc_aware: Unused; reserved for future minimum-image wrapping.
        noise_scale: Std of prior-agnostic Brownian noise. 0.0 disables it.
        langevin_gamma: White-noise scale γ_g for the Langevin term. 0.0
                        disables both the score correction and its Wiener noise.
        langevin_cutoff: Time above which g(t) = 0 (score/noise off).

    Returns:
        Updated state at time t + dt.
    """
    inv = 1.0 / (1 - t_scalar + 1e-8)
    velocity = (x1_pred - x_t) * inv
    x_new = x_t + velocity * dt

    # Langevin score correction + Wiener noise (assumes x_0 ~ N(0, I)).
    if langevin_gamma > 0.0:
        g_t = langevin_schedule(t_scalar, cutoff=langevin_cutoff)
        if g_t > 0.0:
            score = (t_scalar * velocity - x_t) * inv
            wiener = (2.0 * langevin_gamma * g_t) ** 0.5 * torch.randn_like(x_t)
            x_new = x_new + (g_t * score + wiener) * dt

    # Prior-agnostic Brownian noise (independent of the score term above).
    if noise_scale > 0.0:
        x_new = x_new + noise_scale * (dt ** 0.5) * torch.randn_like(x_t)

    return x_new


def ctmc_step(zt: Tensor, z_logits: Tensor, t_scalar: float, dt: float) -> Tensor:
    """One-way CTMC step for discrete atom types.

    Only masked atoms (index NUM_CLASSES-1 = 99) may commit to a real type.
    Atoms that have already committed stay fixed.

    Args:
        zt: (N,) long tensor of current atom type indices (0–98 real, 99 mask).
        z_logits: (N, 99) model output logits over real atom types only.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.

    Returns:
        (N,) updated long tensor.
    """
    NUM_CLASSES = 100
    #z_logits[:, -1] = -torch.inf  # mask token is never a valid prediction
    z1_probs   = F.softmax(z_logits, dim=-1)                              # (N, 99)
    is_masked  = (zt == NUM_CLASSES - 1)                                  # (N,) bool

    # Q1: should this atom unmask? — purely time-dependent
    unmask_prob = min(dt / (1 - t_scalar + 1e-8), 1.0)

    # Q2: if it unmasks, what type? — sampled from model distribution below
    will_unmask = (torch.rand(zt.shape[0], device=zt.device) < unmask_prob) & is_masked

    zt_next = zt.clone()
    if will_unmask.any():
        zt_next[will_unmask] = torch.distributions.Categorical(z1_probs[will_unmask]).sample()

    return zt_next  # (N,) long
