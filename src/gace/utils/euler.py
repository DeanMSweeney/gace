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


def euler_step(
    x_t: Tensor,
    x1_pred: Tensor,
    t_scalar: float,
    dt: float,
    pbc_aware: bool = False,
    noise_scale: float = 0.0,
) -> Tensor:
    """Endpoint-matching Euler step for a continuous modality.

    Computes the velocity as (x1_pred - x_t) / (1 - t) and advances by dt.
    Works for any shape tensor (N, D), (B, D), or (B, D1, D2).

    When noise_scale > 0, adds Brownian noise (Euler-Maruyama) scaled by
    (1 - t) so stochasticity tapers to zero as the trajectory commits near t=1.
    The noise amplitude is: noise_scale * (1 - t) * sqrt(dt) * N(0, I).

    This is prior-agnostic — unlike the score correction in EUCLIDEAN STEP,
    which assumes x_0 ~ N(0, I), Brownian noise is valid for any initial
    distribution (Gaussian, uniform, etc.).

    Args:
        x_t: Current state at time t.
        x1_pred: Model's endpoint prediction.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.
        pbc_aware: Unused; reserved for future minimum-image wrapping.
        noise_scale: Std of injected noise before time-tapering. 0.0 recovers
                     the deterministic Euler step.

    Returns:
        Updated state at time t + dt.
    """
    delta = x1_pred - x_t
    velocity = delta / (1 - t_scalar + 1e-8)
    x_new = x_t + velocity * dt

    if noise_scale > 0.0:
        taper = 1.0 - t_scalar          # → 0 as t → 1
        noise = noise_scale * taper * (dt ** 0.5) * torch.randn_like(x_t)
        x_new = x_new + noise

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
    z_logits[:, -1] = -torch.inf  # mask token is never a valid prediction
    z1_probs   = F.softmax(z_logits, dim=-1)                              # (N, 99)
    is_masked  = (zt == NUM_CLASSES - 1)                                  # (N,) bool

    # Q1: should this atom unmask? — confident atoms unmask faster
    max_probs   = z1_probs.max(dim=-1).values                               # (N,) model confidence
    unmask_prob = (max_probs * dt / (1 - t_scalar + 1e-8)).clamp(max=1.0)  # (N,) per-atom rate

    # Q2: if it unmasks, what type? — sampled from model distribution below
    will_unmask = (torch.rand(zt.shape[0], device=zt.device) < unmask_prob) & is_masked

    zt_next = zt.clone()
    if will_unmask.any():
        zt_next[will_unmask] = torch.distributions.Categorical(z1_probs[will_unmask]).sample()

    return zt_next  # (N,) long
