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


def euler_step(x_t: Tensor, x1_pred: Tensor, t_scalar: float, dt: float, pbc_aware: bool = False) -> Tensor:
    """Endpoint-matching Euler step for a continuous modality.

    Computes the velocity as (x1_pred - x_t) / (1 - t) and advances by dt.
    Works for any shape tensor (N, D), (B, D), or (B, D1, D2).

    Args:
        x_t: Current state at time t.
        x1_pred: Model's endpoint prediction.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.

    Returns:
        Updated state at time t + dt.
    """
    delta = x1_pred - x_t
   # if pbc_aware:
    #    delta = delta - torch.round(delta)
    velocity = delta / (1 - t_scalar + 1e-8)
    return x_t + velocity * dt


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
    x1_probs   = F.softmax(z_logits, dim=-1)                              # (N, 99)
    is_masked  = (zt == NUM_CLASSES - 1)                                  # (N,) bool

    rate_factor = dt / (1 - t_scalar + 1e-8)
    unmask_prob = (rate_factor * x1_probs.max(dim=-1).values).clamp(max=1.0)  # (N,)

    will_unmask = (torch.rand_like(unmask_prob) < unmask_prob) & is_masked

    zt_next = zt.clone()
    if will_unmask.any():
        zt_next[will_unmask] = torch.distributions.Categorical(x1_probs[will_unmask]).sample()

    return zt_next  # (N,) long
