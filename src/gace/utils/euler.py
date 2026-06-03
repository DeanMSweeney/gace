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
    """Full CTMC Euler step for discrete atom types, following zatom's DiscreteInterpolant.step().

    At each step the model predicts the endpoint distribution p(z1 | z_t, t).
    Every atom may transition to any class. The rate of transitioning to class c is
    proportional to p(c | z_t, t), giving a per-step transition probability of
    dt * p(c) / (1 - t). The probability of staying is 1 minus the sum of all
    off-diagonal transition probabilities.

    Args:
        zt: (N,) long tensor of current atom type indices.
        z_logits: (N, C) model output logits for the endpoint distribution.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.

    Returns:
        (N,) updated long tensor.
    """
    z1_probs = F.softmax(z_logits, dim=-1)                        # (N, C)

    rate_factor = dt / (1 - t_scalar + 1e-8)
    step_probs = (rate_factor * z1_probs).clamp(max=1.0)          # (N, C)

    # Zero self-transition, fill with stay probability
    step_probs.scatter_(-1, zt.unsqueeze(-1), 0.0)
    prob_stay = (1.0 - step_probs.sum(dim=-1, keepdim=True))
    step_probs.scatter_(-1, zt.unsqueeze(-1), prob_stay)

    step_probs = step_probs.clamp(min=0.0)
    step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True)

    return torch.distributions.Categorical(probs=step_probs).sample()  # (N,)