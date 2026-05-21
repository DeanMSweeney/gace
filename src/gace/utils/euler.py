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


def ctmc_step(z_hot: Tensor, z_logits: Tensor, t_scalar: float, dt: float) -> Tensor:
    """CTMC Euler step for discrete atom types in sparse (N, C) format.

    At each step the model predicts the endpoint distribution p(z1 | z_t, t).
    The CTMC rate for transitioning from state i to state j is:
        q_{ij}(t) = p(z1=j | ...) / (1 - t)
    giving a per-step transition probability of dt * q_{ij}.

    Adapted from zatom/flow/interpolants.py::DiscreteInterpolant.step(),
    flattened to (N, C) without a batch dimension.

    Args:
        z_hot: (N, C) one-hot tensor of current discrete states.
        z_logits: (N, C) model output logits for the endpoint distribution.
        t_scalar: Current time as a Python float.
        dt: Step size as a Python float.

    Returns:
        (N, C) updated one-hot tensor.
    """
    x1_probs = F.softmax(z_logits, dim=-1)           # (N, C)
    curr_state = z_hot.argmax(dim=-1)                 # (N,)

    rate_factor = dt / (1 - t_scalar + 1e-8)         # scalar

    # Probability of transitioning to each state
    step_probs = (rate_factor * x1_probs).clamp(max=1.0)  # (N, C)

    # Zero out the diagonal (prob of transitioning to current state)
    step_probs.scatter_(-1, curr_state.unsqueeze(-1), 0.0)

    # Prob of staying = 1 - sum of off-diagonal transition probs
    prob_stay = (1.0 - step_probs.sum(dim=-1, keepdim=True))
    step_probs.scatter_(-1, curr_state.unsqueeze(-1), prob_stay)

    # Numerical stability
    step_probs = step_probs.clamp(min=0.0)
    step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True)

    x_next = torch.distributions.Categorical(probs=step_probs).sample()  # (N,)
    return F.one_hot(x_next, num_classes=z_hot.shape[-1]).float()
