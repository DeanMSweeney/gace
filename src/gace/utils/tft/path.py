from dataclasses import dataclass
from typing import Any


@dataclass
class FlowPath:
    """Container for flow path data used in loss computation.

    Attributes:
        x_1: Ground-truth endpoint data (TensorDict or dict-like).
        t: Interpolation time data (TensorDict or dict-like).
    """

    x_1: Any
    t: Any
