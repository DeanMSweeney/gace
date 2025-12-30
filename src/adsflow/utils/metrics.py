import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple, List

from adsflow.utils.invariance import pbc_aware_delta


def struc_error(
    data,
    dw_threshold: float=0.05,
    adw_threshold: List[float]=[0.01, 0.50],
): 
        
    e_gen = pbc_aware_delta(
        delta=data.pos_generated - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
        cell=data.cell,
        batch=data.batch,
        norm=True,
)
    
    e_null = pbc_aware_delta(
        delta=data.pos - data.pos_relaxed, # write generated possitional coordinates to pos_gen attribute 
        cell=data.cell,
        batch=data.batch,
        norm=True,
)
    
    if dw_threshold:
        gen_dw_count = (e_gen < dw_threshold)
        null_dw_count = (e_null < dw_threshold)
        
    if adw_threshold:
        