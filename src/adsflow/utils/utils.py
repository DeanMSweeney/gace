import os
import torch
from torch import optim
from torch.utils.data import random_split
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple

import ase 
from ase.io import read, write


def export_positions(data: Data):
    atomic_numbers = data.atomic_numbers
    pos = data.pos 
    pos_generated = data.pos_generated
    
    if hasattr(data, "pos_realxed"):
        pos_relaxed = data.pos_relaxed
    

    