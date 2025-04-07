import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from fairchem.core.models.base import BackboneInterface, GraphModelMixin, HeadInterface

from ads_flow.models.base import BaseModel

from torch_geometric.data import Data
