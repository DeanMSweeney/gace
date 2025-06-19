from ase.io import read
from ase import Atoms
from ase.visualize import view

import torch 
from torch import optim
from torch import nn
from torch import Tensor

# CG: DEBUG
# _______________________________________
import time
print("Imported modules")
print(torch.backends.mkldnn.is_available(), torch.backends.openmp.is_available())
print(torch.__config__.show())  # will say if it’s using Intel MKL or OpenBLAS

a = torch.randn(5000, 5000)
b = torch.randn(5000, 5000)
# warm up
for _ in range(3): _ = a @ b
# timed run
t0 = time.time()
for _ in range(5): _ = a @ b
print("avg CPU matmul:", (time.time() - t0)/5, "s")
# ________________________________________

