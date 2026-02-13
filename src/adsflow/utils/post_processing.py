import ase
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.invariance import pbc_aware_delta

import os
import shutil
import subprocess

def write_strucs(
    data: Data, 
    gen_POS: str, 
    in_POS: Optional[str]=None, 
    rlxd_POS: Optional[str]=None,  
):
    z_np = data.atomic_numbers.detach().cpu().numpy()
    cell_np = data.cell.squeeze(0).detach().cpu().numpy()
    x1_np = data.pos_generated.detach().cpu().numpy()
    x0_np = data.pos.detach().cpu().numpy()
    xdft_np = data.pos_relaxed.detach().cpu().numpy()
    f_np = data.fixed.detach().cpu().numpy()
              
    # Save to current directory
    atoms = Atoms(
        positions=x1_np, # initial
        numbers=z_np,
        cell=cell_np,
        pbc=True,  # 
    )
    atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
    ase.io.write(gen_POS, atoms)  # Cartesian
    
    if in_POS:
        # Save to current directory
        atoms = Atoms(
            positions=x0_np, # initial
            numbers=z_np,
            cell=cell_np,
            pbc=True,  # or (True, True, False) for a slab
            )
        atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
        ase.io.write(in_POS, atoms)  # Cartesian
    
    if rlxd_POS:
        # Save to current directory
        atoms = Atoms(
            positions=xdft_np, # initial
            numbers=z_np,
            cell=cell_np,
            pbc=True,  # 
            )
        atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
        ase.io.write(rlxd_POS, atoms)  # Cartesian
        

def config_DFT(
    data: Data,
    gen_POS: str,
    in_POS: Optional[str]=None, 
    rlxd_POS: Optional[str]=None,
):
    print(">>> Running VASP DFT calculations w/ user provided input files  🖲️")
    # Get the current working directory to return to later
    original_dir = os.getcwd()

    try:
        # Get the sid from the data object
        sid = data.sid.item()

        # Create the strucs directory if it doesn't exist
        if not os.path.exists("./strucs"):
            os.makedirs("./strucs")

        # Create the sid directory within strucs
        sid_dir = os.path.join("./strucs", str(sid))
        os.makedirs(sid_dir, exist_ok=True)
        
        # Function to build simulation directory
        def build_vasp_dir(sim_dir: Optional[bool]=None, struc_file: Optional[bool]=None):
            sim_path = os.path.join(sid_dir, sim_dir)
            os.makedirs(sim_path, exist_ok=True)
            shutil.move(struc_file, os.path.join(sim_path, "POSCAR"))
            
            for file in ["INCAR", "KPOINTS", "VASP.sh"]:
                if os.path.exists(file):
                    shutil.copy(file, sim_path)
                else: 
                    raise AttributeError(f"Missing {file} file for VASP run.")

            # Submit the .sh file in each directory
            os.chdir(sim_path)
            subprocess.run(["potcar.sh"])
            os.chdir(original_dir)
            
        # Move and rename files to their respective directories
        if os.path.exists(gen_POS):
            build_vasp_dir(sim_dir="generated", struc_file=gen_POS)
        else: 
            raise AttributeError("Generated strucutre file does not exist")
        
        if os.path.exists(in_POS):
            build_vasp_dir(sim_dir="inputted", struc_file=in_POS)
            
        if os.path.exists(rlxd_POS) and hasattr(data, "pos_relaxed"):
            build_vasp_dir(sim_dir="relaxed", struc_file=rlxd_POS)
            
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

def config_MLFF(
    data: Data,
    gen_POS: str,
    in_POS: Optional[str]=None, 
    rlxd_POS: Optional[str]=None,
):
    pass 



DS0 = []
DS1 = []
DS2 = []
DS3 = []
DS4 = []
DS5 = []
DS6 = []
DS7 = []  
def analyze_loss(
    data: Data,
    loss: Tensor, 
    bad_iter: int,
    median_loss: Tensor, 
    max_loss: Tensor, 
    max_loss_idx: Tensor, 
):

    """ 
    Compute statistics for max velocity field loss 
    """
    max_loss_z = data.atomic_numbers[max_loss_idx] # atomic number of atom with max loss
    max_loss_tag = data.tags[max_loss_idx] # tag of atom with max loss 
    
    # evaluate distance from adsorbate for surface atoms 
    if float(max_loss_tag) != 2:
        max_atom_pos = data.pos_relaxed[max_loss_idx] #1,3
        adsorbate = (data.tags == 2) 
        adsorbate_pos = data.pos_relaxed[adsorbate] #N_a, 3, N_a -> # of adsorbate atoms 
        
        # pbc corrected distance calc 
        K = (max_atom_pos - adsorbate_pos).size(0)
        delta_max_ads = pbc_aware_delta(
        delta=max_atom_pos - adsorbate_pos,
        cell=data.cell,
        batch=torch.zeros(K, dtype=torch.long),
        naive=True,
        l1_norm=True,
        )
        # collect diagnostics for surface atom 
        DS0.append(float(delta_max_ads.min())) # distance between atom and adsorbate
        
    """
    Compute statistics for max distance moved during relaxation 
    """
    # pbc aware displacement from relaxation 
    delta = pbc_aware_delta(
        delta=data.pos - data.pos_relaxed,
        cell=data.cell,
        batch=data.batch,
        naive=True,
        l1_norm=True,
    ) 
    
    # N,1, zero-out frozen atoms and average
    free_atoms = (1 - data.fixed).unsqueeze(1) # N
    distance_per_atom = delta * free_atoms  
    avg_distance = distance_per_atom.sum() / free_atoms.sum().clamp_min(1e-8)
    max_distance = distance_per_atom.max()
    max_distance_idx = torch.argmax(distance_per_atom)
    distance_of_max_loss = distance_per_atom[max_loss_idx]

    max_distance_z = data.atomic_numbers[max_distance_idx] # atomic number of atom with max distance moved 
    max_distance_tag = data.tags[max_distance_idx] # tag of atom with max distance moved 
    
    # print statements 
    print(f"High velocity field loss {loss} ❌ || total count: {bad_iter}")
    
    # collect diagnostics 
    DS1.append(int(max_loss_z)) # atomic number 
    DS2.append(int(max_loss_tag)) # tag (surface, bulk, adsorbate)
    DS3.append(int(max_distance_z)) # atomic number 
    DS4.append(int(max_distance_tag)) # tag (surface, bulk, adsorbate)
    DS5.append(float(max_distance)) # max distance moved during relaxation 
    DS6.append(float(distance_of_max_loss)) # max distance moved of atom with max loss 
    DS7.append(float(avg_distance)) # average distance moved of all atoms in struc 
    
    return DS0, DS1, DS2, DS3, DS4, DS5, DS6, DS7 