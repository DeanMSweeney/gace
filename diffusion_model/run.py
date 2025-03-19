import sys
import subprocess
import os
import argparse

# These are different commands to launch training for diffusion models.
def diffusion_training(gpus, model):
    # Define a command for pretraining a diffusion model using a specific configuration.
    pretrain = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml configs/denoising/eqv2_so3.yml \
               --distributed --amp --identifier pretrainis2rstrain200_fewshot_std0.1-10_so30.01-1.55_lr1e-4 \
               --optim.lr_initial=1.e-4"

    # Define a command for training an Equivariant Transformer V2 model with conditional denoising.
    equiformerv2 = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml configs/denoising/eqv2_conditional.yml \
               --distributed --amp --identifier FTis2rstrain200_cond_std0.1-10_so30.01-1.55_wf4 \
               --optim.lr_initial=1.e-4"

    # Define a command for training a PaiNN model with a specific configuration.
    painn = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1234 \
               main.py --mode train \
               --config-yml configs/denoising/painn_so3.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn_new"

    # Define a command for training a GemNet-OC model with a specific configuration.
    gemnet_oc = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port 1234 \
               main.py --mode train \
               --config-yml configs/denoising/gemnet_so3.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_gemnet"

    # Return the command corresponding to the specified model.
    return eval(model)


def sampling_and_relaxation(ngpus=1, nsite=1):
    # Define the output path for storing results.
    out_path = f"/home/jovyan/shared-scratch/adeesh/denoising/valid_rerun/eqv2_conditional_FT"
    
    # Define the path to the checkpoint file for the diffusion model.
    ckpt_path = "/home/jovyan/shared-scratch/adeesh/adsorbdiff_ckpts/eqv2_pt_FT_conditional.pt"
    
    # Define the path to the checkpoint file for the relaxation model.
    relax_ckpt_path = (
        "/home/jovyan/shared-scratch/adeesh/ckpts/gemnet_oc_base_s2ef_2M.pt"
    )
    
    # Define paths to validation datasets (in-distribution and out-of-distribution).
    val_id = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs_train_subsplits/val_nonrelaxed_update"
    val_ood = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs/valood50_R1I0.1"
    
    # Initialize the final command string.
    final_cmd = ""
    
    # Loop over the number of sites (nsite) to generate commands for each step.
    for step in range(nsite):
        if step != 0:
            final_cmd += " && \n"  # Append a command separator if not the first step.
        
        # Define the path for the current step's output.
        step_path = f"{out_path}/{step}"
        
        # Command for running the diffusion model to generate samples (relaxations).
        com_sde = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1235 \
                main.py --mode run-relaxations \
                --config-yml configs/denoising/eqv2_conditional.yml \
                --task.relax_dataset.src={val_id} \
                --task.relax_opt.traj_dir={step_path} \
                --checkpoint {ckpt_path} \
                --distributed --amp  --model.sampling=True --seed {step} --debug"
        
        # Command to convert the generated trajectories into an LMDB database.
        lmdb = f"python scripts/pred_traj_to_lmdb.py \
                --data-path {step_path} \
                --out-path {step_path}/final_struct_lmdb \
                --num-workers 4"
        
        # Command for running the relaxation process using the GemNet-OC model.
        com = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1235 \
                main.py --mode run-relaxations \
                --config-yml configs/relaxation/gemnet_oc/gemnet_relax.yml \
                --checkpoint {relax_ckpt_path} \
                --task.relax_dataset.src={step_path}/final_struct_lmdb \
                --task.relax_opt.traj_dir={step_path}/relaxations \
                --distributed --amp --debug"
        
        # Combine the commands for the current step.
        cmd = com_sde + " && " + lmdb + " && " + com
        final_cmd += cmd  # Append the combined command to the final command string.
    
    return final_cmd  # Return the final command string.


if __name__ == "__main__":
    # For training (commented out in this example).
    # command = diffusion_training(1, "painn")

    # To perform sampling and relaxation (currently active).
    command = sampling_and_relaxation()

    # Write the generated command to a shell script file.
    with open("submit.sh", "w") as f:
        f.write(command)
    f.close()
    
    # Execute the shell script using subprocess.
    p = subprocess.Popen(["bash", "submit.sh"])
    p.wait()  # Wait for the process to complete.