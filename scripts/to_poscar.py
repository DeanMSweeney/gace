import os
import glob
from ase.io import read, write

def convert_traj_to_poscar():
    # 1. Create the target directory if it doesn't exist
    output_dir = './poscars'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find all .traj files in the current directory
    traj_files = glob.glob('*.traj')

    if not traj_files:
        print("No .traj files found in the current directory.")
        return

    print(f"Found {len(traj_files)} .traj file(s). Starting conversion...")

    # 3. Iterate and convert
    for traj_file in traj_files:
        # Get the file name without the .traj extension
        base_name = os.path.splitext(os.path.basename(traj_file))[0]
        
        # Define the output path (e.g., ./poscars/POSCAR_example)
        output_file = os.path.join(output_dir, f"POSCAR_{base_name}")
        
        try:
            # index=-1 ensures we read the final/most recent frame of the trajectory
            atoms = read(traj_file, index=-1)
            
            # Write to VASP POSCAR format
            write(output_file, atoms, format='vasp')
            print(f"✅ Converted: {traj_file} -> {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to convert {traj_file}. Error: {e}")

if __name__ == "__main__":
    convert_traj_to_poscar()
