from ase.io import read

def parse_poscar(file_path):
    """
    Parse a POSCAR file using ASE.
    Returns:
        positions: Atomic positions (fractional coordinates) as a numpy array.
        species: List of atom types (chemical symbols).
    """
    structure = read(file_path, format="vasp")  # Read the POSCAR file
    positions = structure.get_positions()  # Cartesian coordinates
    species = structure.get_chemical_symbols()  # Atom types
    lattice = structure.get_cell()
    return atom_positions, atom_species, lattice_vectors

filepath = "/Users/deanmsweeney/Documents/CFM /CFM-Adsorbate-Placement/flow_matching/utils/POSCAR"
positions, species, lattice_vecotrs = parse_poscar(filepath)
