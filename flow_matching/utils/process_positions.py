from pymatgen.core.structure import Structure

def parse_poscar(file_path):
    '''
    Parse a POSCAR file using pymatgen.
    Returns:
        positions: Atomic positions (fractional coordinates) as a numpy array.
        species: List of atom types (chemical symbols).
    '''
    structure = Structure.from_file(file_path)
    positions = structure.frac_coords  # Fractional coordinates
    species = [str(site.specie) for site in structure]  # Atom types
    return positions, species


filepath = "/Users/deanmsweeney/Documents/CFM /CFM-Adsorbate-Placement/flow_matching/POSCAR"

positions, species = parse_poscar(filepath)
