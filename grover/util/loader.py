from rdkit import Chem

def load_mol(string):
    """Load molecule from either SMILES string or filepath to PDB, Mol2, SDF"""
    if string.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(string)
    elif string.endswith('.mol2'):
        mol = Chem.MolFromMol2File(string)
    elif string.endswith('.sdf'):
        mol = next(Chem.SDMolSupplier(string))
    else:
        mol = Chem.MolFromSmiles(string)
    return mol
