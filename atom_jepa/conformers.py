"""3D conformers from SMILES: RDKit ETKDGv3 embedding + MMFF94 relaxation, hydrogens
included -- the protocol used in the paper. Needs RDKit (pip install atom-jepa[rdkit])."""

from typing import Dict, List

import torch


def smiles_to_samples(smiles: str, num_conformers: int = 1, seed: int = 42,
                      mmff_max_iters: int = 500) -> List[Dict[str, torch.Tensor]]:
    """One sample {atomic_numbers [N], node_coordinates [N, 3]} per conformer of `smiles`.

    Conformers are embedded with ETKDGv3 (random seed `seed`, no RMS pruning) and then
    relaxed with MMFF94 for up to `mmff_max_iters` steps; a conformer that has not
    converged by then keeps its last coordinates."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise ImportError("SMILES input needs RDKit: pip install atom-jepa[rdkit]") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
    if len(conformer_ids) != num_conformers:
        raise RuntimeError(f"{smiles!r}: embedded {len(conformer_ids)} of {num_conformers} conformers")
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError(f"{smiles!r}: MMFF94 has no parameters for some atoms")
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=mmff_max_iters)

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    return [{"atomic_numbers": z,
             "node_coordinates": torch.tensor(mol.GetConformer(cid).GetPositions(),
                                              dtype=torch.float32)}
            for cid in conformer_ids]
