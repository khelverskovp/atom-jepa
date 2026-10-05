"""Element symbol <-> atomic number tables.

The encoder embeds atoms by atomic number, so datasets only need to turn
whatever their raw files store (symbols or Z) into an `atomic_numbers` tensor.
"""

from typing import Sequence

import torch

# Full periodic table; Z = index + 1.
PERIODIC_TABLE = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(PERIODIC_TABLE)}
Z_TO_SYMBOL = {z: s for s, z in SYMBOL_TO_Z.items()}


def symbols_to_z(symbols: Sequence[str]) -> torch.Tensor:
    """Element symbols -> LongTensor of atomic numbers. Raises on unknown symbols."""
    try:
        return torch.tensor([SYMBOL_TO_Z[s] for s in symbols], dtype=torch.long)
    except KeyError as e:
        raise KeyError(f"unknown element symbol {e.args[0]!r}") from None
