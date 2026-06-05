from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Compound:
    name: str
    mw: float                    # Molecular weight
    t_fus: float                # Melting temperature (K)
    h_fus: float                 # Enthalpy of fusion (J/mol)
    t_ss: float = 0.0
    h_ss: float = 0.0
    t_vap: float = 0.0               # Added for future VLE expansion
    h_vap: float = 0.0                # Added for future VLE expansion

@dataclass
class Mixture:
    compounds: List[Compound]
    
    @property
    def num_components(self) -> int:
        return len(self.compounds)
    
    @property
    def names(self) -> List[str]:
        return [c.name for c in self.compounds]