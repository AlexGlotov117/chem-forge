from dataclasses import dataclass, field
from typing import List, Dict
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
    h_f_298: float = 0.0                # Added for CEA calculations

    formula: Dict[str, int] = field(default_factory=dict) # Element tracking e.g., {"C": 2, "H": 7}
    
    # Static attributes for CEA integration
    enthalpy_units: str = "kJ/mol"
    ref_temperature: float = 298.15

    @property
    def cea_reactant(self):
        """
        Dynamically generates the NASA-CEA Reactant object using 
        the dataclass attributes on-demand.
        """
        import cea  # Imported inline to keep chemical.py decoupled if CEA isn't used
        
        if not self.formula:
            raise ValueError(f"Chemical formula dictionary missing for CEA component: {self.name}")
            
        return cea.Reactant(
            name=f"{self.name}",
            formula=self.formula,
            molecular_weight=self.mw,
            enthalpy=self.h_f_298/1000,
            enthalpy_units=self.enthalpy_units,
            temperature=self.ref_temperature
        )

@dataclass
class Mixture:
    compounds: List[Compound]
    oxidizer_name: str = "Air"
    phi: float = 1.0
    pc_psi: float = 200.0
    supar: List[float] = field(default_factory=lambda: [20.0])
    
    def __post_init__(self):
        """
        Initializes the heavy engine solvers ONCE during object instantiation.
        This keeps on-the-fly lookups extremely fast.
        """
        import cea
        self._cea_lib = cea  # Keep module reference local
        
        # 1. Instantiate the background CEA rocket mechanics
        fuels_cea = [comp.cea_reactant for comp in self.compounds]
        reac_names = fuels_cea + [self.oxidizer_name]
        
        self._reac = cea.Mixture(reac_names)
        prod = cea.Mixture(reac_names, products_from_reactants=True)
        
        self._solver_rocket = cea.RocketSolver(prod, reactants=self._reac)
        self._solution = cea.RocketSolution(self._solver_rocket)
        
        # 2. Instantiate your native SLE Engine
        self._solver_sle = SLESolver()

    @property
    def num_components(self) -> int:
        return len(self.compounds)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.compounds]

    def evaluate_at(self, x: np.ndarray, gamma: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Evaluates and returns all Thermodynamic and Rocket performance metrics
        at a specific composition vector 'x'.
        
        Input:
            x (np.ndarray): Mole fraction array of shape (N,) summing to 1.0
            gamma (np.ndarray, optional): Activity coefficients. Defaults to ideal (1.0)
        """
        N = self.num_components
        if len(x) != N:
            raise ValueError(f"Composition array size ({len(x)}) must match number of components ({N})")
            
        if gamma is None:
            gamma = np.ones_like(x)
            
        # ------------------------------------------------=====================
        # 1. On-The-Fly Thermodynamic SLE Evaluation
        # ----------------------------------------------------------------=====
        T_liquidus = np.zeros(N)
        for i, comp in enumerate(self.compounds):
            # Wrap scalar into tiny array to fit your SLESolver design requirements
            x_arr = np.array([x[i]])
            g_arr = np.array([gamma[i]])
            T_liquidus[i] = self._solver_sle._compute_component_liquidus(x_arr, g_arr, comp)[0]
            
        T_sle_envelope = np.max(T_liquidus)

        # ------------------------------------------------=====================
        # 2. On-The-Fly NASA-CEA Performance Evaluation
        # ----------------------------------------------------------------=====
        T_reactant = np.array([300.0] * (N + 1))
        pc_bar = self._cea_lib.units.psi_to_bar(self.pc_psi)
        
        # Convert the passed mole fractions 'x' to localized mass fractions
        mass_components = [x[i] * comp.mw for i, comp in enumerate(self.compounds)]
        total_fuel_mass = sum(mass_components)
        
        fuel_weights = np.array([m / total_fuel_mass for m in mass_components] + [0.0])
        ox_weights = np.array([0.0] * N + [1.0])
        
        of_ratio = self._reac.weight_eq_ratio_to_of_ratio(ox_weights, fuel_weights, self.phi)
        weights = self._reac.of_ratio_to_weights(ox_weights, fuel_weights, of_ratio)
        
        hc = self._reac.calc_property(self._cea_lib.ENTHALPY, weights, T_reactant) / self._cea_lib.R
        self._solver_rocket.solve(self._solution, weights, pc_bar, hc=hc, supar=self.supar, iac=True)
        
        # Calculate heat of combustion
        h_reactants_j = hc * self._cea_lib.R * T_reactant
        h_products_j = self._solution.enthalpy * self._cea_lib.R * self._solution.T
        h_combustion = h_reactants_j - h_products_j

        # Package the evaluated data point beautifully into a query dictionary
        return {
            "x": x,
            "T_liquidus": T_liquidus,
            "T_sle": T_sle_envelope,
            "T_adiabatic": self._solution.T,
            "c_star": self._solution.cstar,
            "isp": self._solution.isp,
            "h_combustion": h_combustion
        }
