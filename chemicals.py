from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class Compound:
    name: str
    mw: float                    # Molecular weight
    T_fus: float                # Melting temperature (K)
    h_fus: float                 # Enthalpy of fusion (J/mol)
    T_ss: float = 0.0
    h_ss: float = 0.0
    T_vap: float = 0.0               # Added for future VLE expansion
    h_vap: float = 0.0                # Added for future VLE expansion
    h_f_298: float = 0.0                # Added for CEA calculations

    formula: Dict[str, int] = field(default_factory=dict)
    
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
        Initializes heavy engines once. Sets up state attributes 
        for tracking composition, gammas, and cached evaluations.
        """
        from solvers import SLESolver
        import cea
        self._cea_lib = cea  
        
        # Instantiate background CEA mechanics
        fuels_cea = [comp.cea_reactant for comp in self.compounds]
        reac_names = fuels_cea + [self.oxidizer_name]
        
        self._reac = cea.Mixture(reac_names)
        prod = cea.Mixture(reac_names, products_from_reactants=True)
        
        self._solver_rocket = cea.RocketSolver(prod, reactants=self._reac)
        self._solution = cea.RocketSolution(self._solver_rocket)

        self._solver_sle = SLESolver()

        # State tracking variables for evaluation / lazy loading
        self._current_x: Optional[np.ndarray] = None
        self._current_gamma: Optional[np.ndarray] = None
        self._cache: Dict[str, Any] = {}

    @property
    def num_components(self) -> int:
        return len(self.compounds)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.compounds]

    # -------------------------------------------------------------------------
    # State Management / Context Setting
    # -------------------------------------------------------------------------
    def set_composition(self, x: np.ndarray, gamma: Optional[np.ndarray] = None):
        """
        Sets the active composition context for the mixture. Resets the internal
        cache only if the inputs have shifted.
        """
        x = np.asarray(x)
        if len(x) != self.num_components:
            raise ValueError(f"Composition size ({len(x)}) must match components ({self.num_components})")
        
        if gamma is None:
            gamma = np.ones_like(x)
        else:
            gamma = np.asarray(gamma)

        # Check if state changed to prevent breaking cache unnecessarily
        if self._current_x is None or not np.allclose(self._current_x, x) or not np.allclose(self._current_gamma, gamma):
            self._current_x = x.copy()
            self._current_gamma = gamma.copy()
            self._cache.clear()  # Purge cache for a fresh lazy-eval cycle

    def _ensure_evaluated(self):
        """Internal safeguard to force processing if properties are called raw."""
        if self._current_x is None:
            raise ValueError("Mixture state not initialized. Call mixture.set_composition(x, gamma) first.")
        if not self._cache:
            self._cache = self.evaluate_at(self._current_x, self._current_gamma)

    # -------------------------------------------------------------------------
    # On-The-Fly Individual Property Accessors
    # -------------------------------------------------------------------------
    @property
    def T_liquidus(self) -> np.ndarray:
        self._ensure_evaluated()
        return self._cache["T_liquidus"]

    @property
    def T_fus(self) -> float:
        """Alias for the overall SLE melting point envelope peak."""
        self._ensure_evaluated()
        return self._cache["T_sle"]

    @property
    def T_adi(self) -> float:
        self._ensure_evaluated()
        return self._cache["T_adiabatic"]

    @property
    def c_star(self) -> float:
        self._ensure_evaluated()
        return self._cache["c_star"]

    @property
    def isp(self) -> float:
        self._ensure_evaluated()
        return self._cache["isp"]

    @property
    def h_c(self) -> float:
        self._ensure_evaluated()
        return self._cache["h_combustion"]

    # -------------------------------------------------------------------------
    # Underlying Core Solvers
    # -------------------------------------------------------------------------
    def evaluate_at(self, x: np.ndarray, gamma: np.ndarray) -> Dict[str, Any]:
        """
        Calculates and returns all raw performance metrics for the cache map.
        """
        N = self.num_components
        
        # 1. SLE Phase Loop
        T_liquidus = np.zeros(N)
        for i, comp in enumerate(self.compounds):
            x_arr = np.array([x[i]])
            g_arr = np.array([gamma[i]])
            T_liquidus[i] = self._solver_sle._compute_component_liquidus(x_arr, g_arr, comp)[0]
            
        T_sle_envelope = np.max(T_liquidus)

        # 2. Rocket Performance Loop
        T_reactant = np.array([300.0] * (N + 1))
        pc_bar = self._cea_lib.units.psi_to_bar(self.pc_psi)
        
        mass_components = [x[i] * comp.mw for i, comp in enumerate(self.compounds)]
        total_fuel_mass = sum(mass_components)
        
        fuel_weights = np.array([m / total_fuel_mass for m in mass_components] + [0.0])
        ox_weights = np.array([0.0] * N + [1.0])
        
        of_ratio = self._reac.weight_eq_ratio_to_of_ratio(ox_weights, fuel_weights, self.phi)
        weights = self._reac.of_ratio_to_weights(ox_weights, fuel_weights, of_ratio)
        
        hc = self._reac.calc_property(self._cea_lib.ENTHALPY, weights, T_reactant) / self._cea_lib.R
        self._solver_rocket.solve(self._solution, weights, pc_bar, hc=hc, supar=self.supar, iac=True)
        
        # h_reactants_j = hc * self._cea_lib.R * T_reactant
        # h_products_j = self._solution.enthalpy * self._cea_lib.R * self._solution.T
        # h_combustion = h_reactants_j - h_products_j

        return {
            "x": x,
            "T_liquidus": T_liquidus,
            "T_sle": T_sle_envelope,
            "T_adiabatic": self._solution.T,
            "c_star": self._solution.c_star,
            "isp": self._solution.Isp,
            # "h_combustion": h_combustion
        }