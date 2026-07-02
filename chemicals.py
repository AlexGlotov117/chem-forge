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
        Pure data allocation. No heavy computational solvers, packages, 
        or chemistry engines are instantiated here.
        """
        # Context/State Tracking
        self._current_x: Optional[np.ndarray] = None
        self._current_gamma: Optional[np.ndarray] = None
        
        # Split Execution Caches
        self._sle_cache: Dict[str, Any] = {}
        self._cea_cache: Dict[str, Any] = {}
        
        # Lazy Solver Handles (Kept completely unallocated at start)
        self._solver_sle = None
        self._cea_lib = None
        self._reac = None
        self._solver_rocket = None
        self._solution = None

    # -------------------------------------------------------------------------
    # Just-In-Time Solver Spin Ups
    # -------------------------------------------------------------------------
    def _init_sle_engine(self):
        """Instantiates the native SLE solver only when needed."""
        if self._solver_sle is not None:
            return
        from solvers import SLESolver
        self._solver_sle = SLESolver()

    def _init_cea_engine(self):
        """Loads CEA library and builds rocket mechanisms only when needed."""
        if self._cea_lib is not None:
            return  
            
        import cea
        self._cea_lib = cea  
        fuels_cea = [comp.cea_reactant for comp in self.compounds]
        reac_names = fuels_cea + [self.oxidizer_name]
        
        self._reac = cea.Mixture(reac_names)
        prod = cea.Mixture(reac_names, products_from_reactants=True)
        
        self._solver_rocket = cea.RocketSolver(prod, reactants=self._reac)
        self._solution = cea.RocketSolution(self._solver_rocket)
    
    # -------------------------------------------------------------------------
    # State & Caching Logic
    # -------------------------------------------------------------------------
    def set_composition(self, x: np.ndarray, gamma: Optional[np.ndarray] = None, force_recalc: bool = False):
        x = np.asarray(x)
        if gamma is None:
            gamma = np.ones_like(x)
        else:
            gamma = np.asarray(gamma)

        state_changed = (self._current_x is None or 
                         not np.allclose(self._current_x, x) or 
                         not np.allclose(self._current_gamma, gamma))

        if state_changed or force_recalc:
            self._current_x = x.copy()
            self._current_gamma = gamma.copy()
            self._sle_cache.clear()
            self._cea_cache.clear()

    # -------------------------------------------------------------------------
    # Isolated On-Demand Evaluation Triggers
    # -------------------------------------------------------------------------
    def _ensure_sle_evaluated(self):
        if self._current_x is None:
            raise ValueError("Set composition first via mixture.set_composition()")
        if self._sle_cache:
            return

        # Explicitly initialize the solver right before using it
        self._init_sle_engine()

        N = len(self.compounds)
        T_liquidus = np.zeros(N)
        for i, comp in enumerate(self.compounds):
            T_liquidus[i] = self._solver_sle._compute_component_liquidus(
                np.array([self._current_x[i]]), np.array([self._current_gamma[i]]), comp
            )[0]
        
        self._sle_cache = {"Component Liquidus Temperature": T_liquidus, 
                           "Solid-Liquid Equilibrium Temperature": np.max(T_liquidus)}

    def _ensure_cea_evaluated(self):
        if self._current_x is None:
            raise ValueError("Set composition first via mixture.set_composition()")
        if self._cea_cache:
            return

        # Explicitly initialize CEA right before running combustion loops
        self._init_cea_engine()

        N = len(self.compounds)
        T_reactant = np.array([300.0] * (N + 1))
        pc_bar = self._cea_lib.units.psi_to_bar(self.pc_psi)
        
        mass_components = [self._current_x[i] * comp.mw for i, comp in enumerate(self.compounds)]
        total_fuel_mass = sum(mass_components)
        
        fuel_weights = np.array([m / total_fuel_mass for m in mass_components] + [0.0])
        ox_weights = np.array([0.0] * N + [1.0])
        
        of_ratio = self._reac.weight_eq_ratio_to_of_ratio(ox_weights, fuel_weights, self.phi)
        weights = self._reac.of_ratio_to_weights(ox_weights, fuel_weights, of_ratio)
        
        hc = self._reac.calc_property(self._cea_lib.ENTHALPY, weights, T_reactant) / self._cea_lib.R
        self._solver_rocket.solve(self._solution, weights, pc_bar, hc=hc, supar=self.supar, iac=True)
        
        h_reactants_j = hc * self._cea_lib.R * T_reactant
        h_products_j = self._solution.enthalpy * self._cea_lib.R * self._solution.T

        self._cea_cache = {
            "Adiabatic Flame Temperature": self._solution.T,
            "Characteristic Velocity": self._solution.c_star,
            "Specific Impulse": self._solution.Isp,
            # "h_combustion": h_reactants_j - h_products_j
        }

    # -------------------------------------------------------------------------
    # Completely Decoupled Public Properties
    # -------------------------------------------------------------------------
    @property
    def T_fus(self) -> float:
        self._ensure_sle_evaluated()
        return self._sle_cache["Solid-Liquid Equilibrium Temperature"]

    @property
    def T_liq(self) -> np.ndarray:
        self._ensure_sle_evaluated()
        return self._sle_cache["Component Liquidus Temperature"]

    @property
    def T_adi(self) -> float:
        self._ensure_cea_evaluated()
        return self._cea_cache["Adiabatic Flame Temperature"]

    @property
    def c_star(self) -> float:
        self._ensure_cea_evaluated()
        return self._cea_cache["Characteristic Velocity"]

    @property
    def isp(self) -> float:
        self._ensure_cea_evaluated()
        return self._cea_cache["Specific Impulse"]

    # @property
    # def H_combustion(self) -> float:
    #     self._ensure_cea_evaluated()
    #     return self._cea_cache["h_combustion"]