from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class Compound:
    name: str
    smiles: str 
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
    ref_temperature: float = 298

    def extract_formula_from_smiles(self, smiles_str: Optional[str] = None) -> dict:
        from rdkit import Chem
        target_smiles = smiles_str or self.smiles
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None: raise ValueError(f"Invalid SMILES: {target_smiles}")
        mol = Chem.AddHs(mol)
        formula_dict = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol().upper()
            formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
        return formula_dict
        

    @property
    def cea_reactant(self):
        """
        Dynamically generates the NASA-CEA Reactant object using 
        the dataclass attributes on-demand.
        """
        import cea  # Imported inline to keep chemical.py decoupled if CEA isn't used
        
        if not self.formula:
            if self.smiles:
                self.formula = self.extract_formula_from_smiles(self.smiles)
            else:
                raise ValueError(f"Chemical formula dictionary missing for CEA component: {self.name}")
            
        return cea.Reactant(
            name=f"{self.name.strip()[:15]}",
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
        self._current_hE: Optional[float] = None
        
        # Split Execution Caches
        self._sle_cache: Dict[str, Any] = {}
        self._cea_cache: Dict[str, Any] = {}
        self._hanna_curve_cache: Dict[str, Any] = {}
        
        # Lazy Solver Handles (Kept completely unallocated at start)
        self._solver_sle = None
        self._nonideal_predictor = None
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
        from models.solvers import SLESolver
        self._solver_sle = SLESolver()

    def _init_nonideal_engine(self):
        """Instantiates the HANNA non-ideal activity coefficient predictor on demand."""
        if self._nonideal_predictor is not None:
            return
        from models.HANNA2.utils.HANNA_predictor import HANNA_Predictor
        self._nonideal_predictor = HANNA_Predictor()

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

    def predict_composition_space(self, temperature: float = 298.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Queries HANNA over the full composition grid using `predict_over_composition`.
        Caches the raw curves and returns (molar_fractions_all, ln_gammas, gE, hE).
        """
        smiles_list = [c.smiles for c in self.compounds]
        if any(not s for s in smiles_list):
            raise ValueError("All compounds in mixture must have a valid SMILES string defined to predict gammas.")
            
        self._init_nonideal_engine()
        
        molar_fractions_all, ln_gammas, gE, hE = self._nonideal_predictor.predict_over_composition(
            smiles_list=smiles_list,
            temperature=temperature
        )
        
        # Cache results for composition lookups / interpolation
        self._hanna_curve_cache = {
            "molar_fractions_all": np.asarray(molar_fractions_all),
            "ln_gammas": np.asarray(ln_gammas),
            "gammas": np.exp(np.asarray(ln_gammas)),
            "gE": np.asarray(gE),
            "hE": np.asarray(hE),
            "temperature": temperature
        }
        
        return molar_fractions_all, ln_gammas, gE, hE

    def interpolate_thermo_properties(
        self, 
        x: np.ndarray, 
        temperature: float = 300.0,
        gamma_scaling_alpha: Optional[np.ndarray] = None,
        scaling_profile: str = "exponential",
        steepness_k: float = 15.0
    ) -> tuple[np.ndarray, float]:
        """
        Interpolates gamma and excess enthalpy (hE), applying a composition-dependent
        scaling profile that anchors at 1.0 for x_i = 1.0 and steeply adjusts away from pure composition.
        
        Parameters:
        -----------
        gamma_scaling_alpha : array-like or None
            Target fractional scaling shift at mixture conditions (e.g. [0.20, -0.10]).
        scaling_profile : str
            'exponential' (default steep drop/climb), 'quadratic', or 'linear'.
        steepness_k : float
            Steepness factor for exponential drop (higher values = sharper crash near x_i = 1.0).
        """
        x = np.asarray(x, dtype=float)
        
        # Ensure HANNA curve cache matches requested temperature
        if not self._hanna_curve_cache or self._hanna_curve_cache.get("temperature") != temperature:
            self.predict_composition_space(temperature=temperature)
            
        grid_x = self._hanna_curve_cache["molar_fractions_all"]
        grid_gammas = self._hanna_curve_cache["gammas"]
        grid_hE = self._hanna_curve_cache["hE"]

        # Interpolate raw predictions
        if grid_x.ndim == 1 or grid_x.shape[1] == 1:
            x1_grid = grid_x.flatten()
            x1_target = x[0]
            
            gamma = np.array([
                np.interp(x1_target, x1_grid, grid_gammas[:, i]) 
                for i in range(len(self.compounds))
            ])
            hE_val = float(np.interp(x1_target, x1_grid, grid_hE.flatten()))
        else:
            distances = np.linalg.norm(grid_x - x, axis=1)
            idx = np.argmin(distances)
            gamma = grid_gammas[idx]
            hE_val = float(grid_hE[idx])

        # Apply steep composition-dependent scaling
        if gamma_scaling_alpha is not None:
            alpha = np.asarray(gamma_scaling_alpha, dtype=float)
            if alpha.shape[0] != len(self.compounds):
                raise ValueError(f"gamma_scaling_alpha must match number of compounds ({len(self.compounds)})")
            
            sum_other = 1.0 - x
            
            if scaling_profile == "exponential":
                # Crashes/climbs steeply as soon as sum_other > 0 (x_i < 1.0)
                f_x = 1.0 - np.exp(-steepness_k * sum_other)
            elif scaling_profile == "quadratic":
                f_x = sum_other ** 2
            elif scaling_profile == "linear":
                f_x = sum_other
            else:
                raise ValueError(f"Unknown scaling profile: {scaling_profile}")
                
            scaling_factors = 1.0 + alpha * f_x
            gamma = gamma * scaling_factors

        return gamma, hE_val
    
    def set_composition(
        self, 
        x: np.ndarray, 
        gamma: Optional[np.ndarray] = None, 
        hE: Optional[float] = None,
        use_hanna: bool = False,
        temperature: float = 300.0,
        gamma_scaling_alpha: Optional[np.ndarray] = None,
        steepness_k: float = 15.0,
        scaling_profile: str = "exponential",
        force_recalc: bool = False
    ):
        x = np.asarray(x, dtype=float)
        
        if use_hanna:
            pred_gamma, pred_hE = self.interpolate_thermo_properties(
                x=x, 
                temperature=temperature,
                gamma_scaling_alpha=gamma_scaling_alpha,
                scaling_profile=scaling_profile,
                steepness_k = steepness_k
            )
            gamma = pred_gamma if gamma is None else np.asarray(gamma, dtype=float)
            hE = pred_hE if hE is None else float(hE)
        else:
            gamma = np.ones_like(x) if gamma is None else np.asarray(gamma, dtype=float)
            hE = 0.0 if hE is None else float(hE)

        state_changed = (self._current_x is None or 
                        not np.allclose(self._current_x, x) or 
                        not np.allclose(self._current_gamma, gamma) or
                        self._current_hE != hE)

        if state_changed or force_recalc:
            self._current_x = x.copy()
            self._current_gamma = gamma.copy()
            self._current_hE = hE
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

        self._init_cea_engine()

        N = len(self.compounds)
        T_reactant = np.array([298.0] * (N + 1))
        pc_bar = self._cea_lib.units.psi_to_bar(self.pc_psi)
        
        mass_components = [self._current_x[i] * comp.mw for i, comp in enumerate(self.compounds)]
        total_fuel_mass = sum(mass_components)
        
        fuel_weights = np.array([m / total_fuel_mass for m in mass_components] + [0.0])
        ox_weights = np.array([0.0] * N + [1.0])
        
        of_ratio = self._reac.weight_eq_ratio_to_of_ratio(ox_weights, fuel_weights, self.phi)
        weights = self._reac.of_ratio_to_weights(ox_weights, fuel_weights, of_ratio)
        
        hc = self._reac.calc_property(self._cea_lib.ENTHALPY, weights, T_reactant) / self._cea_lib.R

        # Incorporate Excess Enthalpy of Mixing (hE) into CEA Enthalpy Pool if present
        if self._current_hE is not None and self._current_hE != 0.0:
            # hE in J/mol converted to dimensionless enthalpy (hc) for CEA
            hc += self._current_hE / self._cea_lib.R

        # Extract liquidus temperature for phase transition correction
        self._ensure_sle_evaluated()
        mixture_melting_point = self._sle_cache["Solid-Liquid Equilibrium Temperature"]
        
        if mixture_melting_point <= 298.15:
            delta_h_phase_change = 0.0
            
            for i, comp in enumerate(self.compounds):
                if comp.T_fus > 298.0:
                    delta_h_phase_change += self._current_x[i] * (comp.h_fus / self._cea_lib.R)
            
            hc += delta_h_phase_change

        self._solver_rocket.solve(self._solution, weights, pc_bar, hc=hc, supar=self.supar, iac=True)
        
        self._cea_cache = {
            "Adiabatic Flame Temperature": self._solution.T,
            "Characteristic Velocity": self._solution.c_star,
            "Specific Impulse": self._solution.Isp/9.81,
        }

    # -------------------------------------------------------------------------
    # Completely Decoupled Public Properties
    # -------------------------------------------------------------------------
    @property
    def num_components(self) -> int:
        return len(self.compounds)
    
    @property
    def names(self) -> List[str]:
        return [c.name for c in self.compounds]
    
    @property
    def current_gamma(self) -> Optional[np.ndarray]:
        return self._current_gamma

    @property
    def current_hE(self) -> Optional[float]:
        return self._current_hE

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