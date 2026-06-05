import numpy as np
import math

from itertools import combinations_with_replacement
from chemicals import Compound, Mixture

def generate_x_grid(num_components: int, steps: int) -> np.ndarray:
    """
    Generates an array of shape (M, num_components) representing all valid mole fraction 
    combinations on a grid where the sum of components equals 1.0.
    """

    # Formula: nCr(steps + num_components - 2, num_components - 1)
    M = math.comb(steps + num_components - 2, num_components - 1)
    
    x_matrix = np.zeros((M, num_components))


    # # Generate all integer combinations that sum to (steps - 1)
    # grid_ticks = steps - 1
    # comps = []
    
    # for combo in combinations_with_replacement(range(grid_ticks + 1), num_components - 1):
    #     # Calculate individual coordinate elements
    #     combo = (0,) + combo + (grid_ticks,)
    #     fractions = [np.float64(combo[i+1] - combo[i]) / grid_ticks for i in range(num_components)]
    #     comps.append(fractions)
        
    return np.array(x_matrix)

def create_gamma_matrix(num_components: int, steps: int, gp_models: list = None, **gp_kwargs) -> np.ndarray:
    """
    Generates an array of shape (M, num_components) representing all activity coefficient 
    combinations on a grid
    """
    # Formula: nCr(steps + num_components - 2, num_components - 1)
    M = math.comb(steps + num_components - 2, num_components - 1)
    
    gamma_matrix = np.ones((M, num_components))
        
    return gamma_matrix

class SLESolver:
    R = 8.31446  # Ideal gas constant
    
    def _compute_component_liquidus(self, x_i: np.ndarray, gamma_i: np.ndarray, comp: Compound) -> np.ndarray:
        """
        Computes the theoretical liquidus temperature profile for a single component 
        given its mole fraction vector array x_i and the activity coefficient vectory array gamma_i.
        """
        # Initialize array with pure melting temperature
        T = np.full_like(x_i, comp.t_fus)
        
        # Avoid runtime warnings safely by masking pure 0 entries
        mask = x_i > 0
        if not np.any(mask):
            return T
            
        # Calculate high temperature behavior (above T_ss)
        # Equation: T = h_fus / (h_fus/t_fus - R*ln(x_i))
        T_high = comp.h_fus / ((comp.h_fus / comp.t_fus) - self.R * np.log(x_i[mask]*gamma_i[mask]))
        
        # Calculate low temperature behavior (below T_ss)
        # Using the exact analytical conversion logic in your loop:
        A = -comp.t_fus * comp.h_fus * comp.t_ss - comp.t_fus * comp.h_ss * comp.t_ss
        B = comp.t_fus * comp.t_ss * self.R * np.log(x_i[mask]*gamma_i[mask])
        C = -comp.h_fus * comp.t_ss - comp.h_ss * comp.t_fus
        T_low = A / (B + C)

        T[mask] = np.where(T_high < comp.t_ss, T_low, T_high)
        T[~mask] = 0.0 
        
        return T

    def solve(self, mixture: Mixture, steps: int = 11) -> np.ndarray:
        """
        Computes the SLE matrix across the entire composition space.
        
        Returns:
            numpy array: Shape (M, N + 1) where columns are [x_1, x_2, ... x_N, T_SLE]
        """
        # Generate multi-dimensional compositional space matrix
        x_matrix = generate_simplex_grid(mixture.num_components, steps=steps)
        M = x_matrix.shape[0]
        
        # Compute individual liquidus temperatures for every single component
        T_liquidus_matrix = np.zeros((M, mixture.num_components))
        
        for i, comp in enumerate(mixture.compounds):
            T_liquidus_matrix[:, i] = self._compute_component_liquidus(x_matrix[:, i], gamma_matrix[:, i], comp)
            
        # Apply the maximum thermodynamic envelope principle
        t_sle = np.max(T_liquidus_matrix, axis=1)
        
        return np.column_stack((x_matrix, t_sle))