import numpy as np
import math

from itertools import combinations_with_replacement
from chemicals import Compound, Mixture

def generate_x_grid(num_components: int, steps: int) -> np.ndarray:
    """
    Generates an array of shape (M, num_components) representing all valid 
    mole fraction combinations where the sum of components equals 1.0.
    """
    # Formula for total combinations: nCr(steps + num_components - 2, num_components - 1)
    M = math.comb(steps + num_components - 2, num_components - 1)
    x_matrix = np.zeros((M, num_components))
    grid_ticks = steps - 1
    
    # Each stack item stores: (component_index, remaining_ticks, history_list)
    # This prevents different branches from overwriting each other's data
    stack = [(0, grid_ticks, [])]
    row_idx = 0
    
    while stack:
        c_idx, remaining, history = stack.pop()
        
        # Base case: We reached the final component, which takes all leftovers
        if c_idx == num_components - 1:
            full_path = history + [remaining]
            for i in range(num_components):
                x_matrix[row_idx, i] = full_path[i] / grid_ticks
            row_idx += 1
            continue
            
        # Push options onto the stack, appending choices to a local history copy
        for ticks in range(0, remaining + 1):
            stack.append((c_idx + 1, remaining - ticks, history + [ticks]))
            
    return x_matrix

def generate_gamma_grid(num_components: int, steps: int) -> np.ndarray:
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
        T = np.full_like(x_i, comp.T_fus)
        
        # Avoid runtime warnings safely by masking pure 0 entries
        mask = x_i > 0
        if not np.any(mask):
            return T
            
        # Calculate high temperature behavior (above T_ss)
        # Equation: T = h_fus / (h_fus/t_fus - R*ln(x_i))
        T_high = comp.h_fus / ((comp.h_fus / comp.T_fus) - self.R * np.log(x_i[mask]*gamma_i[mask]))
        
        # Calculate low temperature behavior (below T_ss)
        # Using the exact analytical conversion logic in your loop:
        # A = -comp.t_fus * comp.h_fus * comp.t_ss - comp.t_fus * comp.h_ss * comp.t_ss
        # B = comp.t_fus * comp.t_ss * self.R * np.log(x_i[mask]*gamma_i[mask])
        # C = -comp.h_fus * comp.t_ss - comp.h_ss * comp.t_fus
        # T_low = A / (B + C)

        T[mask] = T_high # np.where(T_high < comp.t_ss, T_low, T_high)
        T[~mask] = 0.0 
        return T

    # def solve(self, mixture: Mixture, steps: int = 11) -> np.ndarray:
    #     """
    #     Computes individual liquidus trajectories and extracts the collective SLE envelope.

    #     Inputs:
    #         mixture (Mixture): The chemical system containing compounds and active parameters.
    #         steps (int): The number of resolution ticks mapped across the composition grid.

    #     Returns:
    #         tuple: A structural payload containing three distinct matrix collections:
    #             - x_matrix (np.ndarray): Shape (M, N) containing the mole fractions for all N components.
    #             - T_liquidus_matrix (np.ndarray): Shape (M, N) tracking the independent melting curves.
    #             - T_sle (np.ndarray): Shape (M, 1) tracking the final unified solid-liquid equilibrium boundary.
    #     """
    #     # Generate multi-dimensional compositional space and activity coefficient matrix
    #     x_matrix = generate_x_grid(mixture.num_components, steps)
    #     gamma_matrix = generate_gamma_grid(mixture.num_components, steps)
    #     M = x_matrix.shape[0]
        
    #     # Compute individual liquidus temperatures for every single component
    #     T_matrix = np.zeros((M, mixture.num_components))
        
    #     for i, comp in enumerate(mixture.compounds):
    #         T_matrix[:, i] = self._compute_component_liquidus(x_matrix[:, i], gamma_matrix[:, i], comp)
            
    #     # Apply the maximum thermodynamic envelope principle
    #     T_sle = np.max(T_matrix, axis=1)
        
    #     return x_matrix, T_matrix, T_sle