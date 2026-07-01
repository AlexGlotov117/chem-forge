import csv
import numpy as np

from chemicals import Mixture

def exportSLE(filename: str, mixture: Mixture, x_matrix: np.ndarray, T_matrix: np.ndarray, T_sle: np.ndarray):
    """
    Dynamically generates and writes SLE phase data to a CSV file for N components.
    """
    # Build dynamic headers based on compound names
    headers = []
    for comp in mixture.compounds:
        headers.append(f"x_{comp.name}")
        
    for comp in mixture.compounds:
        headers.append(f"T_{comp.name} (K)")
        
    headers.append("T_SLE (K)")
    
    # Combine all arrays side-by-side into one large data matrix
    combined_data = np.column_stack((x_matrix, T_matrix, T_sle))
    
    # Write data to disk
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)      # Write the header row
        writer.writerows(combined_data)  # Write all data rows
        
    print(f"--> Phase diagram data successfully exported to: {filename}")
