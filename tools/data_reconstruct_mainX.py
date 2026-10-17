
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sympy import lambdify
from typing import Callable

def data_reconstruct_mainX(data_description: pd.DataFrame, formula: Callable, target: np.ndarray,
                     tolerance: float = 1e-4, stat_tolerance: float = 5.1, max_global_attempts: int = 1000, seed = None) -> pd.DataFrame:
    """
    Reconstruct data variables that fit a specified formula close to each target value in `target`,
    and validate that reconstructed data has similar descriptive statistics to the original data description.

    Parameters:
    data_description (pd.DataFrame): Descriptive statistics for each variable in the data.
                                     Should contain mean, std, min, max, IQR, and count for each variable.
    formula (Callable): A function or sympy expression that takes in independent variables and outputs y_pred.
    target (np.ndarray): Array of target values for y_pred, which determine the number of rows to reconstruct.
    tolerance (float): Desired accuracy for approximated target. If |approx_target - target| < tolerance, stop iteration.
    stat_tolerance (float): Tolerance level for statistical comparison between final_data and data_description.
    max_global_attempts (int): Maximum number of optimization attempts to achieve desired statistical match.

    Returns:
    pd.DataFrame: Estimated values for each independent variable for each target value,
                  with columns for the target, approximated target, and loss, along with a validation report.
    """
    
    # Extract variable names from the data description
    variable_names = data_description.index.to_list()
    
    # Initialize an empty DataFrame to store all reconstructed data rows
    reconstructed_data = pd.DataFrame(columns=variable_names + ['target', 'approx_target', 'loss'])
    
    global_attempt = 0  # Track the global attempt count
    
    while global_attempt < max_global_attempts:
        reconstructed_data = pd.DataFrame(columns=variable_names + ['target', 'approx_target', 'loss'])
        
        # Attempt to reconstruct each row corresponding to each target value
        for t in target:
            best_loss = float('inf')
            best_approx_target = None
            best_vars = None

            # Define the objective function for each target value
            def objective_function(variables):
                y_pred = formula(*variables)  # Apply the formula with the given variables
                error = (y_pred - t) ** 2  # Squared error for this specific target
                return error

            # Initial guess based on the mean of each variable, adding small random noise to vary attempts
            initial_guess = data_description['mean'].values + np.random.normal(0, data_description['std'].values, len(variable_names))
            
            # Set bounds for each variable based on min and max in data description
            bounds = [(data_description.loc[var, 'min'], data_description.loc[var, 'max']) for var in variable_names]
            
            # Run the optimization for the current target value
            result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            # Calculate the approximated target and loss
            approximated_target = formula(*result.x)
            loss = np.abs(approximated_target - t)
            
            # Early stopping condition: if the approximation is within the tolerance, stop further optimization
            if loss < tolerance:
                best_vars = result.x
                best_approx_target = approximated_target
                best_loss = loss
            else:
                # If not within tolerance, save the best result obtained so far
                if loss < best_loss:
                    best_loss = loss
                    best_approx_target = approximated_target
                    best_vars = result.x

            row_data = list(best_vars) + [t, best_approx_target, best_loss]
            reconstructed_data = pd.concat(
                [reconstructed_data, pd.DataFrame([row_data], columns=variable_names + ['target', 'approx_target', 'loss'])],
                ignore_index=True
            )
        
        # Calculate the statistical description of the reconstructed data
        final_description = reconstructed_data[variable_names].describe().transpose()

        # Compare final_description with data_description
        stats_to_compare = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        validation_report = {}
        all_stats_close = True

        for var in variable_names:
            validation_report[var] = {}
            for stat in stats_to_compare:
                original_value = data_description.loc[var, stat]
                reconstructed_value = final_description.loc[var, stat]
                difference = np.abs(original_value - reconstructed_value)
                is_close = difference <= stat_tolerance * original_value  # Tolerance as a fraction of the original value
                validation_report[var][stat] = {
                    'original': original_value,
                    'reconstructed': reconstructed_value,
                    'difference': difference,
                    'is_close': is_close
                }
                if not is_close:
                    all_stats_close = False

        # Global early stop: If all statistics are within tolerance, stop the iterations
        if all_stats_close and best_loss < tolerance:
            print("Reconstruction succeeded within statistical tolerance.")
            break

        global_attempt += 1

    # Add validation summary
    validation_summary = "All statistics match within tolerance." if all_stats_close else "Some statistics do not match within tolerance."

    return reconstructed_data, final_description, validation_report, validation_summary
