# ===== correlation_calculator.py (Full version with detailed comments) =====

import polars as pl  # Library for handling large datasets efficiently, like a supercharged spreadsheet tool
from scipy.stats import spearmanr, pearsonr  # Functions from a science library to calculate correlations: Spearman for ranked data (good for non-linear relationships), Pearson for linear relationships
import numpy as np  # Library for working with numbers and matrices, like a calculator for arrays of data
from typing import Tuple, List, Dict  # Tools to specify what types of data functions return or accept, making code clearer
import warnings  # Library to control warning messages that Python might show
from concurrent.futures import ThreadPoolExecutor  # Tool to run multiple tasks at the same time (parallel processing) to speed things up
import os  # Library to interact with the operating system, here used to get the number of CPU cores
from tqdm import tqdm  # Library to show progress bars, so you can see how much of the task is done

warnings.filterwarnings('ignore')  # This line tells Python to ignore all warning messages, so the output is cleaner (but be careful, as it might hide important issues)

class CorrelationCalculator:
    """
    Calculates correlations - UNIFIED VERSION WITH DATA_LOADER
    - Purpose: This class helps compute how much two sets of data (like stock returns) move together.
    - Usage: Use it in financial analysis to see relationships between assets, helping with decisions like diversifying investments.
    - Benefits: Handles large data efficiently, skips bad pairs, and runs calculations in parallel for speed.
    """

    def compute_from_combined_dataframe(self, combined_df: pl.DataFrame, asset_names: List[str],
                                        method: str = 'Spearman', min_periods: int = 30) -> Tuple[
        np.ndarray, np.ndarray, List[Dict]]:
        """
        Calculates correlation matrix from an already combined DataFrame (used by data_loader).
        - Purpose: Takes a big table of data with returns for multiple assets and computes how correlated they are.
        - Usage: Call this method with a DataFrame that has columns like 'Asset1_returns', 'Asset2_returns', etc., and a list of asset names.
        - Behavior: It creates two matrices: one for correlation values (-1 to 1, where 1 means they move perfectly together) and one for p-values (how statistically significant the correlation is). Also tracks any pairs that couldn't be calculated.
        - Args:
            combined_df: The big table (DataFrame) with time and returns columns.
            asset_names: List of names like ['BTC', 'ETH'].
            method: 'Spearman' (default, good for rankings) or 'Pearson' (for direct linear links).
            min_periods: Minimum number of data points needed to calculate (default 30, to ensure reliability).
        - Returns: Correlation matrix (numbers array), p-value matrix, and list of failed pairs with reasons.
        """
        if method not in ['Spearman', 'Pearson']:  # Check if the method is valid
            raise ValueError(f"Method {method} not supported. Options: Spearman, Pearson")  # Error if not

        n = len(asset_names)  # Number of assets

        if n < 2:  # Need at least two to compare
            raise ValueError("At least 2 assets needed to calculate correlations")

        print(f"🎯 CALCULATING CORRELATIONS for {n} assets with {method}")  # Inform the user what's happening
        print(f"🔢 Minimum threshold: {min_periods} periods")

        # Create empty matrices: corr_matrix starts with 1s on diagonal (perfect self-correlation), p_matrix with 0s
        corr_matrix = np.eye(n)  # Identity matrix: 1 on diagonal, 0 elsewhere
        p_matrix = np.zeros((n, n))  # All zeros matrix for p-values
        failed_pairs = []  # List to store info on pairs that failed

        # Inner function to calculate for one pair of assets
        def compute_pair_correlation(i, j):
            """
            Helper function to compute correlation between two assets.
            - Purpose: For each pair, extract data, clean it, and calculate correlation if possible.
            - Usage: Called for every unique pair in parallel.
            """
            asset_i = asset_names[i]  # Get name of first asset
            asset_j = asset_names[j]  # Get name of second
            col_i = f"{asset_i}_returns"  # Column name for returns, e.g., 'BTC_returns'
            col_j = f"{asset_j}_returns"

            try:
                # Check if columns exist in the DataFrame
                if col_i not in combined_df.columns or col_j not in combined_df.columns:
                    failed_pairs.append({  # Add to failed list if missing
                        'Pair': f"{asset_i} vs {asset_j}",
                        'Reason': 'Missing columns in combined DataFrame',
                        'Asset1': asset_i,
                        'Asset2': asset_j,
                        'Common_Periods': 0
                    })
                    return 0.0, 1.0, i, j  # Default values: no correlation, insignificant

                # Select time and two columns, remove rows with missing values
                pair_df = combined_df.select(["open_time", col_i, col_j]).drop_nulls()

                if pair_df.height < min_periods:  # If too few rows left
                    failed_pairs.append({
                        'Pair': f"{asset_i} vs {asset_j}",
                        'Reason': f'Less than {min_periods} common periods (have {pair_df.height})',
                        'Asset1': asset_i,
                        'Asset2': asset_j,
                        'Common_Periods': pair_df.height
                    })
                    return 0.0, 1.0, i, j

                # Get the data as arrays for calculation
                x = pair_df[col_i].to_numpy()
                y = pair_df[col_j].to_numpy()

                # Calculate based on method
                if method == 'Spearman':
                    rho, p_val = spearmanr(x, y, nan_policy='omit')  # Spearman: ranks data, omits NaNs
                else:
                    rho, p_val = pearsonr(x, y)  # Pearson: direct calculation

                # If result is NaN (not a number), set defaults
                if np.isnan(rho):
                    failed_pairs.append({
                        'Pair': f"{asset_i} vs {asset_j}",
                        'Reason': 'NaN result in correlation calculation',
                        'Asset1': asset_i,
                        'Asset2': asset_j,
                        'Common_Periods': pair_df.height
                    })
                    rho, p_val = 0.0, 1.0

            except Exception as e:  # Catch any other errors
                error_msg = str(e)[:100]  # Shorten error message
                print(f"❌ Error calculating {asset_i} vs {asset_j}: {error_msg}")
                failed_pairs.append({
                    'Pair': f"{asset_i} vs {asset_j}",
                    'Reason': f'Error in calculation: {error_msg}',
                    'Asset1': asset_i,
                    'Asset2': asset_j,
                    'Common_Periods': 0
                })
                rho, p_val = 0.0, 1.0

            return rho, p_val, i, j  # Return results and indices

        # Calculate number of unique pairs (half the matrix, since symmetric)
        total_pairs = n * (n - 1) // 2
        processed_pairs = 0

        print(f"🔁 Processing {total_pairs} unique pairs...")

        # Use parallel processing to speed up
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:  # Up to 8 workers or CPU count
            futures = []  # List of tasks
            for i in range(n):  # Loop over upper triangle
                for j in range(i + 1, n):
                    futures.append(executor.submit(compute_pair_correlation, i, j))  # Submit task

            for future in tqdm(futures, desc="Calculating correlations"):  # Progress bar
                rho, p_val, i, j = future.result()  # Get result
                corr_matrix[i, j] = corr_matrix[j, i] = rho  # Fill both sides of matrix
                p_matrix[i, j] = p_matrix[j, i] = p_val
                processed_pairs += 1

        print(f"✅ CORRELATIONS COMPLETED: {processed_pairs} pairs processed")
        print(f"⚠️ Failed pairs: {len(failed_pairs)}")

        return corr_matrix, p_matrix, failed_pairs

    def compute_legacy(self, df: pl.DataFrame, method: str = 'Spearman') -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        LEGACY METHOD - Kept for compatibility with old code.
        - Purpose: Older way to calculate correlations, simpler but less robust.
        - Usage: Use if you have old code; otherwise, prefer the new method.
        - Behavior: Assumes DataFrame has 'open_time' and return columns directly (no '_returns' suffix).
        """
        print("⚠️ Using legacy method - Consider migrating to compute_from_combined_dataframe")

        if method not in ['Spearman', 'Pearson']:
            raise ValueError(f"Method {method} not supported. Options: Spearman, Pearson")

        cols = [col for col in df.columns if col != "open_time"]  # Get all columns except time
        n = len(cols)

        if n < 2:
            raise ValueError("At least 2 assets needed to calculate correlations")

        # Matrices like before
        corr_np = np.eye(n)
        p_np = np.zeros((n, n))

        # Loop over pairs (no parallel here)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    col_i = cols[i]
                    col_j = cols[j]

                    # Clean each column separately (note: may not align times perfectly)
                    data_i = df[col_i].drop_nulls().to_numpy()
                    data_j = df[col_j].drop_nulls().to_numpy()

                    min_len = min(len(data_i), len(data_j))
                    if min_len < 10:  # Hardcoded threshold
                        rho, p_val = 0.0, 1.0
                    else:
                        x = data_i[:min_len]  # Slice to same length (potential misalignment)
                        y = data_j[:min_len]

                        if method == 'Spearman':
                            rho, p_val = spearmanr(x, y, nan_policy='omit')
                        else:
                            rho, p_val = pearsonr(x, y)

                        if np.isnan(rho):
                            rho, p_val = 0.0, 1.0

                except Exception as e:
                    print(f"❌ Error in {cols[i]} vs {cols[j]}: {str(e)[:50]}")
                    rho, p_val = 0.0, 1.0

                corr_np[i, j] = corr_np[j, i] = rho
                p_np[i, j] = p_np[j, i] = p_val

        # Convert to Polars DataFrames with column names
        corr_matrix = pl.DataFrame(corr_np, schema=cols)
        p_matrix = pl.DataFrame(p_np, schema=cols)

        return corr_matrix, p_matrix

    # Alias for old code to still work
    compute = compute_legacy