# ===== data_loader.py (Full version with detailed comments) =====

import polars as pl  # Library for handling large datasets efficiently, like a supercharged spreadsheet tool
from pathlib import Path  # For handling file paths in a portable way
from typing import List, Dict, Optional, Tuple  # Tools to specify what types of data functions return or accept, making code clearer
import gc  # Library for manual garbage collection to free up memory during large operations
import numpy as np  # Library for working with numbers and matrices, like a calculator for arrays of data
from scipy.stats import spearmanr, pearsonr  # Functions from a science library to calculate correlations (not used directly here, but imported for potential extensions)
import warnings  # Library to control warning messages that Python might show
import os  # Library to interact with the operating system, here potentially for CPU info if needed
from core.correlation_calculator import CorrelationCalculator  # Import the correlation calculator class from another module
from concurrent.futures import ThreadPoolExecutor  # Tool to run multiple tasks at the same time (parallel processing) to speed things up
from tqdm import tqdm  # Library to show progress bars, so you can see how much of the task is done

warnings.filterwarnings('ignore')  # This line tells Python to ignore all warning messages, so the output is cleaner (but be careful, as it might hide important issues)

class BatchCorrelationLoader:
    """
    Loads and processes correlations - OPTIMIZED AND ROBUST VERSION
    - Purpose: This class handles loading financial asset data in batches, processing missing values, aggregating timeframes, and preparing data for correlation calculations.
    - Usage: Use it to load large numbers of assets without running out of memory, clean data, and build combined tables for analysis like correlations.
    - Benefits: Batches for efficiency, handles missing data without biasing (no zero-filling), and integrates with correlation tools.
    """

    def __init__(self, data_dir: str = "data", cache_dir: str = "cache/processed"):
        """
        Initializes the loader with directories.
        - Purpose: Sets up paths for data and cache folders.
        - Usage: Create an instance like `loader = BatchCorrelationLoader()` to start loading data.
        - Args:
            data_dir: Folder where asset data is stored (default 'data').
            cache_dir: Folder for cached processed data (default 'cache/processed').
        """
        self.data_dir = Path(data_dir)  # Path to the main data directory
        self.cache_dir = Path(cache_dir)  # Path to the cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # Create cache dir if it doesn't exist

    def _handle_missing_data(self, df: pl.DataFrame, strategy: str = "drop") -> pl.DataFrame:
        """
        Robust handling of missing data WITHOUT biasing with zeros.
        - Purpose: Cleans the DataFrame by dealing with nulls or infinite values based on a strategy, ensuring data quality without inventing values.
        - Usage: Called internally when loading assets; chooses how to handle gaps in data.
        - Behavior: For 'drop', removes rows with any nulls; for 'forward_fill' or 'backward_fill', fills gaps with previous/next values and then drops remaining nulls. Always filters out non-finite numbers (inf or NaN).
        - Args:
            df: The DataFrame to clean.
            strategy: How to handle nulls ('drop', 'forward_fill', or 'backward_fill').
        - Returns: Cleaned DataFrame.
        """
        if df.is_empty():  # If empty, return as is
            return df

        # Valid strategies for missing data
        valid_strategies = ["drop", "forward_fill", "backward_fill"]
        if strategy not in valid_strategies:  # Default to 'drop' if invalid
            strategy = "drop"

        numeric_cols = [col for col in df.columns if col != "open_time"]  # Get non-time columns

        if strategy == "drop":
            # Eliminate rows with ANY null
            df = df.drop_nulls()
        elif strategy == "forward_fill":
            df = df.with_columns([  # Fill forward (use previous value)
                pl.col(col).forward_fill() for col in numeric_cols
            ]).drop_nulls()  # Drop any remaining nulls (e.g., leading)
        elif strategy == "backward_fill":
            df = df.with_columns([  # Fill backward (use next value)
                pl.col(col).backward_fill() for col in numeric_cols
            ]).drop_nulls()  # Drop any remaining nulls (e.g., trailing)

        # Only keep finite values (remove inf or NaN) without filling zeros
        df = df.filter(
            pl.all_horizontal([pl.col(col).is_finite() for col in numeric_cols])
        )

        return df

    def load_assets_in_batches(self, assets: List[str], batch_size: int = 50,
                               timeframe: str = "1h", missing_strategy: str = "drop") -> Dict:
        """
        Loads all assets for full matrix with missing data strategy.
        - Purpose: Processes assets in smaller groups (batches) to avoid memory issues, loads and cleans each one.
        - Usage: Call like `loader.load_assets_in_batches(assets=['BTC', 'ETH'], batch_size=10)` for large lists.
        - Behavior: Tracks loaded and failed assets, stores cleaned DataFrames per asset.
        - Args:
            assets: List of asset names to load.
            batch_size: How many assets per batch (default 50).
            timeframe: Time interval to aggregate data to (e.g., '1h').
            missing_strategy: How to handle nulls (default 'drop').
        - Returns: Dictionary with loaded/failed lists, asset data, and stats.
        """
        n_assets = len(assets)  # Total assets
        n_batches = (n_assets + batch_size - 1) // batch_size  # Calculate batches needed

        results = {  # Dictionary to hold results
            'loaded_assets': [],  # Successful assets
            'failed_assets': [],  # Failed ones
            'asset_data': {},  # Cleaned DataFrames per asset
            'n_total': n_assets,
            'n_batches': n_batches,
            'timeframe': timeframe,
            'missing_strategy': missing_strategy
        }

        print(f"🚀 Loading {n_assets} assets in {n_batches} batches of {batch_size}")  # Inform user
        print(f"📊 Missing data strategy: {missing_strategy}")

        for batch_idx in range(n_batches):  # Loop over batches
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_assets)
            batch_assets = assets[start_idx:end_idx]  # Get current batch

            print(f"  Batch {batch_idx + 1}/{n_batches}: {len(batch_assets)} assets")

            for asset in batch_assets:  # Load each asset in batch
                asset_data = self._load_single_asset(asset, timeframe, missing_strategy)
                if asset_data is not None and not asset_data.is_empty():
                    results['loaded_assets'].append(asset)  # Success
                    results['asset_data'][asset] = asset_data
                else:
                    results['failed_assets'].append(asset)  # Fail

            if batch_idx % 3 == 0:  # Every 3 batches, clean memory
                gc.collect()

        print(f"✅ Loaded: {len(results['loaded_assets'])} assets")
        print(f"❌ Failed: {len(results['failed_assets'])} assets")

        return results

    def _load_single_asset(self, asset: str, timeframe: str, missing_strategy: str) -> Optional[pl.DataFrame]:
        """
        Loads a single asset with missing data strategy.
        - Purpose: Reads Parquet file for one asset, selects columns, cleans, aggregates timeframe.
        - Usage: Called internally in batches; skips if file missing or invalid.
        - Args:
            asset: Name of the asset (e.g., 'BTC').
            timeframe: Interval to aggregate (e.g., '1h').
            missing_strategy: How to handle nulls.
        - Returns: Aggregated DataFrame or None if failed.
        """
        try:
            file_path = self.data_dir / asset / f"{asset}.parquet"  # Path to file

            if not file_path.exists():  # Check if exists
                print(f"⚠️ {asset}: file does not exist")
                return None

            df = pl.read_parquet(file_path)  # Load file

            if "1min_returns" not in df.columns or "open_time" not in df.columns:  # Check required columns
                print(f"⚠️ {asset}: missing columns")
                return None

            # Normalize open_time to datetime[us]
            df = df.with_columns(
                pl.col("open_time").cast(pl.Datetime(time_unit="us")).alias("open_time")
            )

            df = df.select(["open_time", "1min_returns"])  # Select only needed

            if df.is_empty():  # Check if empty after select
                print(f"⚠️ {asset}: DataFrame empty after selecting columns")
                return None

            df = self._handle_missing_data(df, missing_strategy)  # Clean missing
            df_agg = self._aggregate_timeframe(df, timeframe)  # Aggregate
            df_agg = df_agg.rename({"1min_returns": f"{asset}_returns"})  # Rename column

            if df_agg.height < 15:  # Skip if too few rows
                print(f"⚠️ {asset}: less than 15 periods after aggregation")
                return None

            return df_agg

        except Exception as e:  # Catch any errors
            print(f"❌ Error loading {asset}: {str(e)[:50]}")
            return None

    def _aggregate_timeframe(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Aggregation by timeframe.
        - Purpose: Groups data by time intervals (e.g., sum returns every hour).
        - Usage: Called after cleaning; defaults to '1h' if invalid.
        - Args:
            df: DataFrame to aggregate.
            timeframe: Interval like '1h'.
        - Returns: Aggregated DataFrame.
        """
        time_map = {  # Mapping of timeframes
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

        if timeframe not in time_map:  # Default if invalid
            timeframe = "1h"

        df = df.filter(pl.col("open_time").is_not_null())  # Remove null times
        return df.group_by_dynamic("open_time", every=time_map[timeframe]).agg(  # Group and sum
            pl.col("1min_returns").sum()
        )

    def _safe_get_date_info(self, df: pl.DataFrame, asset: str) -> Dict:
        """
        Gets date information robustly.
        - Purpose: Extracts min/max dates and row count safely, even if DataFrame is empty.
        - Usage: For debugging or stats on asset data.
        - Args:
            df: DataFrame to check.
            asset: Asset name for context.
        - Returns: Dict with min_date, max_date, total_periods (or N/A if empty).
        """
        try:
            if df.is_empty() or "open_time" not in df.columns:  # Handle empty or missing column
                return {
                    'min_date': "N/A",
                    'max_date': "N/A",
                    'total_periods': 0
                }

            dates = df["open_time"]  # Get time column
            return {
                'min_date': dates.min(),
                'max_date': dates.max(),
                'total_periods': len(dates)
            }
        except Exception:  # Any error, return defaults
            return {
                'min_date': "N/A",
                'max_date': "N/A",
                'total_periods': 0
            }

    def _build_combined_dataframe(self, all_assets: List[str], asset_data: Dict) -> Optional[pl.DataFrame]:
        """
        Builds a combined DataFrame with all assets - FIXED.
        - Purpose: Joins all asset DataFrames into one big table aligned by time.
        - Usage: Called before correlations; handles outer joins and cleans up extra columns.
        - Behavior: Starts with first asset, joins others, cleans duplicate 'open_time' columns by coalescing (merging) them, and sorts by time.
        - Args:
            all_assets: List of loaded asset names.
            asset_data: Dict of DataFrames per asset.
        - Returns: Combined DataFrame or None if no assets.
        """
        if not all_assets:  # No assets, return None
            return None

        print("🔄 Building combined DataFrame...")

        # Start with the first asset
        first_asset = all_assets[0]
        combined_df = asset_data[first_asset]

        # Join the remaining assets with suffixes to avoid collisions
        for i, asset in enumerate(all_assets[1:], 1):
            df_to_join = asset_data[asset]
            combined_df = combined_df.join(
                df_to_join,
                on="open_time",
                how="outer",
                suffix=f"_{i}"  # Unique suffix for each join
            )

        # Clean extra open_time columns (the fix: merge them into one and drop extras)
        open_time_cols = [col for col in combined_df.columns if col.startswith('open_time_')]
        if open_time_cols:
            combined_df = combined_df.with_columns(
                pl.coalesce(['open_time'] + open_time_cols).alias('open_time')  # Merge all into 'open_time'
            ).drop(open_time_cols)  # Remove the extras

        # Sort by time
        combined_df = combined_df.sort("open_time")

        # Rename columns to remove any suffixes on returns (if present)
        column_mapping = {}
        for col in combined_df.columns:
            if col == "open_time":
                continue
            if "_returns_" in col:  # If suffix on returns, remove it
                original_name = col.rsplit('_', 1)[0]  # Remove last numeric suffix
                column_mapping[col] = original_name

        if column_mapping:
            combined_df = combined_df.rename(column_mapping)

        print(f"✅ Combined DataFrame: {combined_df.height} rows, {len([c for c in combined_df.columns if '_returns' in c])} assets")
        return combined_df

    def calculate_correlations_optimized(self, batch_results: Dict, method: str = 'Spearman',
                                         min_periods: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        NEW OPTIMIZED VERSION: Uses unified CorrelationCalculator.
        - Purpose: Builds combined DF and computes correlations using external calculator.
        - Usage: After loading batches, call to get correlation matrices.
        - Args:
            batch_results: Results from load_assets_in_batches.
            method: 'Spearman' or 'Pearson'.
            min_periods: Min data points per pair.
        - Returns: Corr matrix, p matrix, asset list, failed pairs.
        """
        print(f"🎯 CALCULATING MATRIX WITH SINGLE OPTIMIZED JOIN")

        all_assets = batch_results['loaded_assets']  # Loaded assets
        asset_data = batch_results['asset_data']  # Their data
        n_assets = len(all_assets)

        if n_assets < 2:  # Need at least 2
            raise ValueError("At least 2 assets needed for correlations")

        print(f"📊 Processing {n_assets} assets with {method}")
        print(f"🔢 Minimum threshold: {min_periods} periods")
        print(f"🔢 Total unique pairs: {n_assets * (n_assets - 1) // 2:,}")

        # Build the combined DataFrame
        combined_df = self._build_combined_dataframe(all_assets, asset_data)

        if combined_df is None or combined_df.is_empty():  # Check if valid
            raise ValueError("Could not build combined DataFrame")

        # Use CorrelationCalculator for computations
        calculator = CorrelationCalculator()
        corr_matrix, p_matrix, failed_pairs = calculator.compute_from_combined_dataframe(
            combined_df, all_assets, method, min_periods
        )

        print(f"✅ OPTIMIZED CORRELATIONS COMPLETED: {n_assets * (n_assets - 1) // 2} pairs")
        print(f"⚠️ Pairs without enough data: {len(failed_pairs)}")

        return corr_matrix, p_matrix, all_assets, failed_pairs

    # Keep original method for compatibility
    def calculate_correlations(self, batch_results: Dict, method: str = 'Spearman',
                               min_periods: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        Wrapper to keep compatibility.
        - Purpose: Calls the optimized version.
        """
        return self.calculate_correlations_optimized(batch_results, method, min_periods)

    def load_assets_for_rolling_analysis(self, assets: List[str], timeframe: str = "1h") -> Dict[str, pl.DataFrame]:
        """
        Loads specific assets for temporal analysis.
        - Purpose: Similar to batch load but for rolling (time-based) analysis, uses drop for missing.
        - Usage: For time-series studies.
        - Args:
            assets: List to load.
            timeframe: Aggregation interval.
        - Returns: Dict of DataFrames per asset.
        """
        results = {}  # Hold loaded data

        for asset in assets:
            try:
                file_path = self.data_dir / asset / f"{asset}.parquet"

                if not file_path.exists():  # Skip if missing
                    continue

                df = pl.read_parquet(file_path)

                if "1min_returns" not in df.columns or "open_time" not in df.columns:  # Check columns
                    continue

                # Normalize open_time
                df = df.with_columns(
                    pl.col("open_time").cast(pl.Datetime(time_unit="us")).alias("open_time")
                )

                df = df.select(["open_time", "1min_returns"])
                df = self._handle_missing_data(df, "drop")  # Clean with drop

                if df.is_empty():  # Skip empty
                    continue

                df_agg = self._aggregate_timeframe(df, timeframe)  # Aggregate
                df_agg = df_agg.rename({"1min_returns": f"{asset}_returns"})  # Rename

                results[asset] = df_agg  # Store

            except Exception as e:
                print(f"❌ Error loading {asset} for temporal analysis: {str(e)}")
                continue

        return results

    def load_assets_for_session_analysis(self, assets: List[str], timeframe: str = "1h") -> Dict[str, pl.DataFrame]:
        """
        Loads assets for session analysis.
        - Purpose: Similar to rolling, but for session-based (e.g., trading sessions).
        - Usage: Call for specific session studies.
        - Args:
            assets: List to load.
            timeframe: Aggregation interval.
        - Returns: Dict of DataFrames per asset.
        """
        results = {}

        for asset in assets:
            try:
                file_path = self.data_dir / asset / f"{asset}.parquet"

                if not file_path.exists():
                    continue

                df = pl.read_parquet(file_path)

                if "1min_returns" not in df.columns or "open_time" not in df.columns:
                    continue

                # Normalize open_time
                df = df.with_columns(
                    pl.col("open_time").cast(pl.Datetime(time_unit="us")).alias("open_time")
                )

                df = df.select(["open_time", "1min_returns"])
                df = self._handle_missing_data(df, "drop")  # Use drop by default

                if df.is_empty():
                    continue

                df_agg = self._aggregate_timeframe(df, timeframe)
                df_agg = df_agg.rename({"1min_returns": f"{asset}_returns"})

                results[asset] = df_agg

            except Exception as e:
                print(f"❌ Error loading {asset} for session analysis: {str(e)}")
                continue

        print(f"✅ Loaded {len(results)} assets for session analysis")
        return results


class ReturnsLoader:
    """
    Simple loader for returns data.
    - Purpose: Discovers and loads smaller sets of assets without batches.
    - Usage: For quick loads or discovery of available assets.
    """

    def __init__(self, data_dir: str, cache_dir: str):
        """
        Initializes with directories.
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def discover_assets(self):
        """
        Discovers available assets.
        - Purpose: Scans data dir for valid asset folders with non-empty Parquet.
        - Returns: Sorted list of asset names.
        """
        assets = []

        for folder in self.data_dir.iterdir():
            if folder.is_dir() and folder.name not in ['traditional', 'cache']:  # Exclude special dirs
                parquet_file = folder / f"{folder.name}.parquet"
                if parquet_file.exists() and parquet_file.stat().st_size > 100:  # Non-empty
                    assets.append(folder.name)

        return sorted(assets)

    def load(self, assets: list) -> pl.DataFrame:
        """
        Loads selected assets.
        - Purpose: Reads and joins small number of assets (limit 100).
        - Args:
            assets: List to load.
        - Returns: Combined sorted DataFrame.
        """
        if not assets:
            raise ValueError("Must select at least one asset")

        if len(assets) > 100:  # Limit for safety
            raise ValueError(
                f"Too many assets ({len(assets)}). "
                "Use BatchCorrelationLoader for >100 assets"
            )

        dfs = []  # List of DataFrames
        failed = []  # Failed assets

        for asset in assets:
            file_path = self.data_dir / asset / f"{asset}.parquet"
            try:
                if file_path.exists() and file_path.stat().st_size >= 12:  # Check size
                    df = pl.read_parquet(file_path)

                    if "1min_returns" not in df.columns or "open_time" not in df.columns:
                        print(f"⚠️ {asset}: missing columns")
                        failed.append(asset)
                        continue

                    # Normalize open_time
                    df = df.with_columns(
                        pl.col("open_time").cast(pl.Datetime(time_unit="us")).alias("open_time")
                    )

                    df = df.rename({"1min_returns": f"{asset}_return"})  # Note: singular 'return' here
                    dfs.append(df.select(["open_time", f"{asset}_return"]))
                else:
                    print(f"⚠️ {asset}: file does not exist or is empty")
                    failed.append(asset)
            except Exception as e:
                print(f"❌ Error reading {asset}: {str(e)[:50]}")
                failed.append(asset)

        if not dfs:
            raise ValueError(f"Could not load any asset. Failed: {failed}")

        if failed:
            print(f"⚠️ {len(failed)} assets could not be loaded: {failed[:10]}...")

        result = dfs[0]  # Start with first
        for df in dfs[1:]:  # Join others
            result = result.join(df, on="open_time", how="outer")

        return result.sort("open_time")  # Sort final

    def aggregate(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Resamples to target timeframe.
        - Purpose: Aggregates multiple columns by summing over intervals.
        - Args:
            df: DataFrame to aggregate.
            timeframe: Interval like '1h'.
        - Returns: Aggregated DataFrame.
        """
        time_map = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}

        if timeframe not in time_map:
            raise ValueError(f"Timeframe {timeframe} not supported. Options: {list(time_map.keys())}")

        if df["open_time"].null_count() > 0:  # Clean null times
            df = df.filter(pl.col("open_time").is_not_null())

        return df.group_by_dynamic("open_time", every=time_map[timeframe]).agg(  # Group and sum all non-time
            pl.col("*").exclude("open_time").sum()
        )