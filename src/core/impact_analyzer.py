# ===== impact_analyzer.py (Full version with detailed comments) =====

import polars as pl  # Library for handling large datasets efficiently, like a supercharged spreadsheet tool
import numpy as np  # Library for working with numbers and matrices, like a calculator for arrays of data
from scipy import stats  # Library for statistical functions, used here for linear regression and correlations
from typing import Dict, List, Tuple, Optional  # Tools to specify what types of data functions return or accept, making code clearer
from dataclasses import dataclass  # Simple way to create classes that mainly hold data
import warnings  # Library to control warning messages that Python might show

warnings.filterwarnings('ignore')  # This line tells Python to ignore all warning messages, so the output is cleaner (but be careful, as it might hide important issues)

@dataclass
class ImpactResult:
    """
    Result of the impact analysis.
    - Purpose: Holds the key metrics from analyzing how one asset affects another, like sensitivity (beta) and responses to big changes.
    - Usage: Created for each target asset to store and access results easily.
    """
    asset_name: str  # Name of the target asset being analyzed
    beta: float  # Measure of how much the target moves with the reference (e.g., 1.2 means 20% more volatile)
    beta_ci_lower: float  # Lower bound of the confidence interval for beta
    beta_ci_upper: float  # Upper bound of the confidence interval for beta
    r_squared: float  # How well the model fits the data (0-1, higher is better)
    p_value: float  # Statistical significance (low means reliable)
    n_points: int  # Number of data points used
    mean_response_up: float  # Average response when reference goes up big
    mean_response_down: float  # Average response when reference goes down big
    median_response_up: float  # Median response for up events
    median_response_down: float  # Median response for down events
    persistence_up: float  # % of times response follows expected direction for up
    persistence_down: float  # % for down
    avg_consecutive_up: float  # Average consecutive positive responses for up
    avg_consecutive_down: float  # For down
    overreaction_score: float  # >1 means overreaction (actual vs expected)
    asymmetry_ratio: float  # Ratio of down to up responses (measures imbalance)

@dataclass
class CascadeResult:
    """
    Result of the cascade analysis.
    - Purpose: Holds metrics on how quickly an asset responds to the reference (propagation order).
    - Usage: For ordering assets by response speed in a "cascade" effect.
    """
    asset_name: str  # Name of the asset
    optimal_lag: int  # Best time delay for max correlation
    max_correlation: float  # Highest correlation found
    response_time_avg: float  # Weighted average response time
    cascade_order: int  # Order in the propagation chain (lower = faster)

class BTCImpactAnalyzer:
    """
    Analyzes the dynamic impact of a reference asset (e.g., BTC) on other assets.
    - Purpose: Computes sensitivities, responses to events, propagation, and more for financial time series.
    - Usage: Create like `analyzer = BTCImpactAnalyzer()` and call methods with data.
    """

    def __init__(self, reference_asset: str = "BTCUSDT"):
        """
        Initializes with reference asset.
        - Args:
            reference_asset: Name of the main asset (default 'BTCUSDT').
        """
        self.reference_asset = reference_asset  # The main asset to compare others against

    def resample_to_timeframe(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Resamples 1m returns to desired timeframe.
        - Purpose: Aggregates data to larger intervals (e.g., sum returns every 15m).
        - Usage: Call with a DataFrame having 'open_time' and '_returns' columns.
        - Behavior: Groups by time and sums returns.
        - Args:
            df: DataFrame with time and returns.
            timeframe: '5m', '10m', '15m', '30m', '1h'.
        - Returns: Resampled DataFrame.
        """
        timeframe_map = {  # Valid intervals
            '5m': '5m',
            '10m': '10m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h'
        }

        if timeframe not in timeframe_map:  # Check valid
            raise ValueError(f"Timeframe {timeframe} not supported")

        # Ensure open_time is datetime
        df = df.with_columns(
            pl.col("open_time").cast(pl.Datetime(time_unit="us"))
        )

        # Get returns columns
        return_cols = [col for col in df.columns if col.endswith('_returns')]

        resampled = df.group_by_dynamic(  # Group by time interval
            "open_time",
            every=timeframe_map[timeframe]
        ).agg([
            pl.col(col).sum() for col in return_cols  # Sum returns
        ])

        return resampled.sort("open_time")  # Sort by time

    def calculate_general_beta(self,
                               ref_returns: np.ndarray,
                               target_returns: np.ndarray,
                               bootstrap_iterations: int = 1000,
                               seed: Optional[int] = 42) -> Dict:
        """
        Calculates general beta with confidence intervals - FIXED WITH SEED.
        - Purpose: Measures overall sensitivity (beta) using regression, with bootstrap for reliability.
        - Usage: Pass arrays of returns; uses seed for repeatable results.
        - Behavior: Cleans NaN, does linear regression, bootstraps for CI (fallback to std err if few samples).
        - Args:
            ref_returns: Reference asset returns array.
            target_returns: Target asset returns array.
            bootstrap_iterations: Number of resamples (default 1000).
            seed: For reproducibility (default 42; None for random).
        - Returns: Dict with beta, CI, r_squared, etc.
        """
        # Filter NaN
        valid_mask = ~(np.isnan(ref_returns) | np.isnan(target_returns))
        ref_clean = ref_returns[valid_mask]
        target_clean = target_returns[valid_mask]

        if len(ref_clean) < 30:  # Too few points, return defaults
            return {
                'beta': 0.0,
                'beta_ci_lower': 0.0,
                'beta_ci_upper': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'n_points': len(ref_clean)
            }

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(ref_clean, target_clean)

        # FIXED: Bootstrap with seed for reproducibility
        betas = []
        successful_iterations = 0

        # Use generator with seed if provided
        rng = np.random.default_rng(seed) if seed is not None else np.random

        for i in range(bootstrap_iterations):
            try:
                # Use appropriate method based on seed
                if seed is not None:
                    indices = rng.choice(len(ref_clean), size=len(ref_clean), replace=True)
                else:
                    indices = np.random.choice(len(ref_clean), size=len(ref_clean), replace=True)

                ref_boot = ref_clean[indices]
                target_boot = target_clean[indices]

                # Only if enough unique data
                if len(np.unique(ref_boot)) > 1 and len(np.unique(target_boot)) > 1:
                    slope_boot, _, _, _, _ = stats.linregress(ref_boot, target_boot)
                    if not np.isnan(slope_boot) and np.isfinite(slope_boot):
                        betas.append(slope_boot)
                        successful_iterations += 1
            except (ValueError, RuntimeWarning, stats.LinAlgError):  # Skip errors
                continue

        # FIXED: Check enough bootstrap samples
        bootstrap_success_rate = successful_iterations / bootstrap_iterations if bootstrap_iterations > 0 else 0

        if len(betas) >= 100:  # Min 100 valid, use percentile
            ci_lower = np.percentile(betas, 2.5)
            ci_upper = np.percentile(betas, 97.5)
            method_used = 'bootstrap'
        else:  # Fallback to standard error
            ci_lower = slope - 1.96 * std_err
            ci_upper = slope + 1.96 * std_err
            method_used = 'standard_error'

        result = {  # Pack results
            'beta': slope,
            'beta_ci_lower': ci_lower,
            'beta_ci_upper': ci_upper,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'n_points': len(ref_clean),
            'bootstrap_success_rate': bootstrap_success_rate,
            'bootstrap_method': method_used,
            'seed_used': seed
        }

        return result

    def calculate_rolling_beta(self,
                               ref_returns: np.ndarray,
                               target_returns: np.ndarray,
                               window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates rolling beta over time windows.
        - Purpose: Tracks how beta changes over time using sliding windows.
        - Usage: Pass returns arrays; needs at least 30 valid points per window.
        - Returns: Arrays of betas and their indices.
        """
        n = len(ref_returns)  # Length
        betas = []  # Hold betas
        valid_indices = []  # Their positions

        for i in range(window, n):  # Slide window
            ref_window = ref_returns[i - window:i]
            target_window = target_returns[i - window:i]

            # Filter NaN
            valid_mask = ~(np.isnan(ref_window) | np.isnan(target_window))

            if np.sum(valid_mask) >= 30:  # Enough points
                ref_clean = ref_window[valid_mask]
                target_clean = target_window[valid_mask]

                try:
                    slope, _, _, _, _ = stats.linregress(ref_clean, target_clean)
                    betas.append(slope)
                    valid_indices.append(i)
                except:  # Skip errors
                    continue

        return np.array(betas), np.array(valid_indices)

    def calculate_conditional_response(self,
                                       ref_returns: np.ndarray,
                                       target_returns: np.ndarray,
                                       threshold: float,
                                       max_lag: int = 5) -> Dict:
        """
        Calculates conditional responses to large events.
        - Purpose: Analyzes how target responds to big moves (up/down) in reference, over lags.
        - Usage: Threshold defines "big" (e.g., 0.02=2%).
        - Returns: Dict with up/down event stats per lag.
        """
        # Detect events
        up_events = ref_returns > threshold  # Big ups
        down_events = ref_returns < -threshold  # Big downs

        results = {  # Hold results
            'up_events': {},
            'down_events': {}
        }

        # Analyze up events
        if np.sum(up_events) > 0:
            results['up_events'] = self._analyze_events(
                up_events, ref_returns, target_returns, max_lag
            )

        # Analyze down events
        if np.sum(down_events) > 0:
            results['down_events'] = self._analyze_events(
                down_events, ref_returns, target_returns, max_lag
            )

        return results

    def _analyze_events(self,
                        event_mask: np.ndarray,
                        ref_returns: np.ndarray,
                        target_returns: np.ndarray,
                        max_lag: int) -> Dict:
        """
        Analyzes responses at lags for specific events.
        - Purpose: For given events, collect responses at each lag and compute stats.
        - Usage: Internal; expects event mask.
        - Returns: Dict of stats per lag.
        """
        event_indices = np.where(event_mask)[0]  # Event positions
        n = len(ref_returns)  # Length

        lag_responses = {f'lag_{i}': [] for i in range(max_lag + 1)}  # Hold responses per lag

        for event_idx in event_indices:  # For each event
            for lag in range(max_lag + 1):  # Check lags
                if event_idx + lag < n:  # Within bounds
                    response = target_returns[event_idx + lag]
                    if not np.isnan(response):  # Valid
                        lag_responses[f'lag_{lag}'].append(response)

        # Compute stats per lag
        lag_stats = {}
        for lag_key, responses in lag_responses.items():
            if len(responses) > 0:  # If data
                responses_arr = np.array(responses)

                # Persistence: % in expected direction
                # For up events in ref, expect positive in target
                expected_direction = responses_arr > 0 if np.mean(ref_returns[event_mask]) > 0 else responses_arr < 0
                persistence = np.mean(expected_direction) * 100

                # Average consecutive
                consecutive_runs = self._calculate_consecutive_runs(responses_arr > 0)

                lag_stats[lag_key] = {  # Stats
                    'mean': np.mean(responses_arr),
                    'median': np.median(responses_arr),
                    'std': np.std(responses_arr),
                    'persistence': persistence,
                    'avg_consecutive': consecutive_runs,
                    'n_events': len(responses)
                }
            else:  # Defaults
                lag_stats[lag_key] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'persistence': 0.0,
                    'avg_consecutive': 0.0,
                    'n_events': 0
                }

        return lag_stats

    def _calculate_consecutive_runs(self, boolean_array: np.ndarray) -> float:
        """
        Calculates average consecutive runs.
        - Purpose: Finds average length of consecutive trues (e.g., positive responses).
        - Returns: Average run length.
        """
        if len(boolean_array) == 0:  # Empty
            return 0.0

        runs = []  # Hold run lengths
        current_run = 0  # Current count

        for val in boolean_array:  # Loop
            if val:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:  # Last run
            runs.append(current_run)

        return np.mean(runs) if runs else 0.0  # Average or 0

    def detect_cascade_order(self,
                             ref_returns: np.ndarray,
                             target_returns_dict: Dict[str, np.ndarray],
                             max_lag: int = 10) -> List[CascadeResult]:
        """
        Detects propagation order (cascade) of events.
        - Purpose: Finds how quickly each target responds to ref by checking correlations at lags.
        - Returns: Sorted list of CascadeResults by response time.
        """
        cascade_results = []  # Hold results

        for asset_name, target_returns in target_returns_dict.items():  # For each target
            # Correlations per lag
            correlations = []

            for lag in range(max_lag + 1):  # Lags
                if lag == 0:  # No shift
                    ref_lagged = ref_returns
                    target_lagged = target_returns
                else:  # Shift target
                    ref_lagged = ref_returns[:-lag]
                    target_lagged = target_returns[lag:]

                # Filter NaN
                valid_mask = ~(np.isnan(ref_lagged) | np.isnan(target_lagged))

                if np.sum(valid_mask) >= 30:  # Enough points
                    ref_clean = ref_lagged[valid_mask]
                    target_clean = target_lagged[valid_mask]

                    try:
                        corr, _ = stats.pearsonr(ref_clean, target_clean)
                        correlations.append((lag, corr))
                    except:
                        correlations.append((lag, 0.0))
                else:
                    correlations.append((lag, 0.0))

            # Find optimal lag (max abs corr)
            if correlations:
                optimal_lag, max_corr = max(correlations, key=lambda x: abs(x[1]))

                # Weighted average response time
                weighted_lags = [lag * abs(corr) for lag, corr in correlations]
                total_weight = sum(abs(corr) for _, corr in correlations)
                response_time_avg = sum(weighted_lags) / total_weight if total_weight > 0 else 0

                cascade_results.append(CascadeResult(  # Add
                    asset_name=asset_name,
                    optimal_lag=optimal_lag,
                    max_correlation=max_corr,
                    response_time_avg=response_time_avg,
                    cascade_order=0  # Assigned later
                ))

        # Sort by response time (faster first)
        cascade_results.sort(key=lambda x: x.response_time_avg)

        # Assign order
        for i, result in enumerate(cascade_results):
            result.cascade_order = i + 1

        return cascade_results

    def calculate_volatility_spillover(self,
                                       ref_returns: np.ndarray,
                                       target_returns: np.ndarray,
                                       window: int = 20) -> Dict:
        """
        Analyzes volatility spillover (rolling std).
        - Purpose: Checks how volatility (std dev) in ref affects target over windows.
        - Returns: Dict with correlation, beta on vols.
        """
        # Rolling volatilities
        ref_vols = []
        target_vols = []
        valid_indices = []

        for i in range(window, len(ref_returns)):  # Slide
            ref_window = ref_returns[i - window:i]
            target_window = target_returns[i - window:i]

            # Filter NaN
            ref_mask = ~np.isnan(ref_window)
            target_mask = ~np.isnan(target_window)

            if np.sum(ref_mask) >= 15 and np.sum(target_mask) >= 15:  # Enough
                ref_vol = np.std(ref_window[ref_mask])
                target_vol = np.std(target_window[target_mask])

                ref_vols.append(ref_vol)
                target_vols.append(target_vol)
                valid_indices.append(i)

        if len(ref_vols) < 30:  # Too few, defaults
            return {
                'vol_correlation': 0.0,
                'vol_beta': 0.0,
                'vol_r_squared': 0.0,
                'n_points': len(ref_vols)
            }

        ref_vols = np.array(ref_vols)
        target_vols = np.array(target_vols)

        # Correlation between vols
        vol_corr, _ = stats.pearsonr(ref_vols, target_vols)

        # Volatility beta
        slope, _, r_value, _, _ = stats.linregress(ref_vols, target_vols)

        return {
            'vol_correlation': vol_corr,
            'vol_beta': slope,
            'vol_r_squared': r_value ** 2,
            'n_points': len(ref_vols)
        }

    def detect_overreaction(self,
                            beta: float,
                            mean_response_up: float,
                            mean_response_down: float,
                            threshold: float) -> float:
        """
        Detects overreaction by comparing actual vs expected response.
        - Purpose: Checks if target overreacts to ref events.
        - Returns: Score (>1 = overreaction).
        """
        # Expected based on beta
        expected_response = beta * threshold

        # Actual average
        actual_response = (abs(mean_response_up) + abs(mean_response_down)) / 2

        if abs(expected_response) < 0.0001:  # Avoid div0
            return 1.0

        overreaction_score = actual_response / abs(expected_response)
        return overreaction_score

    def analyze_complete(self,
                         combined_df: pl.DataFrame,
                         reference_col: str,
                         target_cols: List[str],
                         threshold: float = 0.02,
                         max_lag: int = 5,
                         rolling_window: int = 100,
                         bootstrap_seed: int = 42) -> Dict[str, ImpactResult]:
        """
        Complete analysis for multiple target assets - FIXED.
        - Purpose: Runs full suite: beta, conditional, overreaction, asymmetry.
        - Usage: Pass combined DF with ref and target columns.
        - Args:
            bootstrap_seed: For reproducible bootstrap.
        - Returns: Dict of ImpactResults per asset.
        """
        ref_returns = combined_df[reference_col].to_numpy()  # Ref array
        results = {}  # Hold per asset

        for target_col in target_cols:  # Each target
            asset_name = target_col.replace('_returns', '')  # Name
            target_returns = combined_df[target_col].to_numpy()  # Array

            # FIXED: Use seed in beta
            beta_result = self.calculate_general_beta(
                ref_returns, target_returns, seed=bootstrap_seed
            )

            # Conditional analysis
            conditional = self.calculate_conditional_response(
                ref_returns, target_returns, threshold, max_lag
            )

            # Extract lag_0 metrics
            up_lag0 = conditional['up_events'].get('lag_0', {})
            down_lag0 = conditional['down_events'].get('lag_0', {})

            # Overreaction
            overreaction = self.detect_overreaction(
                beta_result['beta'],
                up_lag0.get('mean', 0.0),
                down_lag0.get('mean', 0.0),
                threshold
            )

            # Asymmetry
            mean_up = abs(up_lag0.get('mean', 0.0))
            mean_down = abs(down_lag0.get('mean', 0.0))
            asymmetry = mean_down / mean_up if mean_up > 0.0001 else 1.0

            results[asset_name] = ImpactResult(  # Pack
                asset_name=asset_name,
                beta=beta_result['beta'],
                beta_ci_lower=beta_result['beta_ci_lower'],
                beta_ci_upper=beta_result['beta_ci_upper'],
                r_squared=beta_result['r_squared'],
                p_value=beta_result['p_value'],
                n_points=beta_result['n_points'],
                mean_response_up=up_lag0.get('mean', 0.0),
                mean_response_down=down_lag0.get('mean', 0.0),
                median_response_up=up_lag0.get('median', 0.0),
                median_response_down=down_lag0.get('median', 0.0),
                persistence_up=up_lag0.get('persistence', 0.0),
                persistence_down=down_lag0.get('persistence', 0.0),
                avg_consecutive_up=up_lag0.get('avg_consecutive', 0.0),
                avg_consecutive_down=down_lag0.get('avg_consecutive', 0.0),
                overreaction_score=overreaction,
                asymmetry_ratio=asymmetry
            )

        return results