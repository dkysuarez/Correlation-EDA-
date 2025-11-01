# ===== impact_analyzer.py (Fixed conditional response calculation) =====

import polars as pl
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from numpy.linalg import LinAlgError

warnings.filterwarnings('ignore')


@dataclass
class ImpactResult:
    asset_name: str
    beta: float
    beta_ci_lower: float
    beta_ci_upper: float
    r_squared: float
    p_value: float
    n_points: int
    mean_response_up: float
    mean_response_down: float
    median_response_up: float
    median_response_down: float
    persistence_up: float
    persistence_down: float
    avg_consecutive_up: float
    avg_consecutive_down: float
    overreaction_score: float
    asymmetry_ratio: float


@dataclass
class CascadeResult:
    asset_name: str
    optimal_lag: int
    max_correlation: float
    response_time_avg: float
    cascade_order: int


class BTCImpactAnalyzer:

    def __init__(self, reference_asset: str = "BTCUSDT"):
        self.reference_asset = reference_asset

    def resample_to_timeframe(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        timeframe_map = {
            '5m': '5m',
            '10m': '10m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h'
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Timeframe {timeframe} not supported")

        df = df.with_columns(
            pl.col("open_time").cast(pl.Datetime(time_unit="us"))
        )

        return_cols = [col for col in df.columns if col.endswith('_returns')]

        resampled = df.group_by_dynamic(
            "open_time",
            every=timeframe_map[timeframe]
        ).agg([
            pl.col(col).sum() for col in return_cols
        ])

        return resampled.sort("open_time")

    def calculate_general_beta(self,
                               ref_returns: np.ndarray,
                               target_returns: np.ndarray,
                               bootstrap_iterations: int = 1000,
                               seed: Optional[int] = 42) -> Dict:
        # Filter NaN
        valid_mask = ~(np.isnan(ref_returns) | np.isnan(target_returns))
        ref_clean = ref_returns[valid_mask]
        target_clean = target_returns[valid_mask]

        if len(ref_clean) < 30:
            return {
                'beta': 0.0,
                'beta_ci_lower': 0.0,
                'beta_ci_upper': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'n_points': len(ref_clean)
            }

        # Linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(ref_clean, target_clean)
        except (ValueError, LinAlgError):
            return {
                'beta': 0.0,
                'beta_ci_lower': 0.0,
                'beta_ci_upper': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'n_points': len(ref_clean)
            }

        # Bootstrap with limits for large datasets
        betas = []
        successful_iterations = 0

        rng = np.random.default_rng(seed) if seed is not None else np.random

        for i in range(min(bootstrap_iterations, 500)):
            try:
                if seed is not None:
                    indices = rng.choice(len(ref_clean), size=min(len(ref_clean), 5000), replace=True)
                else:
                    indices = np.random.choice(len(ref_clean), size=min(len(ref_clean), 5000), replace=True)

                ref_boot = ref_clean[indices]
                target_boot = target_clean[indices]

                if len(np.unique(ref_boot)) > 1 and len(np.unique(target_boot)) > 1:
                    slope_boot, _, _, _, _ = stats.linregress(ref_boot, target_boot)
                    if not np.isnan(slope_boot) and np.isfinite(slope_boot):
                        betas.append(slope_boot)
                        successful_iterations += 1
            except (ValueError, RuntimeWarning, LinAlgError, MemoryError):
                continue

        bootstrap_success_rate = successful_iterations / bootstrap_iterations if bootstrap_iterations > 0 else 0

        if len(betas) >= 50:
            ci_lower = np.percentile(betas, 2.5)
            ci_upper = np.percentile(betas, 97.5)
            method_used = 'bootstrap'
        else:
            ci_lower = slope - 1.96 * std_err
            ci_upper = slope + 1.96 * std_err
            method_used = 'standard_error'

        result = {
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
        n = len(ref_returns)
        betas = []
        valid_indices = []

        for i in range(window, n):
            ref_window = ref_returns[i - window:i]
            target_window = target_returns[i - window:i]

            valid_mask = ~(np.isnan(ref_window) | np.isnan(target_window))

            if np.sum(valid_mask) >= 30:
                ref_clean = ref_window[valid_mask]
                target_clean = target_window[valid_mask]

                try:
                    slope, _, _, _, _ = stats.linregress(ref_clean, target_clean)
                    betas.append(slope)
                    valid_indices.append(i)
                except (ValueError, LinAlgError, MemoryError):
                    continue

        return np.array(betas), np.array(valid_indices)

    def calculate_conditional_response(self,
                                       ref_returns: np.ndarray,
                                       target_returns: np.ndarray,
                                       threshold: float,
                                       max_lag: int = 5) -> Dict:
        """
        FIXED: Improved event detection and response calculation
        """
        # Convert threshold to absolute value for down events
        threshold_abs = abs(threshold)

        # Detect events - FIXED: Use absolute threshold for both directions
        up_events = ref_returns > threshold_abs
        down_events = ref_returns < -threshold_abs

        # Debug info
        n_up_events = np.sum(up_events)
        n_down_events = np.sum(down_events)

        print(f"DEBUG: Up events: {n_up_events}, Down events: {n_down_events}, Threshold: {threshold_abs}")

        results = {
            'up_events': {},
            'down_events': {}
        }

        # Analyze up events
        if n_up_events > 0:
            up_results = self._analyze_events(
                up_events, ref_returns, target_returns, max_lag, event_type='up'
            )
            results['up_events'] = up_results
            print(f"DEBUG: Up events analysis - {len(up_results)} lags")

        # Analyze down events - FIXED: Ensure we analyze properly
        if n_down_events > 0:
            down_results = self._analyze_events(
                down_events, ref_returns, target_returns, max_lag, event_type='down'
            )
            results['down_events'] = down_results
            print(f"DEBUG: Down events analysis - {len(down_results)} lags")

        return results

    def _analyze_events(self,
                        event_mask: np.ndarray,
                        ref_returns: np.ndarray,
                        target_returns: np.ndarray,
                        max_lag: int,
                        event_type: str = 'up') -> Dict:
        """
        FIXED: Improved event analysis with better statistics
        """
        event_indices = np.where(event_mask)[0]
        n = len(ref_returns)

        lag_responses = {f'lag_{i}': [] for i in range(max_lag + 1)}

        # Collect responses for each lag
        for event_idx in event_indices:
            for lag in range(max_lag + 1):
                response_idx = event_idx + lag
                if response_idx < n:  # Ensure we don't go out of bounds
                    response = target_returns[response_idx]
                    if not np.isnan(response):
                        lag_responses[f'lag_{lag}'].append(response)

        # Compute statistics for each lag
        lag_stats = {}
        for lag_key, responses in lag_responses.items():
            if len(responses) > 0:
                responses_arr = np.array(responses)

                # FIXED: Improved persistence calculation
                if event_type == 'up':
                    # For up events, we expect positive responses (same direction)
                    expected_direction = responses_arr > 0
                else:  # down events
                    # For down events, we expect negative responses (same direction)
                    expected_direction = responses_arr < 0

                persistence = np.mean(expected_direction) * 100 if len(expected_direction) > 0 else 0

                # Average consecutive runs in expected direction
                consecutive_runs = self._calculate_consecutive_runs(expected_direction)

                lag_stats[lag_key] = {
                    'mean': float(np.mean(responses_arr)),
                    'median': float(np.median(responses_arr)),
                    'std': float(np.std(responses_arr)),
                    'persistence': float(persistence),
                    'avg_consecutive': float(consecutive_runs),
                    'n_events': len(responses)
                }

                # Debug info for lag_0
                if lag_key == 'lag_0':
                    print(f"DEBUG {event_type} {lag_key}: mean={np.mean(responses_arr):.6f}, "
                          f"persistence={persistence:.1f}%, n_events={len(responses)}")
            else:
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
        if len(boolean_array) == 0:
            return 0.0

        runs = []
        current_run = 0

        for val in boolean_array:
            if val:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        return np.mean(runs) if runs else 0.0

    def detect_cascade_order(self,
                             ref_returns: np.ndarray,
                             target_returns_dict: Dict[str, np.ndarray],
                             max_lag: int = 10) -> List[CascadeResult]:
        cascade_results = []

        for asset_name, target_returns in target_returns_dict.items():
            correlations = []

            for lag in range(max_lag + 1):
                if lag == 0:
                    ref_lagged = ref_returns
                    target_lagged = target_returns
                else:
                    ref_lagged = ref_returns[:-lag]
                    target_lagged = target_returns[lag:]

                valid_mask = ~(np.isnan(ref_lagged) | np.isnan(target_lagged))

                if np.sum(valid_mask) >= 30:
                    ref_clean = ref_lagged[valid_mask]
                    target_clean = target_lagged[valid_mask]

                    try:
                        corr, _ = stats.pearsonr(ref_clean, target_clean)
                        correlations.append((lag, corr))
                    except (ValueError, LinAlgError, MemoryError):
                        correlations.append((lag, 0.0))
                else:
                    correlations.append((lag, 0.0))

            if correlations:
                optimal_lag, max_corr = max(correlations, key=lambda x: abs(x[1]))

                weighted_lags = [lag * abs(corr) for lag, corr in correlations]
                total_weight = sum(abs(corr) for _, corr in correlations)
                response_time_avg = sum(weighted_lags) / total_weight if total_weight > 0 else 0

                cascade_results.append(CascadeResult(
                    asset_name=asset_name,
                    optimal_lag=optimal_lag,
                    max_correlation=max_corr,
                    response_time_avg=response_time_avg,
                    cascade_order=0
                ))

        cascade_results.sort(key=lambda x: x.response_time_avg)

        for i, result in enumerate(cascade_results):
            result.cascade_order = i + 1

        return cascade_results

    def calculate_volatility_spillover(self,
                                       ref_returns: np.ndarray,
                                       target_returns: np.ndarray,
                                       window: int = 20) -> Dict:
        ref_vols = []
        target_vols = []
        valid_indices = []

        for i in range(window, len(ref_returns)):
            ref_window = ref_returns[i - window:i]
            target_window = target_returns[i - window:i]

            ref_mask = ~np.isnan(ref_window)
            target_mask = ~np.isnan(target_window)

            if np.sum(ref_mask) >= 15 and np.sum(target_mask) >= 15:
                ref_vol = np.std(ref_window[ref_mask])
                target_vol = np.std(target_window[target_mask])

                ref_vols.append(ref_vol)
                target_vols.append(target_vol)
                valid_indices.append(i)

        if len(ref_vols) < 30:
            return {
                'vol_correlation': 0.0,
                'vol_beta': 0.0,
                'vol_r_squared': 0.0,
                'n_points': len(ref_vols)
            }

        ref_vols = np.array(ref_vols)
        target_vols = np.array(target_vols)

        try:
            vol_corr, _ = stats.pearsonr(ref_vols, target_vols)
            slope, _, r_value, _, _ = stats.linregress(ref_vols, target_vols)
        except (ValueError, LinAlgError, MemoryError):
            return {
                'vol_correlation': 0.0,
                'vol_beta': 0.0,
                'vol_r_squared': 0.0,
                'n_points': len(ref_vols)
            }

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
        expected_response = beta * threshold
        actual_response = (abs(mean_response_up) + abs(mean_response_down)) / 2

        if abs(expected_response) < 0.0001:
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
        FIXED: Improved complete analysis with better conditional response handling
        """
        ref_returns = combined_df[reference_col].to_numpy()
        results = {}

        for target_col in target_cols:
            asset_name = target_col.replace('_returns', '')
            target_returns = combined_df[target_col].to_numpy()

            # Use all data for conditional analysis (more events)
            ref_sample = ref_returns
            target_sample = target_returns

            # Calculate beta
            beta_result = self.calculate_general_beta(
                ref_sample, target_sample, seed=bootstrap_seed
            )

            # Calculate conditional responses - FIXED: Use proper threshold
            conditional = self.calculate_conditional_response(
                ref_sample, target_sample, threshold, max_lag
            )

            # Extract lag_0 metrics with proper fallbacks
            up_lag0 = conditional['up_events'].get('lag_0', {})
            down_lag0 = conditional['down_events'].get('lag_0', {})

            # Debug info
            print(f"DEBUG {asset_name}:")
            print(
                f"  Up events - mean: {up_lag0.get('mean', 0):.6f}, persistence: {up_lag0.get('persistence', 0):.1f}%")
            print(
                f"  Down events - mean: {down_lag0.get('mean', 0):.6f}, persistence: {down_lag0.get('persistence', 0):.1f}%")

            # Overreaction score
            overreaction = self.detect_overreaction(
                beta_result['beta'],
                up_lag0.get('mean', 0.0),
                down_lag0.get('mean', 0.0),
                threshold
            )

            # Asymmetry ratio - FIXED: Handle division by zero properly
            mean_up = abs(up_lag0.get('mean', 0.0))
            mean_down = abs(down_lag0.get('mean', 0.0))

            if mean_up > 0.0001 and mean_down > 0.0001:
                asymmetry = mean_down / mean_up
            else:
                asymmetry = 1.0  # Default to symmetric if insufficient data

            results[asset_name] = ImpactResult(
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