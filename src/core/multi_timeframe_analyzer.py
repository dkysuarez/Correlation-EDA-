import polars as pl  # Library for handling large datasets efficiently
import numpy as np  # Library for numerical computations and arrays
from typing import Dict, List, Tuple, Optional  # Type hints for better code clarity
from pathlib import Path  # For handling file paths
from scipy.stats import spearmanr  # For Spearman correlation calculation
import warnings  # To control warning messages
from datetime import datetime, timedelta  # For datetime operations

warnings.filterwarnings('ignore')  # Suppress all warnings for cleaner output

class MultiTimeframeAnalyzer:
    """Analyzes correlations across multiple timeframes"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)  # Directory for data files
        self.timeframes = ["15m", "30m", "1h", "4h", "1d"]  # Supported timeframes

    def load_asset_data(self, asset: str, timeframe: str) -> Optional[pl.DataFrame]:
        """Loads asset data for a specific timeframe"""
        try:
            file_path = self.data_dir / asset / f"{asset}.parquet"  # Construct file path

            if not file_path.exists():  # Check if file exists
                return None

            df = pl.read_parquet(file_path)  # Load parquet file

            if "1min_returns" not in df.columns or "open_time" not in df.columns:  # Validate required columns
                return None

            # Normalize timestamp to datetime
            df = df.with_columns(
                pl.col("open_time").cast(pl.Datetime(time_unit="us")).alias("open_time")
            )

            df = df.select(["open_time", "1min_returns"])  # Select relevant columns
            df = df.drop_nulls()  # Remove rows with nulls

            if df.is_empty():  # Check if DataFrame is empty
                return None

            # Aggregate to target timeframe
            df_agg = self._aggregate_timeframe(df, timeframe)
            df_agg = df_agg.rename({"1min_returns": "returns"})

            return df_agg

        except Exception as e:
            print(f"❌ Error loading {asset} in {timeframe}: {str(e)}")  # Log error
            return None

    def _aggregate_timeframe(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """Aggregates data to the specified timeframe"""
        time_map = {  # Mapping of valid timeframes
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

        if timeframe not in time_map:  # Default to 1h if timeframe invalid
            timeframe = "1h"

        return df.group_by_dynamic("open_time", every=time_map[timeframe]).agg(
            pl.col("1min_returns").sum()  # Sum returns for aggregation
        )

    def calculate_timeframe_correlations(self, asset1: str, asset2: str,
                                         min_periods: int = 30) -> Dict[str, Dict]:
        """Calculates correlations for all timeframes"""
        results = {}

        for tf in self.timeframes:  # Iterate over timeframes
            try:
                # Load data for both assets
                df1 = self.load_asset_data(asset1, tf)
                df2 = self.load_asset_data(asset2, tf)

                if df1 is None or df2 is None or df1.is_empty() or df2.is_empty():  # Check for valid data
                    results[tf] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'periods': 0,
                        'status': 'NO_DATA'
                    }
                    continue

                # Join data on open_time
                combined = df1.join(df2, on="open_time", how="inner", suffix=f"_{asset2}")
                combined = combined.rename({"returns": f"returns_{asset1}"})

                if combined.height < min_periods:  # Check minimum periods
                    results[tf] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'periods': combined.height,
                        'status': 'INSUFFICIENT_DATA'
                    }
                    continue

                # Extract returns
                returns1 = combined[f"returns_{asset1}"].to_numpy()
                returns2 = combined[f"returns_{asset2}"].to_numpy()

                # Filter infinite values
                mask = np.isfinite(returns1) & np.isfinite(returns2)
                returns1_clean = returns1[mask]
                returns2_clean = returns2[mask]

                if len(returns1_clean) < min_periods:  # Check cleaned data size
                    results[tf] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'periods': len(returns1_clean),
                        'status': 'INSUFFICIENT_CLEAN_DATA'
                    }
                    continue

                # Calculate Spearman correlation
                corr, p_value = spearmanr(returns1_clean, returns2_clean, nan_policy='omit')

                if np.isnan(corr):  # Handle NaN correlation
                    corr, p_value = 0.0, 1.0

                # Classify intensity and significance
                intensity = self._classify_correlation_intensity(corr)
                significance = self._classify_significance(p_value)

                results[tf] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'periods': len(returns1_clean),
                    'status': 'SUCCESS',
                    'intensity': intensity,
                    'significance': significance,
                    'start_date': combined["open_time"].min(),
                    'end_date': combined["open_time"].max()
                }

            except Exception as e:
                results[tf] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'periods': 0,
                    'status': f'ERROR: {str(e)[:50]}'
                }

        return results

    def _classify_correlation_intensity(self, corr: float) -> str:
        """Classifies correlation strength"""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "VERY_STRONG"
        elif abs_corr > 0.5:
            return "STRONG"
        elif abs_corr > 0.3:
            return "MODERATE"
        elif abs_corr > 0.1:
            return "WEAK"
        else:
            return "VERY_WEAK"

    def _classify_significance(self, p_value: float) -> str:
        """Classifies statistical significance"""
        if p_value < 0.01:
            return "VERY_SIGNIFICANT"
        elif p_value < 0.05:
            return "SIGNIFICANT"
        elif p_value < 0.1:
            return "MARGINAL"
        else:
            return "NOT_SIGNIFICANT"

    def analyze_correlation_consistency(self, results: Dict[str, Dict]) -> Dict:
        """Analyzes correlation consistency across timeframes"""
        successful_tfs = [tf for tf in self.timeframes
                          if results.get(tf, {}).get('status') == 'SUCCESS']  # Filter successful timeframes

        if len(successful_tfs) < 2:  # Need at least 2 for consistency
            return {
                'consistency_score': 0.0,
                'trend': 'NO_DATA',
                'recommendation': 'Insufficient data',
                'max_correlation_tf': None,
                'min_correlation_tf': None,
                'average_correlation': 0.0,
                'correlation_std': 0.0,
                'successful_timeframes': len(successful_tfs)
            }

        correlations = [results[tf]['correlation'] for tf in successful_tfs]  # Collect correlations
        avg_correlation = np.mean(correlations)  # Average correlation
        std_correlation = np.std(correlations)  # Standard deviation

        # Calculate consistency score (1 - coefficient of variation)
        if abs(avg_correlation) > 0.05:
            cv = std_correlation / abs(avg_correlation)
            consistency_score = max(0, 1 - cv)
        else:
            consistency_score = max(0, 1 - std_correlation / 0.05)

        # Determine trend
        trend = self._calculate_trend(results, successful_tfs)

        # Generate recommendation
        recommendation = self._generate_recommendation_direct(results, successful_tfs, trend, avg_correlation,
                                                              consistency_score)

        # Find max and min correlation timeframes
        max_tf = None
        min_tf = None
        if successful_tfs:
            max_tf = max(successful_tfs, key=lambda x: abs(results[x]['correlation']))
            min_tf = min(successful_tfs, key=lambda x: abs(results[x]['correlation']))

        return {
            'consistency_score': float(consistency_score),
            'average_correlation': float(avg_correlation),
            'correlation_std': float(std_correlation),
            'trend': trend,
            'recommendation': recommendation,
            'max_correlation_tf': max_tf,
            'min_correlation_tf': min_tf,
            'successful_timeframes': len(successful_tfs)
        }

    def _calculate_trend(self, results: Dict, successful_tfs: List[str]) -> str:
        """Calculates trend without recursion"""
        tf_order = ["15m", "30m", "1h", "4h", "1d"]  # Timeframe order
        valid_tfs = [tf for tf in tf_order if tf in successful_tfs]  # Filter valid
        valid_corrs = [results[tf]['correlation'] for tf in valid_tfs]  # Get correlations

        if len(valid_corrs) >= 3:  # Need at least 3 for trend
            try:
                slope = np.polyfit(range(len(valid_corrs)), valid_corrs, 1)[0]  # Linear regression slope
                if slope > 0.01:
                    return "INCREASING"
                elif slope < -0.01:
                    return "DECREASING"
                else:
                    return "STABLE"
            except:
                return "INDETERMINATE"
        else:
            return "INDETERMINATE"

    def _generate_recommendation_direct(self, results: Dict, successful_tfs: List[str],
                                        trend: str, avg_correlation: float, consistency_score: float) -> str:
        """Generates recommendation without recursion"""
        if len(successful_tfs) < 3:  # Need at least 3 timeframes
            return "More timeframes needed for analysis"

        # Get correlations in timeframe order
        tf_order = ["15m", "30m", "1h", "4h", "1d"]
        ordered_tfs = [tf for tf in tf_order if tf in successful_tfs]

        # Pattern 1: Decreasing correlation in short timeframes
        short_tfs = [tf for tf in ordered_tfs if tf in ["15m", "30m", "1h"]]
        if len(short_tfs) >= 2:
            short_corrs = [results[tf]['correlation'] for tf in short_tfs]
            if len(short_corrs) >= 2:
                try:
                    short_slope = np.polyfit(range(len(short_corrs)), short_corrs, 1)[0]
                    if short_slope < -0.1 and any(abs(c) > 0.3 for c in short_corrs):
                        return "OPPORTUNITY: Assets decouple in short timeframes - Scalping strategies possible"
                except:
                    pass

        # Pattern 2: High correlation in long timeframes, low in short
        long_tfs = [tf for tf in ordered_tfs if tf in ["4h", "1d"]]
        short_tfs = [tf for tf in ordered_tfs if tf in ["15m", "30m", "1h"]]

        if long_tfs and short_tfs:
            avg_long = np.mean([results[tf]['correlation'] for tf in long_tfs])
            avg_short = np.mean([results[tf]['correlation'] for tf in short_tfs])

            if avg_long > 0.6 and avg_short < 0.3:
                return "OPPORTUNITY: Structural correlation but intraday disconnection - Mean reversion possible"

        # Pattern 3: High consistency
        if consistency_score > 0.8:
            return "HIGH CONSISTENCY: Assets maintain stable relationship across timeframes"

        # Pattern 4: Low overall correlation
        if abs(avg_correlation) < 0.2:
            return "LOW CORRELATION: Assets move independently - Effective diversification"

        # Pattern 5: Increasing/decreasing trend
        if trend == "INCREASING":
            return "INCREASING TREND: Assets are progressively coupling"
        elif trend == "DECREASING":
            return "DECREASING TREND: Assets are progressively decoupling"

        return "Standard analysis - Consider specific timeframe correlations"