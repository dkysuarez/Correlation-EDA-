import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.data_loader import BatchCorrelationLoader
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class LagStatistics:
    """Detailed statistics for a specific lag period"""
    lag: int
    mean: float
    median: float
    std: float
    n_events: int
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float
    significance: str
    persistence: float


@dataclass
class EventResponse:
    """Complete response to an event with all lag periods"""
    event_time: datetime
    reference_return: float
    lag_statistics: List[LagStatistics]
    crossed_days: bool
    end_time: datetime


class MultiTimeframeLeadLagAnalyzer:
    """
    FINAL VERSION: Detailed lag-by-lag analysis with statistical validation

    Features:
    - Individual lag-by-lag analysis (not just averages)
    - Statistical validation with CI and p-values
    - Correct temporal logic (post-event)
    - Inspired by impact_analyzer for robust statistics
    """

    def __init__(self, reference_asset: str = "BTCUSDT", data_dir: str = "data", cache_dir: str = "cache/processed"):
        self.reference_asset = reference_asset
        self.loader = BatchCorrelationLoader(data_dir=data_dir, cache_dir=cache_dir)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        time_map = {
            "1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }
        return time_map.get(timeframe, 60)

    def _normalize_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize open_time column to consistent datetime format"""
        return df.with_columns([
            pl.col("open_time").cast(pl.Datetime(time_unit="us"))
        ])

    def _load_and_combine_assets(self, assets: List[str], timeframe: str) -> pl.DataFrame:
        """Load and combine assets with unique suffixes"""
        print(f"📊 Loading {len(assets)} assets in timeframe {timeframe}")

        asset_dataframes = []

        for asset in assets:
            try:
                single_data = self.loader.load_assets_in_batches([asset], batch_size=1, timeframe=timeframe)

                if asset in single_data['asset_data']:
                    df = single_data['asset_data'][asset]
                    df = self._normalize_datetime(df)
                    asset_dataframes.append(df)
                    print(f"  ✅ {asset}: {df.height} periods")
                else:
                    print(f"  ❌ {asset}: Could not load")

            except Exception as e:
                print(f"  ❌ {asset}: Error - {str(e)}")
                continue

        if not asset_dataframes:
            raise ValueError("No assets could be loaded")

        # Combine with unique suffixes
        combined_df = asset_dataframes[0]

        for i, df in enumerate(asset_dataframes[1:], 1):
            combined_df = combined_df.join(
                df,
                on="open_time",
                how="outer",
                suffix=f"_join_{i}"
            )

        combined_df = combined_df.sort("open_time")
        return_cols = [c for c in combined_df.columns if '_returns' in c]
        print(f"📈 Combined DataFrame: {combined_df.height} rows, {len(return_cols)} assets")

        return combined_df

    def _find_reference_events(self, ref_df: pl.DataFrame, threshold: float, comparison: str) -> Tuple[List, List]:
        """Find significant events in the reference asset"""
        ref_col = f"{self.reference_asset}_returns"

        if ref_col not in ref_df.columns:
            raise ValueError(f"Column {ref_col} not found in reference data")

        if comparison == "greater_equal":
            positive_mask = pl.col(ref_col) >= threshold
            negative_mask = pl.col(ref_col) <= -threshold
        elif comparison == "greater":
            positive_mask = pl.col(ref_col) > threshold
            negative_mask = pl.col(ref_col) < -threshold
        elif comparison == "equal":
            positive_mask = pl.col(ref_col) == threshold
            negative_mask = pl.col(ref_col) == -threshold
        else:
            raise ValueError(f"Comparison {comparison} not supported")

        positive_events = ref_df.filter(positive_mask)
        negative_events = ref_df.filter(negative_mask)

        positive_times = positive_events.select(["open_time", ref_col]).to_dicts()
        negative_times = negative_events.select(["open_time", ref_col]).to_dicts()

        print(f"🔍 Events found: {len(positive_times)} positive, {len(negative_times)} negative")

        return positive_times, negative_times

    def _calculate_lag_statistics(self, returns: np.ndarray, bootstrap_iterations: int = 1000,
                                  seed: int = 42) -> Dict:
        """
        Calculate detailed statistics for a specific lag period
        Inspired by impact_analyzer for statistical robustness
        """
        if len(returns) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'n_events': 0,
                'confidence_interval_lower': 0.0,
                'confidence_interval_upper': 0.0,
                'p_value': 1.0,
                'significance': 'INSUFFICIENT_DATA',
                'persistence': 0.0
            }

        returns_clean = returns[~np.isnan(returns)]

        if len(returns_clean) < 3:
            return {
                'mean': np.mean(returns_clean) if len(returns_clean) > 0 else 0.0,
                'median': np.median(returns_clean) if len(returns_clean) > 0 else 0.0,
                'std': 0.0,
                'n_events': len(returns_clean),
                'confidence_interval_lower': 0.0,
                'confidence_interval_upper': 0.0,
                'p_value': 1.0,
                'significance': 'INSUFFICIENT_DATA',
                'persistence': 0.0
            }

        # Basic statistics
        mean_val = np.mean(returns_clean)
        median_val = np.median(returns_clean)
        std_val = np.std(returns_clean)
        n_events = len(returns_clean)

        # T-test against zero (is there a significant effect?)
        try:
            t_stat, p_value = stats.ttest_1samp(returns_clean, 0.0)
        except:
            p_value = 1.0

        # Significance levels
        if p_value < 0.001:
            significance = "VERY_SIGNIFICANT"
        elif p_value < 0.01:
            significance = "SIGNIFICANT"
        elif p_value < 0.05:
            significance = "MARGINALLY_SIGNIFICANT"
        else:
            significance = "NOT_SIGNIFICANT"

        # Bootstrap for confidence interval
        if n_events >= 10 and bootstrap_iterations > 0:
            rng = np.random.default_rng(seed)
            bootstrap_means = []

            for _ in range(bootstrap_iterations):
                bootstrap_sample = rng.choice(returns_clean, size=len(returns_clean), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
        else:
            # Fallback to standard error
            se = std_val / np.sqrt(n_events)
            ci_lower = mean_val - 1.96 * se
            ci_upper = mean_val + 1.96 * se

        # Persistence (% of responses in positive direction)
        # NOTE: For negative events, low persistence indicates consistency
        positive_responses = np.sum(returns_clean > 0)
        persistence = (positive_responses / n_events) * 100.0

        return {
            'mean': float(mean_val),
            'median': float(median_val),
            'std': float(std_val),
            'n_events': n_events,
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'p_value': float(p_value),
            'significance': significance,
            'persistence': float(persistence)
        }

    def _extract_lag_responses(self, target_df: pl.DataFrame, event_time: datetime,
                               asset: str, max_lags: int, target_tf: str,
                               same_day_only: bool = False) -> Dict:
        """
        KEY IMPROVEMENT: Extract lag-by-lag responses with PROPER temporal alignment
        FIX: Calculate actual lag based on time difference, not array index
        """
        # Find the correct column for the asset
        asset_col = None
        possible_cols = [
            f"{asset}_returns",
            f"{asset}_returns_join_1",
            f"{asset}_returns_join_2",
            f"{asset}_returns_join_3"
        ]

        for col in possible_cols:
            if col in target_df.columns:
                asset_col = col
                break

        if asset_col is None:
            return None

        # Calculate time boundaries
        target_minutes = self._timeframe_to_minutes(target_tf)
        max_time_delta = timedelta(minutes=target_minutes * max_lags)
        end_time = event_time + max_time_delta

        if same_day_only:
            event_date = event_time.date()
            end_of_day = datetime.combine(event_date + timedelta(days=1), datetime.min.time())
            end_time = min(end_time, end_of_day)

        # CRITICAL: Get POST-EVENT candles sorted by time
        post_event_data = target_df.filter(
            (pl.col("open_time") > event_time) &
            (pl.col("open_time") <= end_time)
        ).select(["open_time", asset_col]).drop_nulls().sort("open_time")

        if post_event_data.height == 0:
            return None

        # MAJOR IMPROVEMENT: Calculate lag based on ACTUAL time difference
        # This fixes the issue where data gaps would cause incorrect lag assignments
        returns_by_lag = {}

        # Get all post-event data as dictionaries for processing
        post_event_rows = post_event_data.to_dicts()

        for row in post_event_rows:
            # Calculate exact time difference from event
            time_diff = row['open_time'] - event_time
            time_diff_minutes = time_diff.total_seconds() / 60.0

            # Calculate the actual lag number based on timeframe
            actual_lag = int(time_diff_minutes / target_minutes)

            # Only include lags within our maximum range
            if 0 <= actual_lag <= max_lags:
                returns_by_lag[actual_lag] = row[asset_col]

        # Fill missing lags with NaN to maintain structure
        for lag in range(max_lags + 1):
            if lag not in returns_by_lag:
                returns_by_lag[lag] = np.nan

        return {
            'returns_by_lag': returns_by_lag,
            'crossed_days': (end_time - event_time).days > 0,
            'end_time': end_time,
            'total_available_lags': len(post_event_rows)
        }

    def analyze(self,
                target_assets: List[str],
                reference_timeframe: str,
                target_timeframes: List[str],
                threshold: float = 0.02,
                max_lags: int = 24,
                comparison: str = "greater_equal",
                same_day_only: bool = False,
                bootstrap_iterations: int = 1000,
                seed: int = 42) -> Dict:
        """
        MAIN METHOD: Complete analysis with lag-by-lag statistics
        IMPROVEMENT: Load data once and reuse across timeframes for efficiency
        """
        print(f"🚀 STARTING MULTI-TIMEFRAME ANALYSIS WITH DETAILED LAGS")
        print(f"📈 Reference: {self.reference_asset} ({reference_timeframe})")
        print(f"🎯 Targets: {target_assets}")
        print(f"⏰ Target timeframes: {target_timeframes}")
        print(f"📊 Threshold: {threshold * 100:.1f}%, Max lags: {max_lags}")

        # Validations
        valid_timeframes = ['1m', '5m', '10m', '15m', '30m', '1h', '4h', '1d']
        if reference_timeframe not in valid_timeframes:
            raise ValueError(f"Reference timeframe {reference_timeframe} not supported")

        for tf in target_timeframes:
            if tf not in valid_timeframes:
                raise ValueError(f"Timeframe {tf} not supported")

            ref_minutes = self._timeframe_to_minutes(reference_timeframe)
            target_minutes = self._timeframe_to_minutes(tf)
            if target_minutes >= ref_minutes:
                raise ValueError(f"Timeframe {tf} must be smaller than {reference_timeframe}")

        # MAJOR IMPROVEMENT: Load ALL data ONCE and reuse
        # This eliminates redundant data loading and improves performance
        print("\n📊 STEP 1: Loading ALL data efficiently (single load)...")

        # Combine all needed assets including reference
        all_assets_needed = list(set([self.reference_asset] + target_assets))
        print(f"   Loading {len(all_assets_needed)} assets: {all_assets_needed}")

        # Load reference data (this will be reused for event detection)
        ref_df = self._load_and_combine_assets([self.reference_asset], reference_timeframe)

        # Pre-load target data for ALL timeframes to avoid repeated loading
        preloaded_target_data = {}
        for target_tf in target_timeframes:
            print(f"   Pre-loading data for timeframe: {target_tf}")
            target_df = self._load_and_combine_assets(all_assets_needed, target_tf)
            preloaded_target_data[target_tf] = target_df

        # 2. Find events in reference asset
        print("\n🔍 STEP 2: Identifying significant events...")
        positive_events, negative_events = self._find_reference_events(ref_df, threshold, comparison)

        if len(positive_events) == 0 and len(negative_events) == 0:
            return {
                "error": f"No events found with threshold {threshold * 100:.1f}%",
                "results": {"positive": {}, "negative": {}},
                "positive_event_count": 0,
                "negative_event_count": 0,
                "failed_assets": []
            }

        # 3. Improved results structure
        results = {
            "positive": {tf: {asset: {"events": [], "lag_statistics": {}}
                              for asset in target_assets} for tf in target_timeframes},
            "negative": {tf: {asset: {"events": [], "lag_statistics": {}}
                              for asset in target_assets} for tf in target_timeframes}
        }

        failed_assets = []

        # IMPROVEMENT: Use pre-loaded data instead of loading repeatedly
        for target_tf in target_timeframes:
            print(f"\n⏰ STEP 3: Analyzing timeframe {target_tf}...")

            try:
                # Use pre-loaded data instead of loading again
                target_df = preloaded_target_data[target_tf]
                print(f"   Using pre-loaded data with {target_df.height} rows")

                # Process each event type
                for event_type, events in [("positive", positive_events), ("negative", negative_events)]:
                    print(f"📊 Processing {len(events)} {event_type} events...")

                    for asset in target_assets:
                        asset_responses = []

                        for event in events:
                            event_time = event["open_time"]
                            event_return = event[f"{self.reference_asset}_returns"]

                            try:
                                # Use improved lag extraction with proper temporal alignment
                                response = self._extract_lag_responses(
                                    target_df, event_time, asset, max_lags, target_tf, same_day_only
                                )

                                if response is not None:
                                    response["event_time"] = event_time
                                    response["reference_return"] = float(event_return)
                                    asset_responses.append(response)

                            except Exception as e:
                                print(f"❌ Error processing {asset} {event_type} event: {str(e)}")
                                continue

                        # Calculate aggregated statistics by lag
                        lag_statistics = {}

                        for lag in range(max_lags + 1):
                            # Collect all returns for this specific lag
                            lag_returns = []
                            for response in asset_responses:
                                if lag in response['returns_by_lag']:
                                    lag_value = response['returns_by_lag'][lag]
                                    if not np.isnan(lag_value):
                                        lag_returns.append(lag_value)

                            # Calculate statistics for this lag
                            lag_stats = self._calculate_lag_statistics(
                                np.array(lag_returns), bootstrap_iterations, seed
                            )
                            lag_stats['lag'] = lag
                            lag_statistics[f'lag_{lag}'] = lag_stats

                        # Save results
                        results[event_type][target_tf][asset] = {
                            "events": asset_responses,
                            "lag_statistics": lag_statistics,
                            "total_events": len(asset_responses)
                        }

            except Exception as e:
                print(f"❌ Error processing timeframe {target_tf}: {str(e)}")
                for asset in target_assets:
                    if asset not in failed_assets:
                        failed_assets.append(asset)

        # Final summary
        total_lag_analyses = 0
        for event_type in ["positive", "negative"]:
            for tf in target_timeframes:
                for asset in target_assets:
                    if results[event_type][tf][asset]["lag_statistics"]:
                        total_lag_analyses += len(results[event_type][tf][asset]["lag_statistics"])

        print(f"\n✅ ANALYSIS COMPLETED")
        print(f"📊 Positive events: {len(positive_events)}")
        print(f"📊 Negative events: {len(negative_events)}")
        print(f"📊 Total lag analyses: {total_lag_analyses}")
        print(f"❌ Failed assets: {len(failed_assets)}")
        print(f"💾 Efficiency: Single data load for {len(target_timeframes)} timeframes")

        return {
            "results": results,
            "failed_assets": list(set(failed_assets)),
            "positive_event_count": len(positive_events),
            "negative_event_count": len(negative_events),
            "reference_timeframe": reference_timeframe,
            "target_timeframes": target_timeframes,
            "threshold_used": threshold,
            "comparison_used": comparison,
            "max_lags": max_lags,
            "bootstrap_iterations": bootstrap_iterations,
            "seed_used": seed,
            "efficiency_note": "Single data load optimization applied"
        }