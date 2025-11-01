import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
from core.data_loader import ReturnsLoader, BatchCorrelationLoader
from core.impact_analyzer import BTCImpactAnalyzer
from utils.impact_visualization import (
    create_beta_comparison_chart,
    create_scatter_with_regression,
    create_lag_response_heatmap,
    create_persistence_chart,
    create_rolling_beta_chart,
    create_cascade_waterfall,
    create_asymmetry_chart,
    create_pattern_detection_summary
)
import pandas as pd
from datetime import datetime
from scipy import stats


def test_asymmetry_significance(analyzer: BTCImpactAnalyzer,
                                ref_returns: np.ndarray,
                                target_returns: np.ndarray,
                                threshold: float) -> bool:
    """
    Statistical test to validate if asymmetry pattern is statistically significant
    - FIXED: Now uses conditional response analysis to get actual response arrays

    Args:
        analyzer: BTCImpactAnalyzer instance
        ref_returns: Reference asset returns
        target_returns: Target asset returns
        threshold: Event threshold

    Returns:
        bool: True if asymmetry is statistically significant (p < 0.05)
    """
    try:
        # Alternative approach: manually extract responses for events
        up_events_mask = ref_returns > threshold
        down_events_mask = ref_returns < -threshold

        up_responses = target_returns[up_events_mask]
        down_responses = target_returns[down_events_mask]

        # Filter out NaN values
        up_responses = up_responses[~np.isnan(up_responses)]
        down_responses = down_responses[~np.isnan(down_responses)]

        # Need sufficient data for reliable statistical test
        if len(up_responses) > 10 and len(down_responses) > 10:
            t_stat, p_value = stats.ttest_ind(up_responses, down_responses)
            return p_value < 0.05  # Significant at 95% confidence level

        return False

    except Exception as e:
        print(f"Warning: Asymmetry test failed: {e}")
        return False


def filter_outliers_robust(data: np.ndarray, lower_percentile: float = 5, upper_percentile: float = 95) -> np.ndarray:
    """
    Remove extreme outliers from data using percentile-based approach

    Why this is important:
    - Financial returns often have extreme outliers that can distort analysis
    - This preserves the main distribution while removing unrealistic extremes
    - Uses percentiles instead of standard deviation (more robust for finance data)

    Args:
        data: Array of numerical values to filter
        lower_percentile: Bottom percentile cutoff (default 5%)
        upper_percentile: Top percentile cutoff (default 95%)

    Returns:
        np.ndarray: Data with extreme outliers removed
    """
    if len(data) == 0:
        return data

    q_low = np.percentile(data, lower_percentile)
    q_high = np.percentile(data, upper_percentile)
    return data[(data >= q_low) & (data <= q_high)]


def render():
    """
    MAIN RENDERING FUNCTION - Dynamic Asset Impact Analysis Dashboard

    This module analyzes how other assets respond to movements in a reference asset (like BTC).
    It provides comprehensive insights into market dynamics, response patterns, and trading opportunities.

    Key Features:
    - Beta sensitivity analysis (how much target assets move with reference)
    - Asymmetry detection (do assets respond differently to ups vs downs?)
    - Lag analysis (how quickly do assets respond?)
    - Cascade effects (what's the order of response across multiple assets?)
    - Pattern detection (overreaction, persistence, momentum patterns)
    - Rolling analysis (how relationships change over time)

    Target Users:
    - Traders looking for pairs trading opportunities
    - Portfolio managers assessing risk exposure
    - Researchers studying market dynamics
    - Analysts identifying market leadership patterns
    """

    st.markdown("## 🎯 Dynamic Asset Impact Analysis")
    st.info("""
    **What This Dashboard Does:**
    - Analyzes how other assets respond to movements in a reference asset (default: BTC)
    - Detects sophisticated patterns: overreaction, asymmetries, persistence, cascade effects
    - Provides quantitative metrics: Beta, R-squared, response percentages, statistical significance
    - Helps identify trading opportunities and risk management insights
    """)

    # ===== SIDEBAR: ANALYSIS CONFIGURATION =====
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")

        with st.expander("📊 Asset Selection", expanded=True):
            # Discover available assets in data directory
            loader = ReturnsLoader(data_dir="data", cache_dir="cache/processed")
            available_assets = loader.discover_assets()

            # Reference asset selection (the influencer)
            reference_asset = st.selectbox(
                "Reference Asset:",
                options=available_assets,
                index=available_assets.index("BTCUSDT") if "BTCUSDT" in available_assets else 0,
                help="Asset whose price movements will be analyzed for impact on others"
            )

            # Target assets selection (the responders)
            default_targets = ["ETHUSDT", "SOLUSDT", "XRPUSDT"]
            # Ensure default targets exist and aren't the reference asset
            default_targets = [a for a in default_targets if a in available_assets and a != reference_asset]

            target_assets = st.multiselect(
                "Target Assets:",
                options=[a for a in available_assets if a != reference_asset],
                default=default_targets[:3] if default_targets else [],
                help="Assets that will be analyzed for response to reference asset movements"
            )

            if not target_assets:
                st.warning("⚠️ Please select at least 1 target asset")

        with st.expander("🕐 Time Configuration", expanded=True):
            # Timeframe for analysis (granularity of data)
            timeframe = st.selectbox(
                "Analysis Timeframe:",
                options=['1m', '5m', '10m', '15m', '30m', '1h'],
                index=2,
                help="Time interval for calculating returns and analyzing responses"
            )

            # Optional date filtering for focused analysis
            use_date_filter = st.checkbox("Filter by Date Range", value=False)

            if use_date_filter:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date:",
                        value=pd.Timestamp('2024-01-01').date()
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date:",
                        value=pd.Timestamp('2024-12-31').date()
                    )
            else:
                start_date = None
                end_date = None

        with st.expander("🎚️ Analysis Parameters", expanded=True):
            # Event threshold - minimum movement to consider as significant
            threshold = st.slider(
                "Event Threshold (%):",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Minimum percentage change in reference asset to consider as a significant event"
            ) / 100  # Convert to decimal for calculations

            # Maximum lags to analyze for response timing
            max_lag = st.slider(
                "Maximum Lags:",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Number of future time periods to analyze for response patterns"
            )

            # Rolling window for dynamic beta analysis
            rolling_window = st.slider(
                "Rolling Beta Window:",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Number of periods for calculating rolling beta (shows how sensitivity changes over time)"
            )

        # Main analysis execution button
        analyze_button = st.button("🚀 Run Impact Analysis", type="primary")

    # ===== INITIALIZE SESSION STATE =====
    # Session state preserves analysis results between user interactions
    if 'impact_analysis_done' not in st.session_state:
        st.session_state.impact_analysis_done = False
        st.session_state.impact_results = None

    # ===== EXECUTE ANALYSIS =====
    if analyze_button and target_assets:
        with st.spinner("Loading and processing market data..."):
            try:
                # 1. Load individual asset data for precise analysis
                all_assets = [reference_asset] + target_assets
                dfs = {}

                for asset in all_assets:
                    file_path = Path("data") / asset / f"{asset}.parquet"

                    if not file_path.exists():
                        st.error(f"❌ Data file not found for {asset}")
                        continue

                    # Load parquet file for this asset
                    df = pl.read_parquet(file_path)

                    # Validate required columns exist
                    if "1min_returns" not in df.columns or "open_time" not in df.columns:
                        st.error(f"❌ {asset}: Required columns missing")
                        continue

                    # Normalize timestamp format for consistency
                    df = df.with_columns([
                        pl.col("open_time").cast(pl.Datetime(time_unit="us"))
                    ])

                    # Select only needed columns and rename for clarity
                    df = df.select(["open_time", "1min_returns"])
                    df = df.rename({"1min_returns": f"{asset}_returns"})

                    # Apply date filtering if requested by user
                    if use_date_filter and start_date and end_date:
                        start_dt = pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date)
                        df = df.filter(
                            (pl.col("open_time") >= start_dt) &
                            (pl.col("open_time") <= end_dt)
                        )

                    dfs[asset] = df

                # Validate we have enough assets loaded
                if len(dfs) < 2:
                    st.error("❌ Insufficient assets loaded for analysis")
                    st.stop()

                # 2. Initialize the impact analyzer
                analyzer = BTCImpactAnalyzer(reference_asset=reference_asset)

                # 3. KEY IMPROVEMENT: Create individual DataFrames for each asset pair
                # This ensures precise alignment and avoids date mismatch issues
                results = {}
                individual_dfs = {}  # Store individual pair DataFrames for later use

                # First display data availability statistics
                st.markdown("### 📊 Data Availability by Asset")
                stats_data = []

                for asset in all_assets:
                    if asset in dfs:
                        total_rows = dfs[asset].height
                        start_date_asset = dfs[asset]["open_time"].min()
                        end_date_asset = dfs[asset]["open_time"].max()

                        stats_data.append({
                            'Asset': asset,
                            'Total Periods': f"{total_rows:,}",
                            'Start Date': start_date_asset.strftime('%Y-%m-%d'),
                            'End Date': end_date_asset.strftime('%Y-%m-%d')
                        })

                st.dataframe(stats_data, use_container_width=True)

                # Analyze each target asset individually against reference
                for target_asset in target_assets:
                    if target_asset not in dfs:
                        continue

                    # Create specific DataFrame for this reference-target pair
                    ref_df = dfs[reference_asset]
                    target_df = dfs[target_asset]

                    # INNER JOIN ensures we only analyze periods with data for both assets
                    pair_df = ref_df.join(
                        target_df,
                        on="open_time",
                        how="inner"  # Critical: only periods with both assets available
                    ).sort("open_time")

                    # Resample to desired timeframe if not 1-minute
                    if timeframe != '1m':
                        pair_df = analyzer.resample_to_timeframe(pair_df, timeframe)

                    # Remove any NaN values for clean analysis
                    pair_df = pair_df.drop_nulls()

                    # Validate we have sufficient overlapping data
                    if pair_df.height < 100:
                        st.warning(f"⚠️ {target_asset}: Only {pair_df.height} common periods with {reference_asset}")
                        continue

                    # Store for later use in visualizations
                    individual_dfs[target_asset] = pair_df

                    st.info(f"📈 {target_asset}: {pair_df.height:,} common periods with {reference_asset}")

                    # Perform comprehensive impact analysis for this pair
                    reference_col = f"{reference_asset}_returns"
                    target_col = f"{target_asset}_returns"

                    target_results = analyzer.analyze_complete(
                        combined_df=pair_df,
                        reference_col=reference_col,
                        target_cols=[target_col],
                        threshold=threshold,
                        max_lag=max_lag,
                        rolling_window=rolling_window
                    )

                    if target_asset in target_results:
                        results[target_asset] = target_results[target_asset]

                # Check if we have any valid results
                if not results:
                    st.error("❌ No valid results calculated for any target assets")
                    st.stop()

                # 4. IMPROVED CASCADE ANALYSIS with robust period validation
                cascade_results = []

                if len(individual_dfs) >= 2:
                    # Find the maximum common period across all assets
                    common_start = None
                    common_end = None

                    for asset, df in individual_dfs.items():
                        df_start = df["open_time"].min()
                        df_end = df["open_time"].max()

                        # Find the narrowest common date range
                        if common_start is None or df_start > common_start:
                            common_start = df_start
                        if common_end is None or df_end < common_end:
                            common_end = df_end

                    # Validate we have a meaningful common period
                    if common_start and common_end and common_start < common_end:
                        st.info(
                            f"📅 Common period for cascade analysis: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")

                        # Create arrays for cascade analysis using only the common period
                        ref_returns_common = None
                        target_returns_dict_common = {}

                        for target_asset in target_assets:
                            if target_asset in individual_dfs:
                                # Filter to common period only
                                df_common = individual_dfs[target_asset].filter(
                                    (pl.col("open_time") >= common_start) &
                                    (pl.col("open_time") <= common_end)
                                )

                                # IMPROVEMENT: Increased minimum for reliable cascade analysis
                                if df_common.height >= 100:  # More periods for reliable analysis
                                    if ref_returns_common is None:
                                        ref_returns_common = df_common[f"{reference_asset}_returns"].to_numpy()
                                    target_returns_dict_common[target_asset] = df_common[
                                        f"{target_asset}_returns"].to_numpy()

                        # Perform cascade analysis if we have sufficient common data
                        if ref_returns_common is not None and len(target_returns_dict_common) >= 2:
                            cascade_results = analyzer.detect_cascade_order(
                                ref_returns=ref_returns_common,
                                target_returns_dict=target_returns_dict_common,
                                max_lag=max_lag
                            )
                        else:
                            st.warning("⚠️ Insufficient common period data for reliable cascade analysis")
                    else:
                        st.warning("⚠️ No sufficient common time period found across assets for cascade analysis")
                else:
                    st.warning("⚠️ Need at least 2 assets with common data for cascade analysis")

                # 5. Store results in session state for persistence
                st.session_state.impact_analysis_done = True
                st.session_state.impact_results = {
                    'results': results,
                    'cascade': cascade_results,
                    'individual_dfs': individual_dfs,
                    'reference_asset': reference_asset,
                    'target_assets': target_assets,
                    'threshold': threshold,
                    'max_lag': max_lag,
                    'rolling_window': rolling_window,
                    'timeframe': timeframe
                }

                st.success("🎉 Analysis completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                import traceback
                with st.expander("🛠 View technical error details"):
                    st.code(traceback.format_exc())

    # ===== DISPLAY RESULTS =====
    if st.session_state.impact_analysis_done and st.session_state.impact_results:
        data = st.session_state.impact_results
        results = data['results']
        cascade_results = data['cascade']
        individual_dfs = data['individual_dfs']
        reference_asset = data['reference_asset']
        target_assets = data['target_assets']
        threshold = data['threshold']
        max_lag = data['max_lag']
        rolling_window = data['rolling_window']
        timeframe = data['timeframe']

        st.markdown("---")
        st.markdown("## 📊 Impact Analysis Results")

        # Display key metrics for each asset in columns
        st.markdown("### 📈 Key Metrics by Asset")

        cols = st.columns(len(results))
        for i, (asset_name, result) in enumerate(results.items()):
            with cols[i]:
                st.metric(
                    f"**{asset_name}**",
                    f"Beta: {result.beta:.3f}",
                    f"±{(result.beta_ci_upper - result.beta_ci_lower) / 2:.3f}"  # Confidence interval half-width
                )
                st.caption(f"R²: {result.r_squared:.3f}")  # Goodness of fit
                st.caption(f"Sample Size: {result.n_points:,}")  # Data points used

        # TABBED INTERFACE FOR DIFFERENT ANALYSIS PERSPECTIVES
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "⬆️ Up Moves",
            "⬇️ Down Moves",
            "🔍 Patterns",
            "🌊 Cascade"
        ])

        with tab1:
            st.markdown("### 📊 General Relationship Analysis")

            # Beta comparison across all assets
            fig_beta = create_beta_comparison_chart(results, reference_asset)
            st.plotly_chart(fig_beta, use_container_width=True)

            # Scatter plots with regression lines
            st.markdown("### 📈 Return Relationship Scatter Plots")
            selected_for_scatter = st.selectbox(
                "Select asset for detailed scatter analysis:",
                options=list(results.keys()),
                key="scatter_select"
            )

            if selected_for_scatter and selected_for_scatter in individual_dfs:
                result = results[selected_for_scatter]
                pair_df = individual_dfs[selected_for_scatter]

                ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                target_returns = pair_df[f"{selected_for_scatter}_returns"].to_numpy()

                fig_scatter = create_scatter_with_regression(
                    ref_returns=ref_returns,
                    target_returns=target_returns,
                    asset_name=selected_for_scatter,
                    beta=result.beta,
                    r_squared=result.r_squared
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Comprehensive results table
            st.markdown("### 📋 Detailed Results Summary")
            summary_data = []
            for asset_name, result in results.items():
                summary_data.append({
                    'Asset': asset_name,
                    'Beta': f"{result.beta:.3f}",
                    '95% CI': f"[{result.beta_ci_lower:.3f}, {result.beta_ci_upper:.3f}]",
                    'R²': f"{result.r_squared:.3f}",
                    'P-value': f"{result.p_value:.3e}",
                    'Data Points': f"{result.n_points:,}"
                })

            st.dataframe(summary_data, use_container_width=True)

        with tab2:
            st.markdown("### ⬆️ Analysis of UP Movements")
            st.info(f"Events where {reference_asset} moves UP more than {threshold * 100:.1f}%")

            # Extract conditional analysis for UP movements
            conditional_up = {}
            for asset in target_assets:
                if asset in individual_dfs:
                    pair_df = individual_dfs[asset]
                    ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                    target_returns = pair_df[f"{asset}_returns"].to_numpy()

                    analyzer_temp = BTCImpactAnalyzer(reference_asset)
                    cond_result = analyzer_temp.calculate_conditional_response(
                        ref_returns, target_returns, threshold, max_lag
                    )
                    conditional_up[asset] = cond_result

            # Heatmap visualization of lag responses
            if conditional_up:
                fig_heatmap_up = create_lag_response_heatmap(
                    conditional_results=conditional_up,
                    event_type='up_events',
                    assets=list(conditional_up.keys()),
                    max_lag=max_lag
                )
                st.plotly_chart(fig_heatmap_up, use_container_width=True)

            # Persistence analysis chart
            fig_persist = create_persistence_chart(results)
            st.plotly_chart(fig_persist, use_container_width=True)

            # Detailed metrics for each asset
            st.markdown("### 📋 Asset Details - UP Movements")
            for asset_name, result in results.items():
                with st.expander(f"📊 {asset_name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Response", f"{result.mean_response_up * 100:.2f}%")
                    with col2:
                        st.metric("Persistence", f"{result.persistence_up:.1f}%")
                    with col3:
                        st.metric("Avg Consecutive Periods", f"{result.avg_consecutive_up:.1f}")

        with tab3:
            st.markdown("### ⬇️ Analysis of DOWN Movements")
            st.info(f"Events where {reference_asset} moves DOWN more than {threshold * 100:.1f}%")

            # Extract conditional analysis for DOWN movements
            conditional_down = {}
            for asset in target_assets:
                if asset in individual_dfs:
                    pair_df = individual_dfs[asset]
                    ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                    target_returns = pair_df[f"{asset}_returns"].to_numpy()

                    analyzer_temp = BTCImpactAnalyzer(reference_asset)
                    cond_result = analyzer_temp.calculate_conditional_response(
                        ref_returns, target_returns, threshold, max_lag
                    )
                    conditional_down[asset] = cond_result

            # Heatmap visualization for down movements
            if conditional_down:
                fig_heatmap_down = create_lag_response_heatmap(
                    conditional_results=conditional_down,
                    event_type='down_events',
                    assets=list(conditional_down.keys()),
                    max_lag=max_lag
                )
                st.plotly_chart(fig_heatmap_down, use_container_width=True)

            # Asymmetry analysis chart
            fig_asym = create_asymmetry_chart(results)
            st.plotly_chart(fig_asym, use_container_width=True)

            # Detailed metrics with IMPROVED statistical validation
            st.markdown("### 📋 Asset Details - DOWN Movements")
            for asset_name, result in results.items():
                with st.expander(f"📊 {asset_name}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Response", f"{result.mean_response_down * 100:.2f}%")
                    with col2:
                        st.metric("Persistence", f"{result.persistence_down:.1f}%")
                    with col3:
                        st.metric("Avg Consecutive Periods", f"{result.avg_consecutive_down:.1f}")
                    with col4:
                        st.metric("Asymmetry Ratio", f"{result.asymmetry_ratio:.2f}")

                    # IMPROVEMENT: Statistical validation of asymmetry patterns
                    if asset_name in individual_dfs:
                        pair_df = individual_dfs[asset_name]
                        ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                        target_returns = pair_df[f"{asset_name}_returns"].to_numpy()

                        analyzer_temp = BTCImpactAnalyzer(reference_asset)
                        is_significant = test_asymmetry_significance(
                            analyzer_temp, ref_returns, target_returns, threshold
                        )

                        if (result.asymmetry_ratio > 1.2 and is_significant):
                            st.warning(
                                "⚠️ **Statistically Significant Bearish Asymmetry**: Stronger response to DOWN moves")
                        elif (result.asymmetry_ratio < 0.8 and is_significant):
                            st.info("ℹ️ **Statistically Significant Bullish Asymmetry**: Stronger response to UP moves")
                        elif result.asymmetry_ratio > 1.2:
                            st.warning("⚠️ **Potential Bearish Asymmetry**: Appears stronger on DOWN moves")
                        elif result.asymmetry_ratio < 0.8:
                            st.info("ℹ️ **Potential Bullish Asymmetry**: Appears stronger on UP moves")

        with tab4:
            st.markdown("### 🔍 Advanced Pattern Detection")

            # Pattern detection dashboard
            fig_patterns = create_pattern_detection_summary(results, reference_asset)
            st.plotly_chart(fig_patterns, use_container_width=True)

            # Pattern alerts with detailed explanations
            st.markdown("### 🚨 Detected Trading Patterns")
            for asset_name, result in results.items():
                patterns_found = []

                # Overreaction pattern
                if result.overreaction_score > 1.5:
                    patterns_found.append("🔥 **Overreaction**: Moves more than expected based on beta")

                # Bearish asymmetry (statistically validated)
                if asset_name in individual_dfs:
                    pair_df = individual_dfs[asset_name]
                    ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                    target_returns = pair_df[f"{asset_name}_returns"].to_numpy()

                    analyzer_temp = BTCImpactAnalyzer(reference_asset)
                    is_significant = test_asymmetry_significance(
                        analyzer_temp, ref_returns, target_returns, threshold
                    )

                    if (result.asymmetry_ratio > 1.2 and is_significant):
                        patterns_found.append("⚠️ **Bearish Asymmetry**: Statistically stronger response to declines")

                # High sensitivity pattern
                if result.beta > 1.3:
                    patterns_found.append("⚡ **High Sensitivity**: Beta > 1.3 indicates amplified movements")

                # Persistence patterns
                if result.persistence_up > 70 and result.persistence_down > 70:
                    patterns_found.append("💪 **High Persistence**: >70% continuation in both directions")

                if result.persistence_up < 50 or result.persistence_down < 50:
                    patterns_found.append("🔄 **Low Persistence**: Potential mean reversion behavior")

                # Display patterns if any found
                if patterns_found:
                    st.markdown(f"**{asset_name}:**")
                    for pattern in patterns_found:
                        st.markdown(f"- {pattern}")
                    st.markdown("---")

            # Rolling Beta Analysis with OUTLIER PROTECTION
            st.markdown("### 📈 Rolling Beta Analysis (Time Evolution)")
            selected_for_rolling = st.selectbox(
                "Select asset for rolling analysis:",
                options=list(results.keys()),
                key="rolling_select"
            )

            if selected_for_rolling and selected_for_rolling in individual_dfs:
                pair_df = individual_dfs[selected_for_rolling]
                ref_returns = pair_df[f"{reference_asset}_returns"].to_numpy()
                target_returns = pair_df[f"{selected_for_rolling}_returns"].to_numpy()
                dates = pair_df["open_time"].to_numpy()

                analyzer_temp = BTCImpactAnalyzer(reference_asset)
                rolling_betas, valid_indices = analyzer_temp.calculate_rolling_beta(
                    ref_returns, target_returns, window=rolling_window
                )

                if len(rolling_betas) > 0:
                    dates_rolling = dates[valid_indices]

                    # IMPROVEMENT: Filter outliers for more robust statistics
                    rolling_betas_filtered = filter_outliers_robust(rolling_betas)

                    fig_rolling = create_rolling_beta_chart(
                        dates=dates_rolling,
                        rolling_betas=rolling_betas_filtered,
                        asset_name=selected_for_rolling,
                        overall_beta=results[selected_for_rolling].beta
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)

                    # Statistics using FILTERED data (more reliable)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Filtered Beta Min", f"{rolling_betas_filtered.min():.3f}")
                    with col2:
                        st.metric("Filtered Beta Max", f"{rolling_betas_filtered.max():.3f}")
                    with col3:
                        st.metric("Filtered Std Dev", f"{rolling_betas_filtered.std():.3f}")
                else:
                    st.warning("Insufficient data for rolling beta analysis")

        with tab5:
            st.markdown("### 🌊 Market Cascade Analysis")
            st.info(
                "Identifies the order in which assets respond to reference movements - useful for momentum strategies")

            if cascade_results:
                # Waterfall visualization of response order
                fig_cascade = create_cascade_waterfall(cascade_results)
                st.plotly_chart(fig_cascade, use_container_width=True)

                # Detailed cascade metrics
                st.markdown("### 📋 Cascade Response Details")
                cascade_data = []
                for cascade in cascade_results:
                    cascade_data.append({
                        'Order': f"#{cascade.cascade_order}",
                        'Asset': cascade.asset_name,
                        'Optimal Lag': cascade.optimal_lag,
                        'Max Correlation': f"{cascade.max_correlation:.3f}",
                        'Avg Response Time': f"{cascade.response_time_avg:.2f} periods"
                    })

                st.dataframe(cascade_data, use_container_width=True)

                # Practical interpretation for traders
                st.markdown("### 💡 Trading Insights from Cascade Analysis")
                fastest = cascade_results[0]
                slowest = cascade_results[-1]

                st.success(
                    f"🏃 **Fastest Responder**: {fastest.asset_name} (reacts in ~{fastest.response_time_avg:.1f} periods)")
                st.warning(
                    f"🐢 **Slowest Responder**: {slowest.asset_name} (reacts in ~{slowest.response_time_avg:.1f} periods)")

                # Trading strategy insights
                if fastest.response_time_avg < 2:
                    st.info(
                        "💡 **Momentum Opportunity**: Some assets respond very quickly (<2 periods). Consider short-term momentum strategies.")

                if len(cascade_results) >= 3:
                    st.info(
                        "💡 **Diversified Response**: Multiple response speeds detected. Opportunities for both fast and slow strategies.")
            else:
                st.warning("No sufficient common time period across assets for cascade analysis")

    else:
        # Welcome and instructions for new users
        st.info("""
        👈 **Getting Started Guide:**

        1. **Select Reference Asset** (e.g., BTCUSDT) - the market influencer
        2. **Choose Target Assets** (e.g., ETH, SOL, XRP) - the assets you want to analyze
        3. **Set Timeframe** (5m, 10m, 15m, 30m, 1h) - data granularity for analysis
        4. **Adjust Event Threshold** - minimum % move in reference to consider significant
        5. **Click 🚀 Run Impact Analysis** to generate insights

        **What You'll Discover:**
        - 📊 **Beta Sensitivity**: How much each target moves with the reference
        - ⬆️⬇️ **Asymmetric Responses**: Do assets respond differently to ups vs downs?
        - 🔍 **Trading Patterns**: Overreaction, persistence, momentum opportunities
        - 🌊 **Cascade Effects**: Which assets respond first/last to market moves
        - 📈 **Time Evolution**: How relationships change over different market conditions

        **Perfect For:**
        - Pairs trading strategy development
        - Risk management and portfolio construction
        - Market microstructure research
        - Trading opportunity identification
        """)


if __name__ == "__main__":
    render()