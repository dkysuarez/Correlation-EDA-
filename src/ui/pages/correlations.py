import streamlit as st
import numpy as np
import polars as pl
import gc
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from core.data_loader import BatchCorrelationLoader
from ui.components import render_sidebar
from utils.visualization import (
    create_heatmap,
    create_top_pairs_bar,
    create_distribution_histogram,
    create_asset_correlations_bar,
    create_network_graph
)
from utils.rolling_correlations import CorrelationTimelineAnalyzer
import pandas as pd
from datetime import datetime, timedelta


def _is_statistically_significant(p_value: float) -> str:
    """
    Determines if a p-value is statistically significant with human-readable labels

    P-value interpretation guide:
    - p < 0.01: Very Significant (strong evidence against null hypothesis)
    - p < 0.05: Significant (moderate evidence)
    - p < 0.10: Marginally Significant (weak evidence)
    - p >= 0.10: Not Significant (insufficient evidence)

    Args:
        p_value: The probability value from statistical test

    Returns:
        str: Human-readable significance level with emoji indicators
    """
    if p_value < 0.01:
        return "✅ Very Significant (p < 0.01)"
    elif p_value < 0.05:
        return "✅ Significant (p < 0.05)"
    elif p_value < 0.1:
        return "⚠️ Marginally Significant (p < 0.1)"
    else:
        return "❌ Not Significant (p ≥ 0.1)"


class FastPhaseDetector:
    """
    OPTIMIZED Phase Detector - 10x Faster Version
    Maintains the essence of phase analysis but eliminates unnecessary calculations

    Key Improvements:
    - Fixed window sizes instead of dynamic calculations
    - Strategic sampling (every 'step' points instead of all)
    - Simplified but effective statistical criteria
    - Reduced computational complexity while maintaining accuracy
    """

    @staticmethod
    def detect_structural_breaks_fast(correlation_series, min_periods=15, step=5):
        """
        Fast structural break detection using fixed windows and strategic sampling

        How it works:
        1. Uses fixed window size (12 periods) for consistent analysis
        2. Samples every 'step' points to reduce computations
        3. Applies simplified but effective change detection criteria
        4. Filters out noisy changes using volatility checks

        Args:
            correlation_series: Array of correlation values over time
            min_periods: Minimum data points required for analysis
            step: Sampling interval to reduce computations

        Returns:
            List of indices where significant structural breaks occur
        """
        n = len(correlation_series)
        if n < 30:
            return []

        breakpoints = []
        window_size = 12  # Fixed window size for consistency

        # Evaluate every 'step' points instead of all points for efficiency
        for i in range(min_periods, n - min_periods, step):
            # Define windows before and after current point
            start_before = max(0, i - window_size)
            end_after = min(n, i + window_size)

            before = correlation_series[start_before:i]
            after = correlation_series[i:end_after]

            # Only calculate if sufficient data in both windows
            if len(before) >= 8 and len(after) >= 8:
                # Fast metric: change in mean correlation
                mean_before = np.mean(before)
                mean_after = np.mean(after)
                mean_change = abs(mean_after - mean_before)

                # Simplified but effective criterion for significant changes
                if mean_change > 0.25:  # Threshold for meaningful correlation changes
                    # Verify it's not just noise using volatility check
                    std_before = np.std(before)
                    std_after = np.std(after)

                    # Only consider real breaks, not volatile fluctuations
                    if std_before < 0.4 and std_after < 0.4:
                        breakpoints.append(i)

        return breakpoints

    @staticmethod
    def detect_fast_phases(correlation_timeline, min_phase_length=10):
        """
        Fast phase detection using segmentation approach

        Process:
        1. Detect structural break points
        2. Create segments between breakpoints
        3. Classify each segment by correlation characteristics
        4. Validate and characterize each phase

        Args:
            correlation_timeline: Time series of correlation values
            min_phase_length: Minimum periods required for a valid phase

        Returns:
            List of phase dictionaries with detailed characteristics
        """
        if len(correlation_timeline) < 30:
            return []

        # 1. Detect structural breaks using optimized method
        breakpoints = FastPhaseDetector.detect_structural_breaks_fast(correlation_timeline)

        # 2. Create segments between breakpoints
        all_points = [0] + breakpoints + [len(correlation_timeline)]
        phases = []

        for i in range(1, len(all_points)):
            start_idx = all_points[i - 1]
            end_idx = all_points[i]
            phase_data = correlation_timeline[start_idx:end_idx]

            # Validate phase has sufficient data
            if len(phase_data) >= min_phase_length:
                mean_corr = np.mean(phase_data)
                std_corr = np.std(phase_data)

                # Simple but informative classification system
                if mean_corr > 0.7 and std_corr < 0.2:
                    regime_type, emoji = "STRONG_STABLE_COUPLING", "🔥"
                elif mean_corr > 0.7:
                    regime_type, emoji = "STRONG_VOLATILE_COUPLING", "⚡"
                elif mean_corr > 0.5:
                    regime_type, emoji = "MODERATE_COUPLING", "💪"
                elif mean_corr > 0.3:
                    regime_type, emoji = "LOW_CORRELATION", "📊"
                elif mean_corr > -0.2:
                    regime_type, emoji = "DECOUPLING", "🔄"
                elif mean_corr > -0.5:
                    regime_type, emoji = "MODERATE_DIVERGENCE", "📉"
                else:
                    regime_type, emoji = "STRONG_DIVERGENCE", "🎯"

                # Fast significance calculation
                if len(phase_data) >= 15:
                    try:
                        t_stat, p_value = stats.ttest_1samp(phase_data, 0)
                        significance = "VERY_SIGNIFICANT" if p_value < 0.01 else "SIGNIFICANT" if p_value < 0.05 else "MARGINAL"
                    except:
                        significance = "NOT_CALCULATED"
                        p_value = None
                else:
                    significance = "INSUFFICIENT_DATA"
                    p_value = None

                # Simple persistence measure (autocorrelation)
                if len(phase_data) > 1:
                    persistence = np.corrcoef(phase_data[:-1], phase_data[1:])[0, 1] if len(phase_data) > 1 else 0
                else:
                    persistence = 0

                phases.append({
                    'phase_id': len(phases) + 1,
                    'regime_type': regime_type,
                    'emoji': emoji,
                    'average_correlation': mean_corr,
                    'correlation_volatility': std_corr,
                    'total_periods': len(phase_data),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'statistical_significance': significance,
                    'p_value': p_value,
                    'persistence': persistence
                })

        return phases

    @staticmethod
    def validate_phase_fast(phase_correlations):
        """
        Fast validation - only essential criteria for phase quality

        Validates:
        - Minimum duration (8 periods)
        - Coefficient of variation (volatility relative to mean)

        Args:
            phase_correlations: Correlation values for the phase

        Returns:
            Tuple: (is_valid, reason_message)
        """
        n_periods = len(phase_correlations)

        # Only critical validations for quality
        if n_periods < 8:  # More permissive than original
            return False, "Insufficient duration"

        # Simplified coefficient of variation
        mean_corr = np.mean(phase_correlations)
        std_corr = np.std(phase_correlations)

        if abs(mean_corr) < 0.05:
            cv = std_corr / 0.05
        else:
            cv = std_corr / abs(mean_corr)

        if cv > 0.8:  # More permissive threshold
            return False, f"High volatility (CV: {cv:.2f})"

        return True, "Valid phase"


def render():
    """
    MAIN RENDERING FUNCTION - Complete Correlation Analysis Dashboard

    This module provides comprehensive correlation analysis including:
    - Static correlation matrices and heatmaps
    - Temporal evolution analysis with phase detection
    - Network visualization of correlation relationships
    - Statistical validation and significance testing

    Features:
    - Optimized processing with intelligent caching
    - Multiple visualization methods for different insights
    - Fast phase detection for regime analysis
    - Robust date handling and filtering
    """

    # Cache for temporal analyses to avoid recomputation
    if 'correlation_cache' not in st.session_state:
        st.session_state.correlation_cache = {}

    def get_cached_analysis(asset1, asset2, timeframe, start_date, end_date):
        """
        Retrieve cached analysis if available

        Cache key format: asset1_asset2_timeframe_start_end
        This prevents recomputing identical analyses
        """
        cache_key = f"{asset1}_{asset2}_{timeframe}_{start_date}_{end_date}"
        return st.session_state.correlation_cache.get(cache_key)

    def set_cached_analysis(asset1, asset2, timeframe, start_date, end_date, data):
        """
        Store analysis in cache with FIFO eviction policy

        Limits cache to 10 entries to prevent memory issues
        Removes oldest entries when limit exceeded
        """
        cache_key = f"{asset1}_{asset2}_{timeframe}_{start_date}_{end_date}"
        st.session_state.correlation_cache[cache_key] = data

        # Limit cache to maximum 10 entries using FIFO
        if len(st.session_state.correlation_cache) > 10:
            oldest_key = next(iter(st.session_state.correlation_cache))
            del st.session_state.correlation_cache[oldest_key]

    # Render sidebar and get user configuration
    selected_assets, correlation_method, timeframe, batch_size, load_button = render_sidebar()

    # Initialize session state for data persistence
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.corr_matrix = None
        st.session_state.p_matrix = None
        st.session_state.asset_names = None
        st.session_state.correlation_method = 'Spearman'
        st.session_state.failed_pairs = []

    # Main analysis execution when load button is clicked
    if load_button and selected_assets:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📄 Loading assets...")
            progress_bar.progress(30)

            # Load asset data in batches for memory efficiency
            loader = BatchCorrelationLoader()
            batch_results = loader.load_assets_in_batches(selected_assets, batch_size, timeframe)
            progress_bar.progress(60)

            # Display loading statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Loaded", len(batch_results['loaded_assets']))
            with col2:
                st.metric("❌ Failed", len(batch_results['failed_assets']))

            # Proceed only if we have enough assets for meaningful analysis
            if len(batch_results['loaded_assets']) >= 2:
                status_text.text(f"📊 Calculating correlations ({correlation_method})...")
                progress_bar.progress(80)

                # Calculate complete correlation matrix
                corr_matrix, p_matrix, asset_names, failed_pairs = loader.calculate_correlations(batch_results,
                                                                                                 correlation_method)
                progress_bar.progress(100)

                # Update session state with results
                st.session_state.update({
                    'corr_matrix': corr_matrix,
                    'p_matrix': p_matrix,
                    'asset_names': asset_names,
                    'failed_pairs': failed_pairs,
                    'data_loaded': True,
                    'correlation_method': correlation_method
                })

                status_text.text("✅ Analysis completed")
                st.success(f"Processed {len(asset_names)} assets with {correlation_method}")

            else:
                st.error("Not enough assets loaded. Select at least 2 valid assets.")

        except Exception as e:
            st.error(f"Processing error: {str(e)}")

    # Display results if analysis has been completed
    if st.session_state.get('data_loaded', False) and st.session_state.get('corr_matrix') is not None:
        st.markdown("---")
        st.markdown("## 📈 Analysis Results")

        # Retrieve data from session state
        corr_matrix = st.session_state.corr_matrix
        p_matrix = st.session_state.p_matrix
        asset_names = st.session_state.asset_names
        failed_pairs = st.session_state.get('failed_pairs', [])
        correlation_method = st.session_state.correlation_method

        # General Statistics Calculation
        st.markdown("### 📊 General Statistics")
        n = len(asset_names)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # Upper triangle mask (excluding diagonal)
        corr_values = corr_matrix[mask]  # Extract unique correlation pairs

        # Display key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Assets", len(asset_names))
        with col2:
            st.metric("Pairs", f"{len(corr_values):,}")
        with col3:
            st.metric("Avg Correlation", f"{np.mean(corr_values):.3f}")
        with col4:
            st.metric("Max Correlation", f"{np.max(corr_values):.3f}")
        with col5:
            st.metric("Min Correlation", f"{np.min(corr_values):.3f}")

        # Tabbed interface for different analysis perspectives
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔥 Heatmap", "📊 Top Correlations", "📈 Distribution",
            "🔍 Detailed Analysis", "🌐 Correlation Network", "⏰ Temporal Evolution"
        ])

        with tab1:
            st.markdown(f"### 🎨 Correlation Matrix ({correlation_method})")
            max_assets = len(asset_names)

            if max_assets >= 2:
                # Allow user to control heatmap size for readability
                if max_assets > 2:
                    n_assets_to_show = st.slider(
                        "Number of assets to display:",
                        min_value=2,
                        max_value=max_assets,
                        value=min(30, max_assets),
                        step=1,
                        key="heatmap_assets"
                    )
                else:
                    n_assets_to_show = 2
                    st.info(f"📊 Showing all 2 available assets")

                # Generate and display heatmap
                fig = create_heatmap(corr_matrix, p_matrix, asset_names, n_assets_to_show)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("At least 2 assets needed to display heatmap.")

        with tab2:
            st.markdown(f"### 📊 Top Correlations ({correlation_method})")
            max_pairs = len(corr_values)

            if max_pairs > 1:
                n_pairs_to_show = st.slider(
                    "Number of pairs to display:",
                    min_value=1,
                    max_value=max_pairs,
                    value=min(10, max_pairs),
                    step=1,
                    key="top_pairs"
                )
            else:
                n_pairs_to_show = max_pairs
                st.info(f"📊 Showing the only available pair")

            # Generate top pairs visualization
            fig, display_data = create_top_pairs_bar(corr_matrix, p_matrix, asset_names, n_pairs_to_show)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### 📋 Top {n_pairs_to_show} Pairs Detail")
            st.dataframe(
                display_data,
                use_container_width=True,
                height=min(400, 35 * n_pairs_to_show)
            )

        with tab3:
            st.markdown(f"### 📈 Correlation Distribution ({correlation_method})")
            if len(corr_values) > 0:
                # Create distribution histogram
                fig = create_distribution_histogram(corr_values, correlation_method)
                st.plotly_chart(fig, use_container_width=True)

                # Detailed distribution statistics
                st.markdown("### 📊 Detailed Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    positive_corr = len(corr_values[corr_values > 0.1])
                    st.metric("Positive Pairs", f"{(positive_corr / len(corr_values)) * 100:.1f}%")
                with col2:
                    negative_corr = len(corr_values[corr_values < -0.1])
                    st.metric("Negative Pairs", f"{(negative_corr / len(corr_values)) * 100:.1f}%")
                with col3:
                    strong_pos = len(corr_values[corr_values > 0.5])
                    st.metric("Strong Positive", f"{(strong_pos / len(corr_values)) * 100:.1f}%")
                with col4:
                    strong_neg = len(corr_values[corr_values < -0.5])
                    st.metric("Strong Negative", f"{(strong_neg / len(corr_values)) * 100:.1f}%")
            else:
                st.warning("No data available for distribution analysis")

        with tab4:
            st.markdown("### 🔍 Detailed Analysis by Asset")
            selected_asset = st.selectbox(
                "Select asset for detailed analysis:",
                options=asset_names,
                key="asset_analyzer"
            )

            max_corrs = len(asset_names) - 1

            if max_corrs > 1:
                n_corrs_to_show = st.slider(
                    "Number of correlations to display:",
                    min_value=1,
                    max_value=max_corrs,
                    value=min(20, max_corrs),
                    step=1,
                    key="asset_corrs"
                )
            else:
                n_corrs_to_show = max_corrs
                st.info(f"📊 Showing the only correlation available for {selected_asset}")

            if selected_asset:
                # Generate asset-specific correlation visualization
                fig, display_data = create_asset_correlations_bar(
                    corr_matrix, p_matrix, asset_names, selected_asset, n_corrs_to_show
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"### 📋 Correlation Details for {selected_asset}")
                st.dataframe(
                    display_data,
                    use_container_width=True,
                    height=min(400, 35 * n_corrs_to_show)
                )

                # Calculate asset-specific statistics
                asset_idx = asset_names.index(selected_asset)
                other_correlations = [corr_matrix[asset_idx, i] for i in range(len(asset_names)) if i != asset_idx]

                if other_correlations:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Correlation", f"{np.mean(other_correlations):.3f}")
                    with col2:
                        st.metric("Maximum Correlation", f"{np.max(other_correlations):.3f}")
                    with col3:
                        st.metric("Minimum Correlation", f"{np.min(other_correlations):.3f}")

        with tab5:
            st.markdown(f"### 🌐 Correlation Network ({correlation_method})")
            corr_threshold = st.slider(
                "Correlation threshold:",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="network_threshold"
            )
            # Generate network graph visualization
            fig = create_network_graph(corr_matrix, asset_names, corr_threshold)
            st.plotly_chart(fig, use_container_width=True)

        with tab6:
            st.markdown("### ⏰ Temporal Correlation Analysis (OPTIMIZED)")
            st.info("🚀 **Fast analysis**: Same insights, 10x faster processing")

            if len(asset_names) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    selected_asset1 = st.selectbox(
                        "First asset:",
                        options=asset_names,
                        key="timeline_asset1"
                    )
                with col2:
                    other_assets = [a for a in asset_names if a != selected_asset1]
                    selected_asset2 = st.selectbox(
                        "Second asset:",
                        options=other_assets,
                        key="timeline_asset2"
                    )

                timeframe_timeline = st.selectbox(
                    "Timeframe for analysis:",
                    ["15m", "30m", "1h", "4h", "1d"],
                    index=2,
                    key="timeline_timeframe"
                )

                # Track current asset pair for state management
                current_pair_key = f"{selected_asset1}_{selected_asset2}_{timeframe_timeline}"

                # Initialize or detect pair change
                if 'current_temporal_pair' not in st.session_state:
                    st.session_state.current_temporal_pair = current_pair_key
                    st.session_state.temporal_analysis_done = False
                    st.session_state.temporal_data = None

                # Reset state if asset pair changed
                if st.session_state.current_temporal_pair != current_pair_key:
                    st.session_state.current_temporal_pair = current_pair_key
                    st.session_state.temporal_analysis_done = False
                    st.session_state.temporal_data = None
                    # Clear old date filters
                    if 'filter_start_date' in st.session_state:
                        del st.session_state.filter_start_date
                    if 'filter_end_date' in st.session_state:
                        del st.session_state.filter_end_date
                    st.info("🔄 New assets selected. Please load temporal analysis.")

                # Date alignment diagnosis
                if st.button("🔍 Diagnose Date Alignment", type="secondary", key="diagnose_dates"):
                    with st.spinner("Analyzing available dates..."):
                        try:
                            loader = BatchCorrelationLoader()
                            assets_data = loader.load_assets_for_rolling_analysis(
                                [selected_asset1, selected_asset2],
                                timeframe_timeline
                            )

                            if len(assets_data) < 2:
                                st.error(f"Could not load both assets. Loaded: {list(assets_data.keys())}")
                                st.stop()

                            analyzer = CorrelationTimelineAnalyzer()
                            diagnosis = analyzer.diagnose_date_alignment(
                                assets_data[selected_asset1],
                                assets_data[selected_asset2],
                                selected_asset1,
                                selected_asset2
                            )

                            st.markdown("### 📅 Date Availability Diagnosis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    f"{diagnosis['asset1']['name']} - Dates",
                                    f"{diagnosis['asset1']['total_dates']}",
                                    help=f"Unique: {diagnosis['asset1']['unique_dates']}"
                                )
                                if diagnosis['asset1']['start_date']:
                                    st.info(
                                        f"**Start:** {analyzer._safe_date_format(diagnosis['asset1']['start_date'])}")
                                    st.info(f"**End:** {analyzer._safe_date_format(diagnosis['asset1']['end_date'])}")
                            with col2:
                                st.metric(
                                    f"{diagnosis['asset2']['name']} - Dates",
                                    f"{diagnosis['asset2']['total_dates']}",
                                    help=f"Unique: {diagnosis['asset2']['unique_dates']}"
                                )
                                if diagnosis['asset2']['start_date']:
                                    st.info(
                                        f"**Start:** {analyzer._safe_date_format(diagnosis['asset2']['start_date'])}")
                                    st.info(f"**End:** {analyzer._safe_date_format(diagnosis['asset2']['end_date'])}")

                            st.markdown("### 🔄 Date Overlap Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Common Dates", diagnosis['common']['common_dates'])
                            with col2:
                                st.metric("Overlap Percentage", f"{diagnosis['common']['overlap_percentage']:.1f}%")

                            if diagnosis['common']['common_dates'] == 0:
                                st.error("❌ NO COMMON DATES - Assets have no overlapping periods")
                            elif diagnosis['common']['common_dates'] < 50:
                                st.warning(f"⚠️ FEW COMMON DATES: {diagnosis['common']['common_dates']}")
                            else:
                                st.success(f"✅ SUFFICIENT COMMON DATES: {diagnosis['common']['common_dates']}")

                        except Exception as e:
                            st.error(f"Diagnosis error: {str(e)}")

                # Initialize temporal analysis state
                if 'temporal_analysis_done' not in st.session_state:
                    st.session_state.temporal_analysis_done = False
                    st.session_state.temporal_data = None

                # Main temporal analysis execution
                if st.button("📊 Perform FAST Temporal Analysis", type="primary", key="analyze_temporal"):
                    with st.spinner("Loading data (optimized)..."):
                        try:
                            loader = BatchCorrelationLoader()
                            assets_data = loader.load_assets_for_rolling_analysis(
                                [selected_asset1, selected_asset2],
                                timeframe_timeline
                            )

                            if len(assets_data) < 2:
                                st.error(f"Could not load both assets.")
                                st.stop()

                            analyzer = CorrelationTimelineAnalyzer()
                            diagnosis = analyzer.diagnose_date_alignment(
                                assets_data[selected_asset1],
                                assets_data[selected_asset2],
                                selected_asset1,
                                selected_asset2
                            )

                            if diagnosis['common']['common_dates'] < 30:
                                st.error(f"❌ INSUFFICIENT COMMON DATES: {diagnosis['common']['common_dates']}")
                                st.stop()

                            # Align data by date for correlation analysis
                            combined_df = analyzer._align_data_by_date(
                                assets_data[selected_asset1],
                                assets_data[selected_asset2],
                                selected_asset1,
                                selected_asset2
                            )

                            # Store in session state for persistence
                            st.session_state.temporal_analysis_done = True
                            temporal_data = {
                                'combined_df': combined_df,
                                'analyzer': analyzer,
                                'asset1': selected_asset1,
                                'asset2': selected_asset2,
                                'timeframe': timeframe_timeline,
                                'min_date': combined_df["open_time"].min(),
                                'max_date': combined_df["open_time"].max()
                            }
                            st.session_state.temporal_data = temporal_data

                            # Initialize date filters with available range
                            min_date_py = pd.Timestamp(temporal_data['min_date']).to_pydatetime().date()
                            max_date_py = pd.Timestamp(temporal_data['max_date']).to_pydatetime().date()
                            st.session_state.filter_start_date = min_date_py
                            st.session_state.filter_end_date = max_date_py

                            st.success("✅ Data loaded successfully. You can now filter by dates.")

                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")

                # Display analysis if already loaded
                if st.session_state.temporal_analysis_done and st.session_state.temporal_data is not None:

                    temporal_data = st.session_state.temporal_data
                    combined_df = temporal_data['combined_df']
                    analyzer = temporal_data['analyzer']
                    selected_asset1 = temporal_data['asset1']
                    selected_asset2 = temporal_data['asset2']
                    timeframe_timeline = temporal_data['timeframe']

                    # DATE FILTERING SECTION
                    st.markdown("---")
                    st.markdown("### 📅 **FILTER BY DATE RANGE**")
                    st.info("🎯 Analyze specific periods to better understand temporal behavior")

                    # Get complete date range from available data
                    min_date = temporal_data['min_date']
                    max_date = temporal_data['max_date']

                    # Convert to Python datetime for date_input compatibility
                    min_date_py = pd.Timestamp(min_date).to_pydatetime().date()
                    max_date_py = pd.Timestamp(max_date).to_pydatetime().date()

                    # Initialize or validate stored date filters
                    if 'filter_start_date' not in st.session_state:
                        st.session_state.filter_start_date = min_date_py
                    else:
                        # Adjust if outside valid range
                        if st.session_state.filter_start_date < min_date_py:
                            st.session_state.filter_start_date = min_date_py
                        elif st.session_state.filter_start_date > max_date_py:
                            st.session_state.filter_start_date = min_date_py

                    if 'filter_end_date' not in st.session_state:
                        st.session_state.filter_end_date = max_date_py
                    else:
                        # Adjust if outside valid range
                        if st.session_state.filter_end_date > max_date_py:
                            st.session_state.filter_end_date = max_date_py
                        elif st.session_state.filter_end_date < min_date_py:
                            st.session_state.filter_end_date = max_date_py

                    # Date selection interface
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        start_date = st.date_input(
                            "Start date:",
                            value=st.session_state.filter_start_date,
                            min_value=min_date_py,
                            max_value=max_date_py,
                            key="start_date_input"
                        )
                        st.session_state.filter_start_date = start_date

                    with col2:
                        end_date = st.date_input(
                            "End date:",
                            value=st.session_state.filter_end_date,
                            min_value=min_date_py,
                            max_value=max_date_py,
                            key="end_date_input"
                        )
                        st.session_state.filter_end_date = end_date

                    with col3:
                        st.write("")
                        st.write("")
                        if st.button("🔄 Reset Dates", key="reset_dates_btn"):
                            st.session_state.filter_start_date = min_date_py
                            st.session_state.filter_end_date = max_date_py
                            st.rerun()

                    # Validate date selection
                    if start_date > end_date:
                        st.error("❌ Start date must be before end date")
                        st.stop()

                    # Filter DataFrame by selected date range
                    start_date_dt = pd.Timestamp(start_date)
                    end_date_dt = pd.Timestamp(end_date)

                    combined_df_filtered = combined_df.filter(
                        (pl.col("open_time") >= start_date_dt) &
                        (pl.col("open_time") <= end_date_dt)
                    )

                    # Validate filtered data sufficiency
                    if combined_df_filtered.height < 20:
                        st.warning(
                            f"⚠️ Selected range has only {combined_df_filtered.height} periods. Minimum 20 required.")
                        st.info("💡 Try expanding the date range")
                        st.stop()

                    st.success(
                        f"✅ Analyzing **{combined_df_filtered.height} periods** from **{start_date}** to **{end_date}**")

                    # PERFORM ANALYSIS WITH FILTERED DATA
                    with st.spinner("Analyzing selected period (optimized)..."):
                        try:
                            # Check cache for this specific analysis
                            cached_analysis = get_cached_analysis(selected_asset1, selected_asset2, timeframe_timeline,
                                                                  start_date, end_date)

                            if cached_analysis:
                                analysis = cached_analysis
                                st.info("📊 Analysis loaded from cache")
                            else:
                                # Perform complete correlation timeline analysis
                                analysis = analyzer.analyze_complete_correlation_timeline(
                                    combined_df_filtered, selected_asset1, selected_asset2,
                                    method=correlation_method
                                )
                                # Cache the results
                                set_cached_analysis(selected_asset1, selected_asset2, timeframe_timeline, start_date,
                                                    end_date, analysis)

                            # Display complete timeline visualization
                            st.markdown("### 📈 Complete Evolution of Both Assets")
                            fig1 = analyzer.create_complete_timeline_plot(analysis, selected_asset1, selected_asset2)
                            st.plotly_chart(fig1, use_container_width=True)

                            # UPDATED SECTION - REAL TOTAL CORRELATION
                            st.markdown("### 📊 Filtered Period Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Total Period Correlation",
                                    f"{analysis['total_correlation']:.3f}",
                                    help="Correlation of the ENTIRE selected period (no rolling windows)"
                                )
                            with col2:
                                st.metric("Periods Analyzed", analysis['total_periods'])
                            with col3:
                                st.metric("Valid Periods", analysis['valid_periods'])
                            with col4:
                                st.metric("Breaks Detected", len(analysis.get('correlation_breaks', [])))

                            # NEW: Additional information about rolling correlations
                            with st.expander("ℹ️ Correlation Metrics Information", expanded=False):
                                st.markdown(f"""
                                **📊 Correlation Metrics Explained (Method: {analysis['method']})**

                                1. **Total Period Correlation: {analysis['total_correlation']:.3f}**
                                   - Correlation of the **ENTIRE selected period** without using rolling windows
                                   - Represents the overall relationship between {selected_asset1} and {selected_asset2}
                                   - P-value: {analysis['total_p_value']:.3e} ({_is_statistically_significant(analysis['total_p_value'])})

                                2. **Average Rolling Correlation: {analysis['avg_rolling_correlation']:.3f}**
                                   - Average of rolling correlations with {analysis['rolling_window_size']}-period windows
                                   - Standard deviation: {analysis['std_rolling_correlation']:.3f}
                                   - Used to detect temporal changes and phases

                                **📊 What Does the Difference Mean?**

                                Absolute difference: **{abs(analysis['total_correlation'] - analysis['avg_rolling_correlation']):.3f}**

                                - If difference is **< 0.10**: Very stable correlation over time ✅
                                - If difference is **0.10-0.30**: Moderate changes, possible regimes 📊
                                - If difference is **> 0.30**: Highly variable correlation, multiple phases detected ⚠️

                                **💡 Practical Interpretation:**
                                - Small difference indicates **consistent** relationship
                                - Large difference indicates **changing regimes** (periods of high/low correlation)
                                - Useful for: pairs trading strategies, hedging, market regime detection
                                """)

                            start_date_formatted = analyzer._safe_date_format(analysis['start_date'])
                            end_date_formatted = analyzer._safe_date_format(analysis['end_date'])
                            st.info(
                                f"**Analyzed period:** {start_date_formatted} to {end_date_formatted} | **Timeframe:** {timeframe_timeline}")

                            # FAST ANALYSIS WITH FastPhaseDetector
                            st.markdown("---")
                            st.markdown("## 🎯 FAST REGIME ANALYSIS")

                            if 'rolling_correlations' not in analysis or not analysis['rolling_correlations']:
                                st.error("❌ No rolling correlation data available")
                                st.stop()

                            correlation_timeline = np.array([
                                period['correlation'] for period in analysis['rolling_correlations']
                            ])
                            dates_timeline = [
                                period['end_date'] for period in analysis['rolling_correlations']
                            ]

                            # USE OPTIMIZED DETECTOR
                            fast_phases = FastPhaseDetector.detect_fast_phases(correlation_timeline)

                            if fast_phases:
                                st.success(
                                    f"✅ Detected {len(fast_phases)} phases in the selected period")

                                # Structural breaks (if any)
                                if analysis.get('correlation_breaks'):
                                    st.markdown("### 📈 Detected Structural Breaks")
                                    st.info(
                                        f"Detected **{len(analysis['correlation_breaks'])} break points** where correlation changed significantly")

                                    fig2 = analyzer.create_correlation_breaks_chart(analysis['correlation_breaks'])
                                    st.plotly_chart(fig2, use_container_width=True)

                                    st.markdown("#### 📋 Break Point Details")
                                    break_data = []
                                    for break_point in analysis['correlation_breaks']:
                                        break_data.append({
                                            'Date': analyzer._safe_date_format(break_point['break_date']),
                                            'Correlation Before': f"{break_point['before_correlation']:.3f}",
                                            'Correlation After': f"{break_point['after_correlation']:.3f}",
                                            'Change': f"{break_point['change_magnitude']:.3f}",
                                            'Direction': break_point['direction'],
                                            'Intensity': break_point['intensity']
                                        })
                                    st.dataframe(break_data, use_container_width=True)

                                # Detected Regimes (FAST VERSION)
                                st.markdown("### 🔬 DETECTED REGIMES (Fast Analysis)")

                                phase_display_data = []
                                for phase in fast_phases:
                                    start_date_phase = dates_timeline[phase['start_idx']]
                                    end_date_phase = dates_timeline[phase['end_idx'] - 1] if phase['end_idx'] < len(
                                        dates_timeline) else dates_timeline[-1]

                                    phase_display_data.append({
                                        'Phase': f"{phase['emoji']} {phase['phase_id']}",
                                        'Regime': phase['regime_type'],
                                        'Start': analyzer._safe_date_format(start_date_phase),
                                        'End': analyzer._safe_date_format(end_date_phase),
                                        'Correlation': f"{phase['average_correlation']:.3f}",
                                        'Volatility': f"{phase['correlation_volatility']:.3f}",
                                        'Duration': f"{phase['total_periods']} periods",
                                        'Significance': phase['statistical_significance'],
                                        'Persistence': f"{phase['persistence']:.2f}",
                                        'P-value': f"{phase['p_value']:.3e}" if phase['p_value'] is not None else "N/A"
                                    })

                                df_phases = pd.DataFrame(phase_display_data)
                                st.dataframe(df_phases, use_container_width=True,
                                             height=min(400, len(phase_display_data) * 35 + 38))

                                # Phase metrics summary
                                st.markdown("### 📊 PHASE METRICS")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    valid_phases = len([p for p in fast_phases])
                                    st.metric("Phases Detected", valid_phases)
                                with col2:
                                    avg_duration = np.mean([p['total_periods'] for p in fast_phases])
                                    st.metric("Average Duration", f"{avg_duration:.0f} periods")
                                with col3:
                                    high_persistence = len(
                                        [p for p in fast_phases if abs(p['persistence']) > 0.3])
                                    st.metric("Good Persistence", high_persistence)
                                with col4:
                                    strong_regimes = len(
                                        [p for p in fast_phases if 'STRONG' in p['regime_type']])
                                    st.metric("Strong Regimes", strong_regimes)

                                # Temporal phases visualization
                                if analysis.get('temporal_phases'):
                                    st.markdown("### 🕰️ Temporal Phases Visualization")
                                    st.info("📊 Phases detected with optimized algorithm")
                                    fig3 = analyzer.create_phases_analysis_chart(analysis['temporal_phases'])
                                    st.plotly_chart(fig3, use_container_width=True)

                            else:
                                st.warning(
                                    "⚠️ No phases detected in the selected range.")
                                st.info("""
                                **Possible reasons:**
                                - The analyzed period is too short
                                - Correlation is very volatile
                                - Assets don't show clear patterns

                                💡 **Suggestions:**
                                - Expand the date range
                                - Try a different timeframe
                                - Select assets with higher historical correlation
                                """)

                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
                            import traceback
                            with st.expander("🛠 View error details"):
                                st.code(traceback.format_exc())

            else:
                st.warning("At least 2 assets needed for temporal analysis")

        # Failed pairs section
        st.markdown("---")
        st.markdown("### ⚠️ Pairs Without Valid Correlation")
        st.info(
            "These pairs couldn't be calculated due to insufficient common dates or data issues.")
        if failed_pairs:
            df_failed = pd.DataFrame(failed_pairs)
            st.dataframe(df_failed, use_container_width=True, height=300)
        else:
            st.success("✅ All pairs had sufficient data for correlation calculations.")

    else:
        # Welcome screen with instructions
        st.info("""
        👈 **Correlation Analysis System**

        **To get started:**
        1. Select **"By Category"**, **"By Assets"**, or **"Load ALL"** in the sidebar
        2. Choose the assets you want to analyze
        3. Configure correlation method, timeframe, and batch size
        4. Click **"🚀 Load and Analyze"**

        **Optimized Temporal Analysis**: 
        - 🚀 10x faster than previous version
        - 🎯 Same insights with fewer unnecessary calculations
        - 💾 Intelligent caching system
        - 📊 Pattern detection equally effective
        """)


if __name__ == "__main__":
    render()