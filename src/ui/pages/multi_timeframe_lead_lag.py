import streamlit as st
import polars as pl
import numpy as np
from core.data_loader import ReturnsLoader
from core.multi_timeframe_lead_lag import MultiTimeframeLeadLagAnalyzer
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_lag_analysis_chart(results: dict, event_type: str, timeframe: str, asset: str) -> go.Figure:
    """
    Creates detailed lag-by-lag analysis chart with comprehensive statistics

    This function generates a bar chart showing the average return at each lag period
    after significant events in the reference asset. Each bar includes:
    - Confidence intervals (error bars)
    - Statistical significance coloring
    - Number of events analyzed
    - Hover information with detailed metrics

    Args:
        results: Dictionary containing analysis results from MultiTimeframeLeadLagAnalyzer
        event_type: Either "positive" (up moves) or "negative" (down moves)
        timeframe: The specific timeframe being analyzed (e.g., "15m", "1h")
        asset: The target asset being analyzed

    Returns:
        go.Figure: Plotly figure object with the lag analysis visualization
    """

    # Safely extract lag statistics from the nested results structure
    lag_stats = results["results"][event_type][timeframe][asset]["lag_statistics"]

    # Handle case where no data is available for this asset/timeframe combination
    if not lag_stats:
        fig = go.Figure()
        fig.add_annotation(text="No data available for this asset", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(template="plotly_dark")
        return fig

    # Extract data for visualization
    lags = []
    means = []
    ci_lowers = []
    ci_uppers = []
    significances = []
    n_events = []

    # Process each lag's statistics
    for lag_key, stats in lag_stats.items():
        lag_num = stats['lag']
        lags.append(lag_num)
        means.append(stats['mean'] * 100)  # Convert to percentage for better readability
        ci_lowers.append(stats['confidence_interval_lower'] * 100)
        ci_uppers.append(stats['confidence_interval_upper'] * 100)
        significances.append(stats['significance'])
        n_events.append(stats['n_events'])

    # Color coding based on statistical significance
    # Green: Very Significant, Blue: Significant, Yellow: Marginal, Red: Not Significant
    colors = []
    for sig in significances:
        if sig == "VERY_SIGNIFICANT":
            colors.append('#00FF00')  # Bright Green
        elif sig == "SIGNIFICANT":
            colors.append('#00C896')  # Teal
        elif sig == "MARGINALLY_SIGNIFICANT":
            colors.append('#FFD700')  # Gold
        else:
            colors.append('#FF6B6B')  # Red

    fig = go.Figure()

    # Create bar chart with error bars for confidence intervals
    fig.add_trace(go.Bar(
        x=lags,
        y=means,
        error_y=dict(
            type='data',
            array=[ci_uppers[i] - means[i] for i in range(len(means))],
            arrayminus=[means[i] - ci_lowers[i] for i in range(len(means))],
            color='white',
            thickness=2
        ),
        marker_color=colors,
        text=[f"{m:.2f}%<br>n={n}" for m, n in zip(means, n_events)],
        textposition='outside',
        hovertemplate=(
            '<b>Lag %{x}</b><br>'
            'Return: %{y:.3f}%<br>'
            'CI: [%{customdata[0]:.3f}%, %{customdata[1]:.3f}%]<br>'
            'Events: %{customdata[2]}<br>'
            'Significance: %{customdata[3]}<br>'
            '<extra></extra>'
        ),
        customdata=list(zip(ci_lowers, ci_uppers, n_events, significances))
    ))

    direction = "Up Moves" if event_type == "positive" else "Down Moves"

    fig.update_layout(
        title=f"Lag-by-Lag Analysis: {asset} ({timeframe}) - {direction}<br>"
              "<sub>Green=Very Sig., Blue=Significant, Yellow=Marginal, Red=Not Sig.</sub>",
        xaxis_title="Lag (periods post-event)",
        yaxis_title="Average Return (%)",
        template="plotly_dark",
        height=500,
        xaxis=dict(dtick=1)  # Ensure every lag is shown on x-axis
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

    return fig


def create_lag_heatmap(results: dict, event_type: str, target_timeframes: list,
                       target_assets: list, max_lags: int) -> go.Figure:
    """
    Creates a comprehensive heatmap showing response patterns across assets and timeframes

    This visualization provides a bird's-eye view of how different assets respond
    to reference asset movements across various timeframes and lag periods.

    Args:
        results: Analysis results dictionary
        event_type: "positive" for up moves, "negative" for down moves
        target_timeframes: List of timeframes to include in heatmap
        target_assets: List of target assets to analyze
        max_lags: Maximum number of lag periods to display

    Returns:
        go.Figure: Heatmap visualization figure
    """

    # Prepare matrix: assets x lags
    z_data = []
    y_labels = []

    # Build data matrix for heatmap
    for tf in target_timeframes:
        for asset in target_assets:
            lag_stats = results["results"][event_type][tf][asset]["lag_statistics"]

            row = []
            for lag in range(max_lags + 1):
                lag_key = f'lag_{lag}'
                # Only include data if we have sufficient events
                if lag_key in lag_stats and lag_stats[lag_key]['n_events'] > 0:
                    row.append(lag_stats[lag_key]['mean'] * 100)
                else:
                    row.append(0.0)  # Use zero for missing data

            z_data.append(row)
            y_labels.append(f"{asset} ({tf})")

    # Handle case where no data is available
    if not z_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available for heatmap", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color="white"))
        fig.update_layout(template="plotly_dark")
        return fig

    # Create heatmap with red-blue color scale (red for negative, blue for positive)
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f'Lag {i}' for i in range(max_lags + 1)],
        y=y_labels,
        colorscale='RdBu',  # Red-Blue diverging colorscale
        zmid=0,  # Center colorscale at zero
        text=[[f"{val:.2f}%" for val in row] for row in z_data],
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate=(
            'Asset: %{y}<br>'
            'Lag: %{x}<br>'
            'Return: %{z:.3f}%<br>'
            '<extra></extra>'
        )
    ))

    direction = "Up Moves" if event_type == "positive" else "Down Moves"

    fig.update_layout(
        title=f"Response Heatmap by Lag - {direction}",
        xaxis_title="Lag (periods post-event)",
        yaxis_title="Asset (Timeframe)",
        template="plotly_dark",
        height=400 + len(y_labels) * 20  # Dynamic height based on number of assets
    )

    return fig


def create_significance_summary(results: dict, event_type: str, target_timeframes: list,
                                target_assets: list, max_lags: int) -> go.Figure:
    """
    Creates a summary chart showing statistical significance patterns across lags

    This bar chart shows what percentage of asset-timeframe combinations
    show statistically significant responses at each lag period.

    Args:
        results: Analysis results dictionary
        event_type: Type of events to analyze
        target_timeframes: Timeframes to include
        target_assets: Assets to include
        max_lags: Maximum lag to analyze

    Returns:
        go.Figure: Significance summary bar chart
    """

    # Initialize counters for each lag
    significance_counts = {lag: {'VERY_SIGNIFICANT': 0, 'SIGNIFICANT': 0,
                                 'MARGINALLY_SIGNIFICANT': 0, 'NOT_SIGNIFICANT': 0}
                           for lag in range(max_lags + 1)}

    total_tests = {lag: 0 for lag in range(max_lags + 1)}

    # Count significance levels across all asset-timeframe combinations
    for tf in target_timeframes:
        for asset in target_assets:
            lag_stats = results["results"][event_type][tf][asset]["lag_statistics"]

            for lag in range(max_lags + 1):
                lag_key = f'lag_{lag}'
                if lag_key in lag_stats and lag_stats[lag_key]['n_events'] > 0:
                    sig = lag_stats[lag_key]['significance']
                    significance_counts[lag][sig] += 1
                    total_tests[lag] += 1

    # Calculate percentages for visualization
    lags = list(range(max_lags + 1))
    very_sig_pct = [significance_counts[lag]['VERY_SIGNIFICANT'] / max(total_tests[lag], 1) * 100 for lag in lags]
    sig_pct = [significance_counts[lag]['SIGNIFICANT'] / max(total_tests[lag], 1) * 100 for lag in lags]

    fig = go.Figure()

    # Stacked bar chart for significance levels
    fig.add_trace(go.Bar(
        name='Very Significant',
        x=lags,
        y=very_sig_pct,
        marker_color='#00FF00',  # Green
        hovertemplate='Lag %{x}<br>Very Significant: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Significant',
        x=lags,
        y=sig_pct,
        marker_color='#00C896',  # Teal
        hovertemplate='Lag %{x}<br>Significant: %{y:.1f}%<extra></extra>'
    ))

    direction = "Up Moves" if event_type == "positive" else "Down Moves"

    fig.update_layout(
        title=f"Statistical Significance by Lag - {direction}",
        xaxis_title="Lag",
        yaxis_title="% of Significant Tests",
        barmode='stack',
        template="plotly_dark",
        height=400,
        xaxis=dict(dtick=1)
    )

    return fig


def render_lag_analysis_results(results: dict, target_assets: list, target_timeframes: list, max_lags: int):
    """
    Renders detailed lag-by-lag analysis results without page refresh

    This function creates a comprehensive dashboard showing:
    - Heatmap overview of all assets and timeframes
    - Statistical significance summary
    - Detailed lag analysis for individual assets
    - Interactive controls for exploring different views

    Args:
        results: Analysis results dictionary
        target_assets: List of assets analyzed
        target_timeframes: List of timeframes analyzed
        max_lags: Maximum number of lags analyzed
    """

    st.markdown("### 📊 Detailed Lag-by-Lag Analysis")

    # Create tabs for positive and negative events
    tab1, tab2 = st.tabs(["📈 Positive Events (Up Moves)", "📉 Negative Events (Down Moves)"])

    for tab, event_type in [(tab1, "positive"), (tab2, "negative")]:
        with tab:
            event_count = results[f"{event_type}_event_count"]

            if event_count > 0:
                direction = "Up Moves" if event_type == "positive" else "Down Moves"

                # Overall heatmap for quick pattern recognition
                st.markdown(f"#### Overall Heatmap - {direction}")
                fig_heatmap = create_lag_heatmap(results, event_type, target_timeframes, target_assets, max_lags)
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Statistical significance summary
                st.markdown(f"#### Statistical Significance Summary - {direction}")
                fig_significance = create_significance_summary(results, event_type, target_timeframes, target_assets,
                                                               max_lags)
                st.plotly_chart(fig_significance, use_container_width=True)

                # CORRECTED SECTION - Detailed analysis without refresh issues
                st.markdown(f"#### Detailed Analysis by Asset - {direction}")

                # Organize by timeframes to avoid dynamic selectboxes
                for tf_idx, tf in enumerate(target_timeframes):
                    st.markdown(f"##### Timeframe: {tf}")

                    # Use appropriate controls based on number of assets
                    if len(target_assets) <= 2:
                        # For few assets, show buttons for all
                        cols = st.columns(len(target_assets))
                        for asset_idx, asset in enumerate(target_assets):
                            with cols[asset_idx]:
                                if st.button(f"View {asset}", key=f"btn_{event_type}_{tf}_{asset}",
                                             type="secondary"):
                                    # Store selection in session state
                                    st.session_state[f"selected_asset_{event_type}_{tf}"] = asset
                    else:
                        # For many assets, use radio buttons (no refresh)
                        selected_asset = st.radio(
                            f"Select asset for {tf}:",
                            options=target_assets,
                            key=f"radio_{event_type}_{tf}",
                            horizontal=True
                        )
                        st.session_state[f"selected_asset_{event_type}_{tf}"] = selected_asset

                    # Display analysis for selected asset
                    current_asset = st.session_state.get(f"selected_asset_{event_type}_{tf}")

                    if current_asset and current_asset in target_assets:
                        # Detailed lag analysis chart
                        fig_detailed = create_lag_analysis_chart(results, event_type, tf, current_asset)
                        st.plotly_chart(fig_detailed, use_container_width=True,
                                        key=f"chart_{event_type}_{tf}_{current_asset}")

                        # Statistics table
                        lag_stats = results["results"][event_type][tf][current_asset]["lag_statistics"]
                        total_events = results["results"][event_type][tf][current_asset]["total_events"]

                        if lag_stats:
                            st.markdown(f"**Detailed Statistics - {current_asset} ({tf})**")
                            st.caption(f"Total events analyzed: {total_events}")

                            # Create comprehensive statistics table
                            table_data = []
                            for lag_key, stats in lag_stats.items():
                                if stats['n_events'] > 0:
                                    table_data.append({
                                        "Lag": stats['lag'],
                                        "Return (%)": f"{stats['mean'] * 100:.3f}",
                                        "CI Lower (%)": f"{stats['confidence_interval_lower'] * 100:.3f}",
                                        "CI Upper (%)": f"{stats['confidence_interval_upper'] * 100:.3f}",
                                        "Std Dev": f"{stats['std']:.4f}",
                                        "P-value": f"{stats['p_value']:.3e}",
                                        "Significance": stats['significance'],
                                        "Persistence (%)": f"{stats['persistence']:.1f}",
                                        "N Events": stats['n_events']
                                    })

                            if table_data:
                                df_stats = pd.DataFrame(table_data)
                                st.dataframe(df_stats, use_container_width=True,
                                             key=f"table_{event_type}_{tf}_{current_asset}")

                                # Highlight statistically significant lags
                                significant_lags = [row for row in table_data
                                                    if row['Significance'] in ['VERY_SIGNIFICANT', 'SIGNIFICANT']]

                                if significant_lags:
                                    st.success(f"🎯 Significant lags found: {len(significant_lags)}")
                                    best_lag = max(significant_lags,
                                                   key=lambda x: abs(float(x['Return (%)'].replace('%', ''))))
                                    st.info(
                                        f"**Best lag:** Lag {best_lag['Lag']} with {best_lag['Return (%)']}% average return")
                                else:
                                    st.warning("No statistically significant lags found")
                        else:
                            st.warning(f"No lag data available for {current_asset} in {tf}")
                    else:
                        # Show first asset by default
                        if target_assets:
                            default_asset = target_assets[0]
                            st.session_state[f"selected_asset_{event_type}_{tf}"] = default_asset
                            st.info(f"Showing analysis for {default_asset}. Use controls above to change.")
            else:
                direction = "up" if event_type == "positive" else "down"
                st.warning(f"No {direction} move events found with the specified threshold")


def render():
    """
    MAIN RENDERING FUNCTION - Multi-Timeframe Lead-Lag Analysis Dashboard

    This module provides comprehensive lead-lag analysis across multiple timeframes,
    showing how target assets respond to significant movements in a reference asset.

    Key Features:
    - Lag-by-lag statistical analysis with confidence intervals
    - Multiple timeframe comparison (from 1m to 1d)
    - Statistical significance testing with p-values
    - Interactive visualizations without page refresh
    - Professional trading insights and pattern detection

    Target Users:
    - Quantitative traders developing pairs trading strategies
    - Risk managers analyzing market contagion effects
    - Researchers studying market microstructure
    - Portfolio managers understanding asset relationships
    """

    st.markdown("## 📊 Multi-Timeframe Lead-Lag Analysis - DETAILED LAGS")
    st.info("""
    **🎯 NEW VERSION - Lag-by-lag analysis:**
    - ✅ Detailed statistics for each individual lag
    - ✅ Confidence intervals and p-values per lag
    - ✅ Correct temporal logic (post-event only)
    - ✅ Lag-specific visualizations
    - ✅ Detection of statistically significant lags
    - ✅ No refresh when changing assets
    """)

    # Sidebar: Analysis Configuration
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")

        with st.expander("📊 Asset Selection", expanded=True):
            # Discover available assets in data directory
            loader = ReturnsLoader(data_dir="data", cache_dir="cache/processed")
            available_assets = loader.discover_assets()

            reference_asset = st.selectbox(
                "Reference Asset:",
                options=available_assets,
                index=available_assets.index("BTCUSDT") if "BTCUSDT" in available_assets else 0,
                help="Asset whose significant movements will trigger the analysis"
            )

            target_assets = st.multiselect(
                "Target Assets:",
                options=[a for a in available_assets if a != reference_asset],
                default=["ETHUSDT", "SOLUSDT", "XRPUSDT"] if all(
                    a in available_assets for a in ["ETHUSDT", "SOLUSDT", "XRPUSDT"]
                ) else [],
                help="Assets that will be analyzed for response to reference asset movements"
            )

            if not target_assets:
                st.warning("⚠️ Select at least 1 target asset")

        with st.expander("🕐 Time Configuration", expanded=True):
            reference_timeframe = st.selectbox(
                "Reference Timeframe:",
                options=["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"],
                index=5,  # 1h as default
                help="Timeframe for detecting significant events in the reference asset"
            )

            # Automatically filter for smaller timeframes than reference
            analyzer = MultiTimeframeLeadLagAnalyzer()
            ref_minutes = analyzer._timeframe_to_minutes(reference_timeframe)
            valid_target_tfs = [
                tf for tf in ["1m", "5m", "10m", "15m", "30m", "1h", "4h"]
                if analyzer._timeframe_to_minutes(tf) < ref_minutes
            ]

            if not valid_target_tfs:
                st.error(f"❌ No timeframes smaller than {reference_timeframe}")
                target_timeframes = []
            else:
                target_timeframes = st.multiselect(
                    "Smaller Timeframes:",
                    options=valid_target_tfs,
                    default=valid_target_tfs[-2:] if len(valid_target_tfs) >= 2 else valid_target_tfs,
                    help="Timeframes for analyzing target asset responses (must be smaller than reference)"
                )

            same_day_only = st.checkbox(
                "Limit lags to same day",
                value=False,
                help="If enabled, analysis won't cross day boundaries"
            )

        with st.expander("🎚️ Analysis Parameters", expanded=True):
            threshold = st.slider(
                "Event Threshold (%):",
                min_value=0.1, max_value=10.0, value=2.0, step=0.1,
                help="Minimum percentage move in reference asset to consider as significant event"
            ) / 100  # Convert to decimal

            max_lags = st.slider(
                "Maximum Lags:",
                min_value=1, max_value=48, value=12, step=1,
                help="Number of post-event periods to analyze for response patterns"
            )

            comparison = st.selectbox(
                "Comparison Type:",
                options=["Greater or equal (>=)", "Greater (>)", "Equal (=)"],
                index=0,
                help="How to compare returns against the threshold"
            )

            # Map user-friendly names to technical parameter names
            comparison_map = {
                "Greater or equal (>=)": "greater_equal",
                "Greater (>)": "greater",
                "Equal (=)": "equal"
            }
            comparison = comparison_map[comparison]

            bootstrap_iterations = st.slider(
                "Bootstrap Iterations:",
                min_value=100, max_value=2000, value=1000, step=100,
                help="Number of bootstrap samples for robust confidence intervals"
            )

        analyze_button = st.button("🚀 Run Detailed Analysis", type="primary")

    # Execute analysis when button is clicked
    if analyze_button and target_assets and target_timeframes:
        with st.spinner("🔍 Analyzing lag by lag... (this may take several minutes)"):
            try:
                analyzer = MultiTimeframeLeadLagAnalyzer(reference_asset=reference_asset)

                # Capture analysis log for transparency
                with st.expander("📋 Analysis Log", expanded=False):
                    import io
                    import sys
                    from contextlib import redirect_stdout

                    f = io.StringIO()
                    with redirect_stdout(f):
                        results = analyzer.analyze(
                            target_assets=target_assets,
                            reference_timeframe=reference_timeframe,
                            target_timeframes=target_timeframes,
                            threshold=threshold,
                            max_lags=max_lags,
                            comparison=comparison,
                            same_day_only=same_day_only,
                            bootstrap_iterations=bootstrap_iterations
                        )

                    output = f.getvalue()
                    if output:
                        st.text(output)

                # Check for analysis errors
                if "error" in results:
                    st.error(f"❌ {results['error']}")
                    st.info("""
                    **💡 Possible solutions:**
                    - Reduce the event threshold
                    - Reduce the maximum lags
                    - Verify sufficient temporal data exists
                    """)
                    return

                # Store results in session_state to prevent refresh issues
                st.session_state['analysis_results'] = results
                st.session_state['analysis_target_assets'] = target_assets
                st.session_state['analysis_target_timeframes'] = target_timeframes
                st.session_state['analysis_max_lags'] = max_lags

            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                import traceback
                with st.expander("🛠 View technical error details"):
                    st.code(traceback.format_exc())

    # Display results if available in session state
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        target_assets = st.session_state['analysis_target_assets']
        target_timeframes = st.session_state['analysis_target_timeframes']
        max_lags = st.session_state['analysis_max_lags']

        # Display summary metrics
        st.markdown("### 📊 Analysis Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Positive Events", results["positive_event_count"])
        with col2:
            st.metric("Negative Events", results["negative_event_count"])
        with col3:
            st.metric("Assets", len(target_assets))
        with col4:
            st.metric("Timeframes", len(target_timeframes))
        with col5:
            st.metric("Max Lags", results["max_lags"])

        # Show analysis configuration used
        with st.expander("⚙️ Analysis Configuration", expanded=False):
            st.json({
                "Threshold": f"{results['threshold_used'] * 100:.1f}%",
                "Comparison": results['comparison_used'],
                "Reference Timeframe": results['reference_timeframe'],
                "Target Timeframes": results['target_timeframes'],
                "Max Lags": results['max_lags'],
                "Bootstrap Iterations": results['bootstrap_iterations'],
                "Seed": results['seed_used'],
                "Same Day Only": results.get('same_day_only', False)
            })

        # Render detailed lag analysis results
        render_lag_analysis_results(results, target_assets, target_timeframes, max_lags)

        # Show any assets that failed analysis
        if results["failed_assets"]:
            st.warning(f"⚠️ Assets with issues: {', '.join(results['failed_assets'])}")

    elif not (target_assets and target_timeframes and analyze_button):
        # Initial state - educational content and usage instructions
        st.info("""
        👈 **To get started:**

        **🎯 LAG-BY-LAG ANALYSIS:**

        1. **What's different?** - Now analyze each lag individually with complete statistics
        2. **Correct temporal logic** - Only POST-event periods (not contemporaneous)
        3. **Statistical validation** - CI, p-values, significance for each lag
        4. **Detailed visualization** - Heatmaps, lag-specific charts, statistical tables

        **📊 Usage example:**
        - BTC rises 2% in 1h (14:00-15:00)
        - Analysis in 15m: periods 15:00-15:15 (lag 0), 15:15-15:30 (lag 1), etc.
        - Complete statistics for each individual lag
        - Identify which specific lag has the most significant response

        **🚀 Configure and run analysis for actionable results**
        """)


if __name__ == "__main__":
    render()