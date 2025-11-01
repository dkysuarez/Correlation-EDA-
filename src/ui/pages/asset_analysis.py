import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
import psutil
import os
import gc
from core.data_loader import BatchCorrelationLoader
from core.asset_metadata import AssetMetadataManager
from core.correlation_calculator import CorrelationCalculator
from utils.visualization import (
    create_heatmap,
    create_top_pairs_bar,
    create_distribution_histogram,
    create_asset_correlations_bar,
    create_network_graph
)


def render():
    """
    MAIN RENDERING FUNCTION - Individual Asset Analysis Dashboard
    This page allows users to select one asset and analyze its correlation patterns with ALL other available assets
    """
    st.markdown("## 🔍 Individual Asset Analysis")
    st.markdown("Select an asset to see **ALL its correlation pairs** across the entire market")

    # Configuration panel with two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Load available assets from metadata manager
        metadata_mgr = AssetMetadataManager()
        available_assets = metadata_mgr.available_assets

        selected_asset = st.selectbox(
            "Select Asset to Analyze:",
            options=available_assets,
            index=0 if available_assets else None,
            help="Choose the asset you want to analyze against all others"
        )

    with col2:
        # Correlation method selection
        correlation_method = st.selectbox(
            "Correlation Method:",
            ["Spearman", "Pearson"],
            index=0,
            help="Spearman: Rank-based (non-linear relationships), Pearson: Linear relationships"
        )

        # Timeframe selection for analysis
        timeframe = st.selectbox(
            "Analysis Timeframe:",
            ["15m", "30m", "1h", "4h", "1d"],
            index=2,
            help="Time interval for calculating returns and correlations"
        )

    # Advanced processing configuration section
    st.markdown("### ⚙️ Advanced Processing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Performance optimization toggle
        max_performance = st.checkbox(
            "🚀 Maximum Performance Mode",
            value=True,
            help="Automatically adjusts batch size based on system resources for optimal speed"
        )

    with col2:
        if not max_performance:
            # Manual batch size control
            batch_size = st.slider(
                "Processing Batch Size:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of assets to process per batch. Higher = faster but more memory usage"
            )
        else:
            # Automatic batch size calculation based on system resources
            available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
            cpu_count = os.cpu_count() or 4  # Fallback to 4 cores if detection fails

            # Optimized formula for individual asset analysis
            # Balances memory usage with CPU utilization
            batch_size = min(int(available_memory * 150 + cpu_count * 75), 1000)

            # Display system information and auto-configuration
            st.info(f"""
            **Automatic Configuration:**
            - 🧠 Available RAM: {available_memory:.1f} GB
            - ⚡ CPU Cores: {cpu_count}
            - 📦 Optimal Batch Size: {batch_size} assets
            """)

    # Main analysis execution button
    if st.button("🚀 Analyze ALL Correlation Pairs", type="primary") and selected_asset:
        with st.spinner(f"📊 Loading data and calculating ALL correlations for {selected_asset}..."):
            try:
                # Display execution configuration for transparency
                st.info(f"""
                **Execution Configuration:**
                - 🎯 Target Asset: {selected_asset}
                - 📊 Correlation Method: {correlation_method}
                - ⏰ Timeframe: {timeframe}
                - 📦 Batch Size: {batch_size}
                - 🚀 Performance Mode: {'Maximum Performance' if max_performance else 'Custom'}
                """)

                # Initialize data loader and prepare asset list
                loader = BatchCorrelationLoader()
                all_assets = [selected_asset] + [asset for asset in available_assets if asset != selected_asset]

                # Progress tracking setup
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Load all asset data
                status_text.text("📥 Loading data for all assets...")
                progress_bar.progress(20)

                # Load asset data in batches to manage memory usage
                batch_results = loader.load_assets_in_batches(
                    all_assets,
                    batch_size=batch_size,
                    timeframe=timeframe
                )

                # Step 2: Calculate correlation matrix
                status_text.text("📊 Calculating complete correlation matrix...")
                progress_bar.progress(60)

                # Verify we have enough assets for meaningful analysis
                if len(batch_results['loaded_assets']) >= 2:
                    # Calculate correlation matrix using the same method as correlations.py
                    corr_matrix, p_matrix, asset_names, failed_pairs = loader.calculate_correlations(
                        batch_results,
                        correlation_method
                    )

                    progress_bar.progress(90)

                    # Critical validation: Ensure selected asset exists in results
                    if selected_asset not in asset_names:
                        st.error(f"❌ Asset {selected_asset} doesn't have sufficient data for analysis")
                        st.info("💡 Try selecting a different asset or changing the timeframe")
                        st.stop()

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")

                    # Display information about failed pairs (assets without enough common data)
                    if failed_pairs:
                        st.warning(f"⚠️ {len(failed_pairs)} pairs skipped due to insufficient common data")
                        with st.expander("🔍 View failed pairs details"):
                            df_failed = pd.DataFrame(failed_pairs)
                            st.dataframe(df_failed, use_container_width=True)

                    # Store results in session state for persistence across interactions
                    st.session_state.asset_analysis_data = {
                        'corr_matrix': corr_matrix,
                        'p_matrix': p_matrix,
                        'asset_names': asset_names,
                        'selected_asset': selected_asset,
                        'correlation_method': correlation_method,
                        'timeframe': timeframe,
                        'total_assets_processed': len(asset_names),
                        'failed_pairs': failed_pairs
                    }

                    # Clean up memory to prevent leaks in long sessions
                    gc.collect()

                else:
                    st.error("❌ Not enough assets with valid data for comparison")
                    st.info("💡 Try a different timeframe or verify assets have sufficient data")

            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                import traceback
                with st.expander("🐛 View technical error details"):
                    st.code(traceback.format_exc())
                st.info("""
                💡 Troubleshooting suggestions:
                - Reduce batch size for lower memory usage
                - Try a different timeframe
                - Select a different asset
                - Check data availability for selected assets
                """)

    # Display results if analysis has been completed
    if st.session_state.get('asset_analysis_data'):
        data = st.session_state.asset_analysis_data
        corr_matrix = data['corr_matrix']
        p_matrix = data['p_matrix']
        asset_names = data['asset_names']
        selected_asset = data['selected_asset']
        correlation_method = data['correlation_method']
        timeframe = data['timeframe']
        total_assets = data['total_assets_processed']
        failed_pairs = data.get('failed_pairs', [])

        # Find the index of the selected asset in the results
        if selected_asset in asset_names:
            asset_idx = asset_names.index(selected_asset)

            st.markdown("---")
            st.markdown(f"## 📊 Results for **{selected_asset}**")

            # Extract all correlations for the selected asset
            other_correlations = []
            significant_pairs = []

            # Collect correlation data and identify statistically significant pairs
            for i in range(len(asset_names)):
                if i != asset_idx:  # Skip self-correlation
                    corr_value = corr_matrix[asset_idx, i]
                    p_value = p_matrix[asset_idx, i]
                    other_correlations.append(corr_value)

                    # Identify statistically significant correlations (95% confidence)
                    if p_value < 0.05:
                        significant_pairs.append({
                            'asset': asset_names[i],
                            'correlation': corr_value,
                            'p_value': p_value
                        })

            # Display key metrics dashboard
            st.markdown("### 📈 Analysis Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Assets Processed", total_assets)
            with col2:
                st.metric("Pairs Analyzed", len(other_correlations))
            with col3:
                st.metric("Significant Pairs", len(significant_pairs))
            with col4:
                if other_correlations:
                    st.metric("Average Correlation", f"{np.mean(other_correlations):.3f}")
                else:
                    st.metric("Average Correlation", "N/A")
            with col5:
                if other_correlations:
                    st.metric("Maximum Correlation", f"{np.max(other_correlations):.3f}")
                else:
                    st.metric("Maximum Correlation", "N/A")

            # Tabbed interface for different visualization types
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Complete Table", "📊 Top Correlations", "📈 Distribution Analysis",
                "🔥 Significant Pairs", "🌐 Correlation Network"
            ])

            with tab1:
                st.markdown(f"### 📋 All Correlation Pairs for {selected_asset} ({len(other_correlations)} pairs)")

                # Build comprehensive data table with all correlation pairs
                pairs_data = []
                for i, asset in enumerate(asset_names):
                    if asset != selected_asset:
                        corr_value = corr_matrix[asset_idx, asset_names.index(asset)]
                        p_value = p_matrix[asset_idx, asset_names.index(asset)]
                        # Significance indicators for quick visual assessment
                        significance = "✅" if p_value < 0.05 else "⚠️" if p_value < 0.1 else "❌"

                        pairs_data.append({
                            'Asset': asset,
                            'Correlation': corr_value,
                            'P-Value': p_value,
                            'Significance': significance,
                            'Interpretation': _get_correlation_interpretation(corr_value),
                            'Strength': _get_correlation_intensity(corr_value)
                        })

                # Display sorted and formatted correlation table
                if pairs_data:
                    df = pd.DataFrame(pairs_data)
                    # Sort by absolute correlation value (strongest relationships first)
                    df_sorted = df.sort_values('Correlation', ascending=False, key=abs)

                    # Format numbers for better readability
                    df_display = df_sorted.copy()
                    df_display['Correlation'] = df_display['Correlation'].apply(lambda x: f"{x:.4f}")
                    df_display['P-Value'] = df_display['P-Value'].apply(lambda x: f"{x:.4e}")

                    st.dataframe(df_display, use_container_width=True, height=600)

                    # CSV download functionality for further analysis
                    csv = df_sorted.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV with all pairs",
                        data=csv,
                        file_name=f"correlations_{selected_asset}_{timeframe}_{correlation_method}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No valid pairs to display")

            with tab2:
                st.markdown(f"### 📊 Top Correlations for {selected_asset}")

                if other_correlations:
                    max_pairs_asset = len(other_correlations)

                    # Safe slider configuration with validation
                    if max_pairs_asset > 1:
                        n_pairs_to_show = st.slider(
                            "Number of pairs to display in chart:",
                            min_value=1,
                            max_value=min(50, max_pairs_asset),
                            value=min(25, max_pairs_asset),
                            step=1,
                            key="top_pairs_slider"
                        )
                    else:
                        n_pairs_to_show = max_pairs_asset
                        st.info("📊 Displaying the only available pair")

                    # Generate bar chart visualization
                    fig, display_data = create_asset_correlations_bar(
                        corr_matrix, p_matrix, asset_names, selected_asset, n_pairs_to_show
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display detailed table for top pairs
                    st.markdown(f"#### 📋 Top {n_pairs_to_show} Pairs Detail")
                    st.dataframe(display_data, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for correlation visualization")

            with tab3:
                st.markdown(f"### 📈 Correlation Distribution Analysis")

                if len(other_correlations) > 0:
                    # Create distribution histogram
                    fig = create_distribution_histogram(
                        np.array(other_correlations),
                        f"Correlation Distribution - {selected_asset} ({correlation_method})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed distribution statistics
                    st.markdown("#### 📊 Distribution Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        positive = len([c for c in other_correlations if c > 0.1])
                        st.metric("% Positive Correlations", f"{(positive / len(other_correlations)) * 100:.1f}%")
                    with col2:
                        negative = len([c for c in other_correlations if c < -0.1])
                        st.metric("% Negative Correlations", f"{(negative / len(other_correlations)) * 100:.1f}%")
                    with col3:
                        strong_pos = len([c for c in other_correlations if c > 0.5])
                        st.metric("% Strong Positive", f"{(strong_pos / len(other_correlations)) * 100:.1f}%")
                    with col4:
                        strong_neg = len([c for c in other_correlations if c < -0.5])
                        st.metric("% Strong Negative", f"{(strong_neg / len(other_correlations)) * 100:.1f}%")
                else:
                    st.warning("⚠️ Insufficient data for distribution analysis")

            with tab4:
                st.markdown(f"### 🔥 Statistically Significant Pairs (p < 0.05)")

                if significant_pairs:
                    st.success(f"✅ **{len(significant_pairs)} statistically significant pairs found**")

                    # Sort by correlation strength (absolute value)
                    significant_pairs_sorted = sorted(significant_pairs,
                                                      key=lambda x: abs(x['correlation']),
                                                      reverse=True)

                    # Build significant pairs table
                    sig_data = []
                    for pair in significant_pairs_sorted:
                        sig_data.append({
                            'Asset': pair['asset'],
                            'Correlation': pair['correlation'],
                            'P-Value': pair['p_value'],
                            'Interpretation': _get_correlation_interpretation(pair['correlation']),
                            'Strength': _get_correlation_intensity(pair['correlation'])
                        })

                    df_sig = pd.DataFrame(sig_data)
                    df_sig_display = df_sig.copy()
                    df_sig_display['Correlation'] = df_sig_display['Correlation'].apply(lambda x: f"{x:.4f}")
                    df_sig_display['P-Value'] = df_sig_display['P-Value'].apply(lambda x: f"{x:.4e}")

                    st.dataframe(df_sig_display, use_container_width=True)

                    # Download functionality for significant pairs
                    csv_sig = df_sig.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV of significant pairs",
                        data=csv_sig,
                        file_name=f"significant_pairs_{selected_asset}_{timeframe}_{correlation_method}.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning("⚠️ No statistically significant pairs found (p < 0.05)")
                    st.info("""
                    **This could indicate:**
                    - Observed correlations may be due to random chance
                    - More temporal data is needed for statistical power
                    - The asset has low correlation with others in the current market

                    💡 Try changing the timeframe or selecting a different asset
                    """)

            with tab5:
                st.markdown(f"### 🌐 Correlation Network Visualization")

                if len(other_correlations) > 0:
                    # Configurable correlation threshold for network connections
                    corr_threshold = st.slider(
                        "Correlation threshold for network connections:",
                        min_value=0.3,
                        max_value=0.8,
                        value=0.5,
                        step=0.1,
                        key="network_threshold"
                    )

                    # Generate network graph visualization
                    fig = create_network_graph(corr_matrix, asset_names, corr_threshold)
                    st.plotly_chart(fig, use_container_width=True)

                    # Count strong connections based on threshold
                    strong_connections = 0
                    for i in range(len(asset_names)):
                        if i != asset_idx and abs(corr_matrix[asset_idx, i]) >= corr_threshold:
                            strong_connections += 1

                    st.info(
                        f"**{strong_connections} strong connections** (|corr| ≥ {corr_threshold}) with {selected_asset}")
                else:
                    st.warning("⚠️ Insufficient data for network visualization")

        else:
            st.error(f"❌ Asset {selected_asset} not found in processed results")
            st.info("Try reloading the analysis data")

    else:
        # Welcome screen with detailed instructions
        st.info("""
        ## 🔍 **Individual Asset Analysis**

        **Analyze one specific asset against ALL other available assets in the market**

        ### ✨ Key Features:

        **📊 Comprehensive Analysis:**
        - Calculate correlations with hundreds of assets simultaneously
        - Identify statistically significant relationships
        - Discover strong positive and negative correlations

        **🎯 Key Metrics:**
        - Average correlation across all pairs
        - Top correlations by absolute strength
        - Correlation distribution analysis
        - P-values for statistical validation

        **📈 Multiple Visualizations:**
        - Sortable and downloadable complete table
        - Top correlations bar charts
        - Distribution histograms
        - Network graphs of strong connections

        ### 🚀 Getting Started:
        1. Select the **asset** you want to analyze
        2. Configure **correlation method** and **timeframe**
        3. Adjust **batch size** (or use Maximum Performance mode)
        4. Click **"🚀 Analyze ALL Correlation Pairs"**

        **💡 Pro Tip:** Use Spearman for non-linear relationships and Pearson for linear relationships
        """)


def _get_correlation_interpretation(corr_value: float) -> str:
    """
    Provides human-readable interpretation of correlation values

    Args:
        corr_value: Correlation coefficient between -1 and 1

    Returns:
        str: Descriptive interpretation of correlation strength and direction
    """
    abs_corr = abs(corr_value)

    if abs_corr > 0.7:
        direction = "Positive" if corr_value > 0 else "Negative"
        return f"Strong {direction}"
    elif abs_corr > 0.5:
        direction = "Positive" if corr_value > 0 else "Negative"
        return f"Moderate {direction}"
    elif abs_corr > 0.3:
        direction = "Positive" if corr_value > 0 else "Negative"
        return f"Weak {direction}"
    elif abs_corr > 0.1:
        direction = "Positive" if corr_value > 0 else "Negative"
        return f"Very Weak {direction}"
    else:
        return "Neutral/Insignificant"


def _get_correlation_intensity(corr_value: float) -> str:
    """
    Classifies correlation intensity with visual emoji indicators

    Args:
        corr_value: Correlation coefficient between -1 and 1

    Returns:
        str: Intensity classification with emojis for quick visual assessment
    """
    abs_corr = abs(corr_value)

    if abs_corr > 0.7:
        return "🔥 Very Strong"
    elif abs_corr > 0.5:
        return "💪 Strong"
    elif abs_corr > 0.3:
        return "📊 Moderate"
    elif abs_corr > 0.1:
        return "📈 Weak"
    else:
        return "📉 Very Weak"