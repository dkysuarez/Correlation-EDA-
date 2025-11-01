import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
from core.data_loader import BatchCorrelationLoader
from ui.components import render_sidebar
from utils.lead_lag_analyzer import LeadLagAnalyzer
from utils.visualization import plot_cross_correlation, plot_granger_results, create_lead_lag_summary


def render():
    """
    MAIN RENDERING FUNCTION - Lead-Lag Relationship Analysis Dashboard

    This module analyzes temporal relationships between financial assets to determine:
    - Which assets lead (move first) and which lag (follow later)
    - Optimal time delays for maximum correlation
    - Statistical causality using Granger causality tests
    - Trading insights for momentum and pairs trading strategies

    Key Use Cases:
    - Pairs trading: Identify leader-follower relationships
    - Momentum strategies: Find assets that consistently lead others
    - Risk management: Understand propagation of market movements
    - Market microstructure: Study information flow between assets
    """

    st.markdown("## 🕒 Lead-Lag Relationship Analysis")
    st.markdown("Analyze which assets lead price movements and which follow")

    # Reuse sidebar for asset selection to maintain consistency across the application
    selected_assets, correlation_method, timeframe, batch_size, load_button = render_sidebar()

    # Session state management for lead-lag analysis results
    if 'lead_lag_results' not in st.session_state:
        st.session_state.lead_lag_results = None
        st.session_state.lead_lag_assets_loaded = False

    # Load data when button is pressed and sufficient assets are selected
    if load_button and len(selected_assets) >= 2:
        with st.spinner("🔄 Loading data for lead-lag analysis..."):
            try:
                # Initialize data loader and load assets for temporal analysis
                loader = BatchCorrelationLoader()

                # Load asset data specifically formatted for session-based analysis
                # This ensures proper time alignment and data quality
                assets_data = loader.load_assets_for_session_analysis(selected_assets, timeframe)

                # Validate we have enough assets with valid data
                if len(assets_data) >= 2:
                    st.session_state.assets_data = assets_data
                    st.session_state.lead_lag_assets_loaded = True
                    st.success(f"✅ {len(assets_data)} assets loaded for analysis")
                else:
                    st.error("Not enough assets with valid data for analysis")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Display analysis interface if data is successfully loaded
    if st.session_state.get('lead_lag_assets_loaded', False):
        assets_data = st.session_state.assets_data
        asset_names = list(assets_data.keys())

        st.markdown("---")
        st.markdown("### 🔧 Analysis Configuration")

        # Asset pair selection in two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            asset1 = st.selectbox(
                "First Asset:",
                options=asset_names,
                key="lead_lag_asset1",
                help="Select the first asset for lead-lag comparison"
            )

        with col2:
            # Filter out the first asset to prevent self-comparison
            other_assets = [a for a in asset_names if a != asset1]
            asset2 = st.selectbox(
                "Second Asset:",
                options=other_assets,
                key="lead_lag_asset2",
                help="Select the second asset for lead-lag comparison"
            )

        # Analysis method selection
        st.markdown("#### 📊 Analysis Method")

        method = st.radio(
            "Select Analysis Method:",
            ["📊 Cross-Correlation", "🧠 Granger Causality"],
            horizontal=True,
            help="""Cross-Correlation: Finds optimal time lag for maximum correlation between assets.
                   Granger Causality: Tests if past values of one asset help predict another asset."""
        )

        # Method-specific configuration
        if method == "📊 Cross-Correlation":
            max_lag = st.slider(
                "Maximum Lag to Test:",
                min_value=1,
                max_value=168,
                value=24,
                help="Maximum number of periods to shift in both directions for correlation analysis"
            )
            st.info("🔍 Finds the time delay that maximizes correlation between the two assets")

        else:  # Granger Causality
            max_lag = st.slider(
                "Maximum Lags to Include:",
                min_value=1,
                max_value=24,
                value=12,
                help="Maximum number of past lags to include in the causality test"
            )
            st.info("🧠 Tests whether past prices of one asset contain predictive information for another asset")

        # Analysis execution button
        if st.button("🚀 Execute Lead-Lag Analysis", type="primary"):
            with st.spinner("Analyzing lead-lag relationship..."):
                try:
                    # Initialize the lead-lag analyzer
                    analyzer = LeadLagAnalyzer()

                    # Determine method key for the analyzer
                    method_key = 'cross_correlation' if method == "📊 Cross-Correlation" else 'granger_causality'

                    # Execute the lead-lag analysis
                    results = analyzer.analyze_lead_lag(
                        assets_data[asset1],
                        assets_data[asset2],
                        asset1,
                        asset2,
                        method=method_key,
                        max_lag=max_lag
                    )

                    # Store results in session state for persistence
                    st.session_state.lead_lag_results = results
                    st.session_state.current_assets = (asset1, asset2)
                    st.session_state.current_method = method

                except Exception as e:
                    st.error(f"Error in analysis: {str(e)}")

        # Display results if analysis has been completed
        if st.session_state.get('lead_lag_results'):
            results = st.session_state.lead_lag_results
            asset1, asset2 = st.session_state.current_assets

            st.markdown("---")
            st.markdown("## 📈 Analysis Results")

            # Main summary visualization
            st.plotly_chart(
                create_lead_lag_summary(results, asset1, asset2),
                use_container_width=True
            )

            # Key metrics display in three columns
            col1, col2, col3 = st.columns(3)

            with col1:
                # Display optimal lag regardless of method
                st.metric("Optimal Lag", f"{results['optimal_lag']} periods")

            with col2:
                if results['method'] == 'cross_correlation':
                    # Cross-correlation: Show maximum correlation value
                    st.metric("Maximum Correlation", f"{results['max_correlation']:.3f}")
                else:
                    # Granger causality: Count significant lags in both directions
                    sig_12 = len(results.get('significant_lags_12', []))
                    sig_21 = len(results.get('significant_lags_21', []))
                    st.metric("Significant Lags", f"{sig_12 + sig_21}")

            with col3:
                # Determine and display the leading asset
                relationship = results['relationship']
                if relationship in ['asset1_leads', 'asset1_causes_asset2']:
                    st.metric("Leader", asset1)
                elif relationship in ['asset2_leads', 'asset2_causes_asset1']:
                    st.metric("Leader", asset2)
                else:
                    st.metric("Relationship", "Bidirectional/Unclear")

            # Method-specific detailed visualizations
            st.markdown("### 📊 Detailed Visualization")

            if results['method'] == 'cross_correlation':
                # Cross-correlation plot showing correlation at different lags
                fig = plot_cross_correlation(results)
                st.plotly_chart(fig, use_container_width=True)

                # Extended interpretation for cross-correlation results
                st.markdown("#### 📝 Interpretation")
                if results['optimal_lag'] > 0:
                    st.success(f"**{asset1} LEADS {asset2} by {results['optimal_lag']} periods**")
                    st.info(
                        f"Price movements in {asset1} tend to precede those in {asset2} by {results['optimal_lag']} periods. "
                        f"This suggests that {asset1} may serve as an early indicator for {asset2}. "
                        f"Traders could monitor {asset1} for signals about future movements in {asset2}."
                    )
                elif results['optimal_lag'] < 0:
                    st.warning(f"**{asset2} LEADS {asset1} by {abs(results['optimal_lag'])} periods**")
                    st.info(
                        f"Price movements in {asset2} tend to precede those in {asset1} by {abs(results['optimal_lag'])} periods. "
                        f"Consider monitoring {asset2} to anticipate movements in {asset1}. "
                        f"This relationship could inform timing decisions for entry and exit points."
                    )
                else:
                    st.info("**No clear lead-lag relationship detected**")
                    st.info(
                        "The assets appear to move synchronously without a clear leader. "
                        "They may be responding to the same market factors simultaneously, "
                        "or their relationship may be more complex than a simple lead-lag pattern."
                    )

            else:  # granger_causality
                # Granger causality results visualization
                fig = plot_granger_results(results)
                st.plotly_chart(fig, use_container_width=True)

                # Extended interpretation for Granger causality results
                st.markdown("#### 📝 Interpretation")
                if results['relationship'] == 'asset1_causes_asset2':
                    st.success(f"**{asset1} Granger-causes {asset2}**")
                    st.info(
                        f"Past prices of {asset1} contain statistically useful information for predicting {asset2}. "
                        f"This indicates a causal relationship where {asset1} influences {asset2}. "
                        f"From a trading perspective, monitoring {asset1} could provide predictive insights for {asset2}."
                    )
                elif results['relationship'] == 'asset2_causes_asset1':
                    st.warning(f"**{asset2} Granger-causes {asset1}**")
                    st.info(
                        f"Past prices of {asset2} contain statistically useful information for predicting {asset1}. "
                        f"{asset2} appears to have statistical influence over {asset1}. "
                        f"This relationship could be valuable for developing predictive models and trading strategies."
                    )
                elif results['relationship'] == 'bidirectional':
                    st.info("**Bidirectional Relationship (Feedback Loop)**")
                    st.info(
                        "Both assets influence each other in a feedback relationship. "
                        "Past prices of both assets are useful for predicting each other. "
                        "This complex interaction suggests a tightly coupled relationship "
                        "where both assets respond to and influence each other's price movements."
                    )
                else:
                    st.info("**No significant Granger causality detected**")
                    st.info(
                        "No statistical evidence found that past prices of one asset help predict the other. "
                        "The assets may be influenced by independent factors, "
                        "or their relationship may not be captured by linear Granger causality tests."
                    )

                # Stationarity warning for Granger causality
                if not results.get('stationary_asset1', True) or not results.get('stationary_asset2', True):
                    st.warning(
                        "⚠️ Some data may not be stationary. Granger causality results should be interpreted with caution. "
                        "Non-stationary data can lead to spurious regression results."
                    )

    else:
        # Welcome screen and instructions for new users
        st.info("""
        ## 🕒 Lead-Lag Relationship Analysis

        **Discover which assets lead price movements and which follow**

        ### 📊 Available Methods:

        **1. Cross-Correlation Analysis**
        - Finds the optimal time lag that maximizes correlation between two assets
        - Result example: "BTC leads ETH by 2 hours"
        - Ideal for: Trade timing, pairs trading strategies, momentum analysis

        **2. Granger Causality Testing**  
        - Detects statistical causality between time series
        - Result example: "BTC Granger-causes ETH"
        - Ideal for: Structural analysis, influence relationships, predictive modeling

        ### 🚀 Getting Started:
        1. Select assets in the sidebar (minimum 2 required)
        2. Configure timeframe and analysis parameters
        3. Click **"🚀 Load and Analyze"** to load the data
        4. Choose specific asset pairs and analysis method
        5. Execute lead-lag analysis and interpret results

        ### 💡 Practical Applications:
        - **Pairs Trading**: Identify leader-follower pairs for statistical arbitrage
        - **Momentum Strategies**: Find assets that consistently lead market movements
        - **Risk Management**: Understand how price movements propagate through markets
        - **Market Research**: Study information flow and market microstructure

        ### 📈 Interpretation Guide:
        - **Positive Lag**: First asset leads the second
        - **Negative Lag**: Second asset leads the first  
        - **Zero Lag**: Simultaneous movement, no clear leader
        - **Granger Causality**: Statistical evidence of predictive relationship
        - **Bidirectional**: Mutual influence between assets
        """)


if __name__ == "__main__":
    render()