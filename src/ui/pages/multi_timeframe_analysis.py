import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go

from core.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from utils.multi_timeframe_viz import MultiTimeframeVisualizer
from ui.components import render_sidebar


def render():
    """
    MAIN RENDERING FUNCTION - Multi-Timeframe Correlation Analysis Dashboard

    This module provides comprehensive analysis of how correlations between financial assets
    behave across different timeframes (15min, 30min, 1h, 4h, 1d). It helps traders and
    analysts understand temporal consistency, identify timeframe-specific opportunities,
    and detect market structure patterns.

    Key Analytical Features:
    - Correlation calculation across 5 different timeframes
    - Consistency scoring between timeframes
    - Pattern detection (decoupling, structural consistency, trending correlations)
    - Actionable trading insights and recommendations
    - Multiple visualization types for different analytical perspectives
    """

    st.markdown("## 📊 Multi-Timeframe Correlation Analysis")

    st.info("""
    **🔍 Analytical Objective:** Analyze how correlations between assets behave 
    across different timeframes (15min, 30min, 1h, 4h, 1d) to identify:
    - Temporal consistency or inconsistency patterns
    - Timeframe-specific trading opportunities
    - Market decoupling/coupling patterns
    - Structural relationships in market dynamics
    """)

    # Sidebar for asset selection - reusing the consistent component
    selected_assets, _, _, _, _ = render_sidebar()

    # Specific selector for this page (maximum 2 assets for focused analysis)
    st.markdown("### 🎯 Asset Selection for Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if selected_assets:
            asset1 = st.selectbox(
                "First Asset:",
                options=selected_assets,
                index=0,
                key="mtf_asset1",
                help="Select the first asset for multi-timeframe correlation analysis"
            )
        else:
            st.warning("Please select assets in the sidebar first")
            return

    with col2:
        # Filter out the first asset to prevent self-comparison
        other_assets = [a for a in selected_assets if a != asset1]
        if other_assets:
            asset2 = st.selectbox(
                "Second Asset:",
                options=other_assets,
                index=min(1, len(other_assets) - 1),
                key="mtf_asset2",
                help="Select the second asset to compare with the first asset"
            )
        else:
            st.warning("At least 2 different assets are required for analysis")
            return

    # Analysis configuration section
    st.markdown("### ⚙️ Analysis Configuration")

    col1, col2 = st.columns(2)

    with col1:
        min_periods = st.slider(
            "Minimum Required Periods:",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Minimum number of common periods required to calculate reliable correlation"
        )

    with col2:
        analysis_type = st.selectbox(
            "Visualization Type:",
            options=["Complete Dashboard", "Simple Comparison", "Trend Analysis", "Consistency Radar"],
            index=0,
            help="Choose how to visualize the multi-timeframe correlation results"
        )

    # Main analysis execution button
    if st.button("🚀 Execute Multi-Timeframe Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing correlations across all timeframes..."):
            try:
                # Initialize analyzer and visualizer components
                analyzer = MultiTimeframeAnalyzer()
                visualizer = MultiTimeframeVisualizer()

                # Calculate correlations across all 5 timeframes
                # This is the core analytical operation that computes relationships
                # between the two selected assets at different temporal granularities
                results = analyzer.calculate_timeframe_correlations(
                    asset1, asset2, min_periods
                )

                # Analyze consistency patterns across timeframes
                # This determines how stable the relationship is across different time horizons
                consistency_analysis = analyzer.analyze_correlation_consistency(results)

                # Display success message and quick overview metrics
                st.success(f"✅ Analysis completed for {asset1} vs {asset2}")

                # Quick metrics dashboard for immediate insights
                st.markdown("### 📈 Quick Overview Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Number of timeframes with successful correlation calculations
                    successful_tfs = consistency_analysis['successful_timeframes']
                    st.metric("Valid Timeframes", f"{successful_tfs}/5")

                with col2:
                    # Average correlation across all valid timeframes
                    avg_corr = consistency_analysis['average_correlation']
                    st.metric("Average Correlation", f"{avg_corr:.3f}")

                with col3:
                    # Consistency score (1 - coefficient of variation)
                    # Higher scores indicate more stable relationships across timeframes
                    consistency_score = consistency_analysis['consistency_score']
                    st.metric("Consistency Score", f"{consistency_score:.2f}")

                with col4:
                    # Overall trend direction (Increasing/Decreasing/Stable)
                    trend = consistency_analysis['trend']
                    st.metric("Overall Trend", trend)

                # Main visualization section - different chart types for different insights
                st.markdown("### 📊 Analytical Visualizations")

                if analysis_type == "Complete Dashboard":
                    # Comprehensive view with all metrics and patterns
                    fig = visualizer.create_summary_dashboard(
                        results, consistency_analysis, asset1, asset2
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis_type == "Simple Comparison":
                    # Clean comparison of correlations across timeframes
                    fig = visualizer.create_timeframe_comparison_chart(
                        results, asset1, asset2
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis_type == "Trend Analysis":
                    # Focus on how correlations change with timeframe duration
                    fig = visualizer.create_timeframe_trend_analysis(results)
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis_type == "Consistency Radar":
                    # Radar chart showing consistency patterns visually
                    fig = visualizer.create_consistency_radar_chart(
                        results, consistency_analysis
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed results table for quantitative analysis
                st.markdown("### 📋 Detailed Results by Timeframe")

                detailed_data = []
                timeframes = ["15m", "30m", "1h", "4h", "1d"]

                for tf in timeframes:
                    tf_data = results.get(tf, {})
                    detailed_data.append({
                        'Timeframe': tf,
                        'Correlation': f"{tf_data.get('correlation', 0):.3f}",
                        'P-value': f"{tf_data.get('p_value', 1):.3e}",
                        'Periods': tf_data.get('periods', 0),
                        'Intensity': tf_data.get('intensity', 'NO_DATA'),
                        'Significance': tf_data.get('significance', 'NO_DATA'),
                        'Status': tf_data.get('status', 'NO_DATA')
                    })

                df_detailed = pd.DataFrame(detailed_data)
                st.dataframe(df_detailed, use_container_width=True)

                # Actionable insights and trading recommendations
                st.markdown("### 💡 Trading Opportunities Analysis")

                recommendation = consistency_analysis.get('recommendation', '')
                st.info(f"**TRADING RECOMMENDATION:** {recommendation}")

                # Detailed insights expandable section
                with st.expander("🔍 Detailed Market Insights", expanded=True):
                    max_tf = consistency_analysis.get('max_correlation_tf')
                    min_tf = consistency_analysis.get('min_correlation_tf')

                    if max_tf and min_tf:
                        max_corr = results.get(max_tf, {}).get('correlation', 0)
                        min_corr = results.get(min_tf, {}).get('correlation', 0)

                        st.write(f"**📈 Maximum Correlation:** {max_tf} timeframe ({max_corr:.3f})")
                        st.write(f"**📉 Minimum Correlation:** {min_tf} timeframe ({min_corr:.3f})")

                        # Correlation gap analysis - identifies arbitrage opportunities
                        gap = abs(max_corr - min_corr)
                        if gap > 0.4:
                            st.success(
                                f"**🎯 HIGH OPPORTUNITY:** Correlation gap of {gap:.3f} - "
                                f"Assets show significantly different behavior across timeframes. "
                                f"Consider timeframe-specific strategies or mean reversion approaches."
                            )
                        elif gap > 0.2:
                            st.warning(
                                f"**⚠️ MODERATE OPPORTUNITY:** Correlation gap of {gap:.3f} - "
                                f"Differentiated behavior across timeframes presents trading opportunities."
                            )
                        else:
                            st.info(
                                "**🔍 CONSISTENT BEHAVIOR:** Assets maintain similar correlation "
                                "across all timeframes. Suitable for consistent trend-following strategies."
                            )

                # Methodological explanation for transparency
                with st.expander("📚 Methodological Explanation"):
                    st.markdown("""
                    **MULTI-TIMEFRAME ANALYSIS METHODOLOGY:**

                    1. **Correlation Calculation**: 
                       - Method: Spearman Rank Correlation (robust to outliers and non-linear relationships)
                       - Focus: Monotonic relationships rather than strictly linear

                    2. **Statistical Validation**:
                       - Significance Threshold: p-value < 0.05 for statistical significance
                       - Confidence: 95% confidence level for reliable relationships

                    3. **Intensity Classification**:
                       - > 0.7: VERY_STRONG (highly predictable relationship)
                       - 0.5-0.7: STRONG (reliable for trading strategies)  
                       - 0.3-0.5: MODERATE (discernible but less reliable)
                       - 0.1-0.3: WEAK (minimal practical significance)
                       - < 0.1: VERY_WEAK (essentially independent)

                    4. **Consistency Scoring**:
                       - Formula: 1 - Coefficient of Variation (CV)
                       - Range: 0-1, where 1 indicates perfect consistency
                       - Interpretation: How stable the correlation is across timeframes

                    5. **Pattern Detection**:
                       - Intraday Decoupling: Different behavior in short vs long timeframes
                       - Structural Consistency: Similar correlations across all timeframes
                       - Temporal Trends: Systematic changes with timeframe duration
                       - Market Regime Detection: Changes in correlation structure
                    """)

            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info(
                    "💡 Troubleshooting: Verify that selected assets have sufficient historical data across all timeframes")

    else:
        # Initial state - educational content and pattern examples
        st.markdown("---")
        st.markdown("### 💡 Common Correlation Patterns to Detect")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 PATTERN 1: Intraday Decoupling**
            ```
            15m: 0.25 (Weak)
            30m: 0.28 (Weak)  
            1h:  0.31 (Weak)
            4h:  0.72 (Strong)
            1d:  0.78 (Very Strong)
            ```
            **👉 Trading Insight:** Assets behave independently in short timeframes but 
            show strong long-term relationship. Opportunity for mean reversion strategies 
            in intraday trading while maintaining trend-following in longer timeframes.
            """)

        with col2:
            st.markdown("""
            **🎯 PATTERN 2: High Structural Consistency**
            ```
            15m: 0.65 (Strong)
            30m: 0.67 (Strong)
            1h:  0.66 (Strong)
            4h:  0.68 (Strong)  
            1d:  0.70 (Strong)
            ```
            **👉 Trading Insight:** Consistent relationship across all timeframes. 
            Ideal for reliable trend-following strategies and pairs trading. 
            Lower model risk due to temporal stability.
            """)

        st.markdown("""
        **🎯 PATTERN 3: Progressive Coupling**
        ```
        15m: 0.20 (Weak)
        30m: 0.35 (Weak)
        1h:  0.45 (Moderate)
        4h:  0.55 (Strong)
        1d:  0.60 (Strong)
        ```
        **👉 Trading Insight:** Assets are progressively coupling over longer timeframes. 
        Early detection of emerging relationships. Potential for momentum strategies 
        as correlation strengthens with market synchronization.
        """)

        # Additional educational content
        st.markdown("""
        ### 🎓 Understanding Multi-Timeframe Analysis

        **Why Analyze Multiple Timeframes?**

        1. **Market Microstructure Insights**:
           - Short-term noise vs long-term signal separation
           - Market maker behavior across different time horizons
           - Liquidity provision patterns

        2. **Risk Management**:
           - Understand timeframe-specific risks
           - Diversify strategies across time horizons
           - Avoid over-optimization for single timeframe

        3. **Strategy Development**:
           - Identify optimal timeframes for specific strategies
           - Detect regime changes early
           - Improve model robustness

        **Practical Applications:**

        - **Hedge Funds**: Portfolio construction and risk factor analysis
        - **Proprietary Trading**: Timeframe-specific arbitrage opportunities
        - **Risk Managers**: Correlation stability monitoring
        - **Quant Researchers**: Market microstructure studies
        - **Retail Traders**: Strategy timeframe optimization
        """)


if __name__ == "__main__":
    render()