import streamlit as st
from ui.components import apply_global_styles
from ui.pages.correlations import render as render_correlations
from ui.pages.session_analysis import render as render_session_analysis
from ui.pages.lead_lag_analysis import render as render_lead_lag
from ui.pages.asset_analysis import render as render_asset_analysis
from ui.pages.multi_timeframe_analysis import render as render_multi_timeframe
from ui.pages.impact_analysis import render as render_impact_analysis
from ui.pages.multi_timeframe_lead_lag import render as render_multi_timeframe_lead_lag

# Configure the main application page settings
st.set_page_config(
    page_title="Crypto Correlations EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """
    Main application entry point for Crypto Correlations Dashboard
    Handles page routing and global UI configuration
    """
    # Apply global CSS styles for consistent theming across all pages
    apply_global_styles()

    # Sidebar navigation configuration
    st.sidebar.title("📊 Crypto Correlations Dashboard")

    # Page selection dropdown with all available analysis modules
    page = st.sidebar.selectbox(
        "Select a page:",
        [
            "Correlations",
            "Session Analysis",
            "Lead-Lag Analysis",
            "Asset Analysis",
            "Multi-Timeframe Analysis",
            "Impact Analysis",
            "Multi-Timeframe Lead-Lag Analysis"
        ],
        help="Choose which analysis module to explore"
    )

    # Page routing logic - render the selected analysis module
    if page == "Correlations":
        render_correlations()
    elif page == "Session Analysis":
        render_session_analysis()
    elif page == "Lead-Lag Analysis":
        render_lead_lag()
    elif page == "Asset Analysis":
        render_asset_analysis()
    elif page == "Multi-Timeframe Analysis":
        render_multi_timeframe()
    elif page == "Impact Analysis":
        render_impact_analysis()
    elif page == "Multi-Timeframe Lead-Lag Analysis":
        render_multi_timeframe_lead_lag()

    # Optional: Add footer or additional global information
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **About this Dashboard:**

        Advanced exploratory data analysis tool for cryptocurrency 
        correlation patterns across different timeframes and market conditions.

        Built with Streamlit for interactive financial analysis.
        """
    )


# Application entry point
if __name__ == "__main__":
    main()