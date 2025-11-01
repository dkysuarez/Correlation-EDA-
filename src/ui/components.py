
import streamlit as st
import psutil
import os
from core.asset_metadata import AssetMetadataManager
from core.data_loader import ReturnsLoader


def apply_global_styles():
    """Apply global CSS styles for consistent UI theming"""
    st.markdown("""
    <style>
    /* Main app background and text colors */
    .stApp { 
        background-color: #0E1117; 
        color: #FAFAFA; 
    }

    /* Button styling with gradient effect */
    .stButton>button { 
        background: linear-gradient(135deg, #00C896, #00D9FF);
        color: #000000; 
        font-weight: bold; 
        border: none;
        border-radius: 8px;
    }

    /* Metric card styling for data visualization */
    .stMetric {
        background-color: #2D3038;
        padding: 12px;
        border-radius: 10px;
        border: 2px solid #00C896;
        color: #FAFAFA !important;
        font-weight: bold;
    }

    /* Metric label styling */
    .stMetric [data-testid="stMetricLabel"] {
        color: #FAFAFA !important;
        font-weight: bold !important;
    }

    /* Metric value styling with accent color */
    .stMetric [data-testid="stMetricValue"] {
        color: #00C896 !important;
        font-weight: bold !important;
    }

    /* Tab container and individual tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1E2128;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #2D3038;
        color: #FAFAFA;
    }

    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #00C896 !important;
        color: #000000 !important;
    }

    /* Headers with brand color */
    h1, h2, h3 {
        color: #00C896 !important;
        font-weight: bold;
        font-size: 1.5em !important;
    }

    /* General text and markdown content */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #FAFAFA !important;
        font-size: 1.1em;
    }

    /* Warning messages styling */
    .stWarning {
        background-color: #FF6B6B !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px;
    }

    /* Slider component text visibility */
    .stSlider > div > div > div > div {
        color: #FAFAFA !important;
    }

    /* Enhanced UX styling components */

    /* Warning box for important alerts */
    .warning-box {
        background-color: #FF6B6B20;
        border: 1px solid #FF6B6B;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }

    /* Info box for success messages and status */
    .info-box {
        background-color: #00C89620;
        border: 1px solid #00C896;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }

    /* Disabled button state visual feedback */
    .disabled-button {
        opacity: 0.6;
        cursor: not-allowed;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """
    Render reusable sidebar with IMPROVED VALIDATIONS and user guidance

    Returns:
        tuple: (selected_assets, correlation_method, timeframe, batch_size, load_button)
    """
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")

        # User guidance expandable section
        with st.expander("ℹ️ Usage Guide", expanded=False):
            st.markdown("""
            ### Usage Guide
            - **Correlations**: Measure how assets move together.
              - **Spearman**: Detects non-linear relationships, ideal for financial data.
              - **Pearson**: Detects linear relationships, more sensitive to outliers.
            - **P-values**: Indicate if a correlation is reliable. A p-value < 0.05 means the correlation is statistically significant (not by chance).
            - **Sliders**: Adjust how many assets or rows to display in each tab. Select maximum to see all data.
            - **Batch Processing**: Process assets in groups to optimize memory usage and performance.
            """)

        # Asset selection configuration
        with st.container():
            st.subheader("📋 Asset Selection")
            metadata_mgr = AssetMetadataManager()
            stats = metadata_mgr.get_stats()
            st.success(f"✅ {stats['total_assets_with_data']} assets available")

            # Asset selection mode with three options
            selection_mode = st.radio(
                "Selection mode:",
                ["📁 By Category", "🔍 By Assets", "🚀 Load ALL"],
                index=0,
                help="Choose how to select assets for correlation analysis"
            )

            selected_assets = []

            # CATEGORY-BASED SELECTION: Filter assets by predefined categories
            if selection_mode == "📁 By Category":
                with st.expander("Configure Categories and Assets", expanded=True):
                    categories = metadata_mgr.get_categories()
                    selected_categories = st.multiselect(
                        "Select categories:",
                        options=categories,
                        default=categories[:1] if categories else [],
                        help="Choose one or more asset categories to filter available assets"
                    )
                    if selected_categories:
                        # Get assets belonging to selected categories
                        filtered_assets = metadata_mgr.filter_assets(categories=selected_categories, max_assets=500)
                        st.info(f"📊 {len(filtered_assets)} assets in selected categories")
                        selected_assets = st.multiselect(
                            "Select assets:",
                            options=filtered_assets,
                            default=filtered_assets,
                            help="Choose specific assets from the filtered category list"
                        )

            # ASSET-BASED SELECTION: Manual selection from all available assets        
            elif selection_mode == "🔍 By Assets":
                with st.expander("Select Specific Assets", expanded=True):
                    loader = ReturnsLoader(data_dir="data", cache_dir="cache/processed")
                    available_assets = loader.discover_assets()
                    selected_assets = st.multiselect(
                        "Select assets:",
                        options=available_assets,
                        default=["BTCUSDT", "ETHUSDT", "DYX", "SP500"] if all(
                            a in available_assets for a in ["BTCUSDT", "ETHUSDT", "DYX", "SP500"]) else [],
                        help="Choose specific assets like BTCUSDT, ETHUSDT, DYX, SP500, etc."
                    )
                    if selected_assets:
                        st.info(f"📊 {len(selected_assets)} assets selected")

            # BULK LOAD: Load all available assets with configurable limits
            elif selection_mode == "🚀 Load ALL":
                with st.expander("Configure Bulk Load", expanded=True):
                    st.warning("⚠️ Loading ALL assets may take several minutes")
                    col1, col2 = st.columns(2)
                    with col1:
                        load_all_confirmed = st.checkbox("Confirm bulk load")
                    with col2:
                        max_to_load = st.number_input(
                            "Maximum asset limit:",
                            min_value=100,
                            max_value=2000,  # Increased to support more than 500 assets
                            value=500,
                            step=100,
                            help="Maximum number of assets to load for analysis"
                        )
                    if load_all_confirmed:
                        selected_assets = metadata_mgr.filter_assets(load_all=True, max_assets=max_to_load)
                        st.success(f"🚀 Prepared to load {len(selected_assets)} assets")

        # Processing configuration section
        with st.container():
            st.subheader("🔧 Processing Configuration")

            # Correlation method selection
            correlation_method = st.selectbox(
                "Correlation method:",
                ["Spearman", "Pearson"],
                index=0,
                help="Statistical method for calculating correlations between assets"
            )

            # Timeframe selection for data aggregation
            timeframe = st.selectbox(
                "Timeframe:",
                ["15m", "30m", "1h", "4h", "1d"],
                index=2,
                help="Time interval for aggregating asset return data"
            )

            # Performance mode toggle
            max_performance = st.checkbox(
                "Maximum performance mode",
                help="Automatically adjust batch size based on system resources (memory and CPU)"
            )

            # BATCH SIZE CONFIGURATION: Intelligent resource-based calculation
            if max_performance:
                # System resource detection for optimal performance
                available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Memory in GB
                cpu_count = os.cpu_count() or 4  # CPU core count with fallback

                # Optimized formula considering both memory and CPU capabilities
                calculated_batch_size = int(available_memory * 80 + cpu_count * 40)

                # Intelligent limits based on available system resources
                if available_memory < 4:  # Less than 4GB RAM
                    max_recommended = 300
                elif available_memory < 8:  # Less than 8GB RAM
                    max_recommended = 600
                elif available_memory < 16:  # Less than 16GB RAM
                    max_recommended = 1000
                else:  # 16GB+ RAM systems
                    max_recommended = 1500

                # Apply calculated batch size with safety limits
                batch_size = min(calculated_batch_size, max_recommended)

                # Display system resource information
                st.info(f"""
                **⚡ Maximum performance mode activated:**
                - 🧠 Available RAM: {available_memory:.1f} GB
                - ⚡ CPU Cores: {cpu_count}
                - 📦 Batch size: {batch_size} assets
                - 🛡️ Safe limit: {max_recommended} (resource-based)
                """)
            else:
                # Manual batch size configuration
                batch_size = st.slider(
                    "Batch size:",
                    min_value=10,
                    max_value=1500,  # Increased but with reasonable limit
                    value=100,
                    step=10,
                    help="Number of assets to process per batch. Higher = faster but more memory usage"
                )

            # IMPROVEMENT 1: Enhanced validation with clear user feedback
            assets_count = len(selected_assets)
            is_valid_selection = assets_count >= 2  # Minimum 2 assets required for correlation

            # Dynamic validation messages based on selection state
            if assets_count > 0:
                if assets_count == 1:
                    st.markdown(
                        '<div class="warning-box">⚠️ <b>Select at least 1 more asset</b> to perform correlation analysis</div>',
                        unsafe_allow_html=True)
                elif assets_count == 2:
                    st.markdown(
                        '<div class="info-box">✅ <b>Minimum requirement met:</b> 2 assets selected</div>',
                        unsafe_allow_html=True)
                else:
                    # Calculate unique correlation pairs for n assets: n*(n-1)/2
                    n_pairs = assets_count * (assets_count - 1) // 2
                    st.markdown(
                        f'<div class="info-box">✅ <b>Optimal:</b> {assets_count} assets selected ({n_pairs:,} unique pairs)</div>',
                        unsafe_allow_html=True)

            # IMPROVEMENT 2: Enhanced button state with clear user guidance
            load_button = st.button(
                "🚀 Load and Analyze",
                type="primary",
                disabled=not is_valid_selection,
                help="Enable by selecting at least 2 assets" if not is_valid_selection else "Click to execute analysis"
            )

        # Configuration summary section
        if selected_assets:
            n_pairs = len(selected_assets) * (len(selected_assets) - 1) // 2
            st.markdown("---")
            st.subheader("📊 Configuration Summary")

            config_text = f"""
            • **Selected assets**: {len(selected_assets)}
            • **Unique pairs**: {n_pairs:,}
            • **Correlation method**: {correlation_method}
            • **Timeframe**: {timeframe}
            • **Batch size**: {batch_size}
            • **Mode**: {'⚡ Maximum Performance' if max_performance else '⚙️ Custom'}
            """

            # Add status indicator based on validation
            if not is_valid_selection:
                config_text += "\n• **Status**: ❌ Waiting for more assets..."
            else:
                config_text += "\n• **Status**: ✅ Ready to analyze!"

            st.info(config_text)

        return selected_assets, correlation_method, timeframe, batch_size, load_button


def validate_analysis_parameters(selected_assets, batch_size):
    """
    Validate analysis parameters before execution to prevent system issues

    Args:
        selected_assets (list): List of selected asset identifiers
        batch_size (int): Number of assets to process per batch

    Returns:
        tuple: (is_valid, message) where is_valid is boolean and message is str
    """
    # Minimum asset count validation
    if not selected_assets or len(selected_assets) < 2:
        return False, "At least 2 assets are required for analysis"

    # Batch size basic validation
    if batch_size <= 0:
        return False, "Batch size must be greater than 0"

    # Batch size upper limit validation
    if batch_size > 2000:
        return False, "Batch size is too large for system resources"

    # System memory availability validation
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    estimated_memory_need = batch_size * 0.05  # Conservative estimate: 0.05 GB (50 MB) per asset

    # Safety check: don't use more than 80% of available RAM
    if estimated_memory_need > available_memory * 0.8:
        return False, f"Batch size requires ~{estimated_memory_need:.1f}GB of RAM, but only {available_memory:.1f}GB available"

    return True, "Parameters are valid"


def render_correlation_matrix_controls():
    """
    Render controls for correlation matrix visualization and filtering

    Returns:
        tuple: (show_p_values, significance_filter, cluster_matrix, top_n_filter)
    """
    st.subheader("🎛️ Matrix Visualization Controls")

    col1, col2 = st.columns(2)

    with col1:
        # Display options for correlation matrix
        show_p_values = st.checkbox(
            "Show P-values",
            value=True,
            help="Display statistical significance values alongside correlations"
        )

        significance_filter = st.slider(
            "Significance filter (p-value):",
            min_value=0.001,
            max_value=0.1,
            value=0.05,
            step=0.001,
            help="Filter correlations by statistical significance level"
        )

    with col2:
        # Matrix organization and filtering options
        cluster_matrix = st.checkbox(
            "Cluster matrix",
            value=True,
            help="Organize correlation matrix using hierarchical clustering for better pattern recognition"
        )

        top_n_filter = st.slider(
            "Top N correlations to display:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Limit display to top N strongest correlations for better readability"
        )

    return show_p_values, significance_filter, cluster_matrix, top_n_filter


def render_performance_metrics(processing_time, memory_usage, assets_processed):
    """
    Display performance metrics and system resource usage after analysis

    Args:
        processing_time (float): Time taken for analysis in seconds
        memory_usage (float): Memory used during processing in MB
        assets_processed (int): Number of assets successfully processed
    """
    st.subheader("📈 Performance Metrics")

    # Create metrics columns for performance data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Processing Time",
            f"{processing_time:.2f}s",
            help="Total time taken to compute correlations"
        )

    with col2:
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f} MB",
            help="Peak memory usage during processing"
        )

    with col3:
        st.metric(
            "Assets Processed",
            f"{assets_processed:,}",
            help="Number of assets successfully analyzed"
        )

    # Additional system information
    with st.expander("System Resources", expanded=False):
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent()

        st.write(f"**System Memory:** {system_memory.percent}% used")
        st.write(f"**Available RAM:** {system_memory.available / (1024 ** 3):.1f} GB")
        st.write(f"**CPU Usage:** {system_cpu}%")
        st.write(f"**CPU Cores:** {os.cpu_count()}")


def render_export_controls(correlation_data, file_name="correlation_matrix"):
    """
    Render controls for exporting correlation results

    Args:
        correlation_data: DataFrame or dictionary containing correlation results
        file_name (str): Default filename for exported data
    """
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export option
        if st.button("📥 Export to CSV", help="Download correlation matrix as CSV file"):
            # Convert correlation data to CSV
            csv_data = convert_to_csv(correlation_data)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{file_name}.csv",
                mime="text/csv"
            )

    with col2:
        # Excel export option
        if st.button("📊 Export to Excel", help="Download correlation matrix as Excel file"):
            excel_data = convert_to_excel(correlation_data)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"{file_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col3:
        # JSON export option
        if st.button("🔤 Export to JSON", help="Download correlation data as JSON"):
            json_data = convert_to_json(correlation_data)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{file_name}.json",
                mime="application/json"
            )


def convert_to_csv(data):
    """
    Convert correlation data to CSV format

    Args:
        data: Correlation data in DataFrame or dictionary format

    Returns:
        str: CSV formatted string
    """
    # Implementation depends on data structure
    # Placeholder for actual conversion logic
    return "correlation,value\nBTC-ETH,0.75\n"


def convert_to_excel(data):
    """
    Convert correlation data to Excel format

    Args:
        data: Correlation data in DataFrame or dictionary format

    Returns:
        bytes: Excel file as bytes
    """
    # Implementation depends on data structure
    # Placeholder for actual conversion logic
    return b"excel_file_content"


def convert_to_json(data):
    """
    Convert correlation data to JSON format

    Args:
        data: Correlation data in DataFrame or dictionary format

    Returns:
        str: JSON formatted string
    """
    # Implementation depends on data structure
    # Placeholder for actual conversion logic
    return '{"correlations": []}'


def display_loading_progress(current, total, message="Processing assets"):
    """
    Display a progress bar for long-running operations

    Args:
        current (int): Current progress value
        total (int): Total steps to complete
        message (str): Progress bar label
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{message}: {current}/{total} ({progress:.1%})")


def render_error_message(error, context="analysis"):
    """
    Display formatted error messages to users

    Args:
        error (Exception): The error that occurred
        context (str): Context where error happened (analysis, loading, etc.)
    """
    st.error(f"""
    ## ❌ Error during {context}

    **Error Type:** {type(error).__name__}
    **Message:** {str(error)}

    ### Suggested solutions:
    - Reduce the number of selected assets
    - Decrease the batch size
    - Check available system memory
    - Try a different correlation method
    """)

    # Additional debug information in expandable section
    with st.expander("Technical Details"):
        st.code(f"""
        Error details:
        {repr(error)}

        System info:
        Memory: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB available
        CPU Cores: {os.cpu_count()}
        """)


def render_success_message(assets_processed, correlations_found):
    """
    Display success message after successful analysis

    Args:
        assets_processed (int): Number of assets processed
        correlations_found (int): Number of correlations computed
    """
    st.success(f"""
    ## ✅ Analysis Completed Successfully!

    - **Assets Processed:** {assets_processed:,}
    - **Correlations Computed:** {correlations_found:,}
    - **Unique Pairs:** {assets_processed * (assets_processed - 1) // 2:,}

    Navigate to the different tabs to explore your correlation results.
    """)


