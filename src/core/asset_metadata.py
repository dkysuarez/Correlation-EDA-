# ===== asset_metadata.py (Full version with detailed comments) =====

import polars as pl  # Library for efficient DataFrame handling (used for loading and filtering metadata)
from pathlib import Path  # For handling file paths in a portable way
from typing import List, Optional, Dict  # Types for function annotations (improves readability and static checks)

class AssetMetadataManager:
    """
    Manages asset metadata and categories.
    - Purpose: Load, filter, and discover available assets based on metadata (CSV) and on-disk data.
    - Usage: In the dashboard to select assets by category, search, or full load, ensuring only assets with valid data.
    - Benefits: Avoids loading non-existent assets, categorizes for segmented analysis (e.g., by 'DeFi' or 'Layer1').
    """

    def __init__(self, metadata_file: str = "data/asset_categorized.csv"):
        """
        Initializes the manager with the metadata file.
        - Purpose: Sets up the path to the CSV and loads the metadata DataFrame, then discovers available assets.
        - Usage: Create an instance like `manager = AssetMetadataManager()` to start using the class methods.
        - Args:
            metadata_file: Path to the CSV with 'symbol' and 'category' columns (default: 'data/asset_categorized.csv').
        """
        self.metadata_file = Path(metadata_file)  # Path to the metadata CSV file
        self.metadata = self._load_metadata()  # Loads the metadata DataFrame
        self.available_assets = self._discover_available_assets()  # List of assets with real data on disk

    def _load_metadata(self) -> pl.DataFrame:
        """
        Loads the CSV with categories.
        - Purpose: Reads and normalizes the metadata CSV file.
        - Usage: Called internally during initialization; returns a DataFrame for further filtering.
        - Behavior: If the file doesn't exist, returns an empty DataFrame as a fallback to use only available assets.
        - Returns:
            Polars DataFrame with 'symbol' and 'category' columns.
        """
        if not self.metadata_file.exists():
            print(f"⚠️ File {self.metadata_file} not found, using only available assets")
            return pl.DataFrame({'symbol': [], 'category': []})  # Empty DataFrame if no metadata

        # Load CSV with structure symbol,category
        df = pl.read_csv(self.metadata_file)

        # Normalize column names if necessary (handling inconsistent capitalization)
        if 'Symbol' in df.columns:
            df = df.rename({'Symbol': 'symbol'})
        if 'Category' in df.columns:
            df = df.rename({'Category': 'category'})

        return df

    def _discover_available_assets(self) -> List[str]:
        """
        Discovers which assets have data in data/.
        - Purpose: Scans directories for assets with valid Parquet files (>100 bytes to avoid empty ones).
        - Usage: Called internally; ensures only assets with real data are considered, excluding 'traditional' and 'cache'.
        - Returns:
            Sorted list of available asset symbols.
        """
        data_dir = Path("data")  # Base data directory
        assets = []

        for folder in data_dir.iterdir():
            if folder.is_dir() and folder.name not in ['traditional', 'cache']:  # Exclude special directories
                # Check for the parquet file
                parquet_file = folder / f"{folder.name}.parquet"
                if parquet_file.exists() and parquet_file.stat().st_size > 100:  # Non-empty file
                    assets.append(folder.name)

        print(f"📊 Found {len(assets)} assets with available data")
        return sorted(assets)  # Sort alphabetically for consistency

    def filter_assets(self,
                      categories: Optional[List[str]] = None,
                      search_text: Optional[str] = None,
                      load_all: bool = False,
                      max_assets: int = 500) -> List[str]:
        """
        Filters assets by criteria.
        - Purpose: Selects subsets of assets based on categories, search, or full load.
        - Usage: Call like `manager.filter_assets(categories=['DeFi'], search_text='ETH')` in UI sidebars like Streamlit, limiting to max_assets for safety/performance.
        - Args:
            categories: List of categories to include (e.g., ['DeFi', 'Layer1']).
            search_text: Text to search in symbols (e.g., 'ETH' for ETHUSDT).
            load_all: If True, loads ALL available assets (ignores filters).
            max_assets: Maximum number of assets to return (default 500).
        - Returns:
            List of symbols with valid data.
        """
        if load_all:
            # Load ALL available assets
            return self.available_assets[:max_assets]  # Limit for safety

        # If metadata exists, filter it
        if not self.metadata.is_empty():
            df = self.metadata

            # Filter by categories
            if categories and len(categories) > 0:
                df = df.filter(pl.col('category').is_in(categories))  # Include only specified categories

            # Search by text (case-insensitive)
            if search_text and search_text.strip():
                df = df.filter(
                    pl.col('symbol').str.contains(search_text.upper(), literal=False)
                )  # Contains the text in symbols

            # Filter only those with available data
            df = df.filter(pl.col('symbol').is_in(self.available_assets))  # Intersection with real assets

            # Limit quantity
            symbols = df.head(max_assets)['symbol'].to_list()  # Take first max_assets
            return symbols
        else:
            # No metadata, return all available
            return self.available_assets[:max_assets]  # Limit for safety

    def get_categories(self) -> List[str]:
        """
        Returns unique available categories.
        - Purpose: Lists categories for UI options (e.g., selectbox in Streamlit).
        - Usage: Call `manager.get_categories()` to dynamically show categories based on loaded metadata.
        - Returns:
            Sorted list of unique categories (without nulls).
        """
        if self.metadata.is_empty():
            return []  # No categories if no metadata

        categories = self.metadata['category'].unique().drop_nulls().to_list()
        return sorted(categories)  # Sort alphabetically

    def get_stats(self) -> Dict:
        """
        Statistics of available data.
        - Purpose: Provides a summary of assets and categories for debugging or dashboard display.
        - Usage: Call `manager.get_stats()` to show counts like total assets and per-category in UI.
        - Returns:
            Dict with totals and count per category.
        """
        stats = {
            'total_assets_with_data': len(self.available_assets),  # Assets with valid Parquet
            'total_assets_in_metadata': len(self.metadata) if not self.metadata.is_empty() else 0,  # Assets in CSV
            'categories': {}  # Count per category
        }

        if not self.metadata.is_empty():
            # Count per category
            for cat in self.get_categories():
                count = len(self.metadata.filter(
                    (pl.col('category') == cat) &
                    (pl.col('symbol').is_in(self.available_assets))  # Only with data
                ))
                stats['categories'][cat] = count

        return stats