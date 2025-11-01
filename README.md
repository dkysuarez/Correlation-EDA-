### Detailed README for the Asset Correlation Analysis System

**Version:** 1.0  
**Language:** English  
**Run:** `streamlit run app.py`  
**Author:** DkySuarez — dkysuarez1@gmail.com

---

### System overview

This project is an interactive Streamlit application designed to analyze statistical relationships between financial assets, with an emphasis on cryptocurrencies. The goal is explanatory, not commercial: measure, validate and visualize how assets relate to each other across horizons and contexts (pairwise, by category, by market session and in time series), and provide advanced analytical tools such as lead/lag, impact analysis and break detection.

Key points readers should know up front:
- The required input is **log returns** of assets; the system works on returns, not raw prices.
- Development data sources: Binance (500+ assets), categorization from Binance, CoinMarketCap and internal sources (`asset_categorized`).
- Main statistical methods: **Spearman** (default, robust), **Pearson** (optional), p-values, t-test for breaks, cross-correlation and Granger causality.

---

### Contents of the app and how to use each section

#### Correlation Analysis
- Purpose: computes correlation matrices between asset pairs and their p-values of significance.
- Selection options:
  - By category (e.g., DeFi, Layer-1) using `asset_categorized`.
  - By individual asset.
  - Load the entire available universe.
- Available methods: **Spearman** (default), **Pearson**.
- Error prevention: the system detects and filters pairs with non-matching time series or excessive NaNs; check the log of failed pairs.
- Temporal resolution: user can choose frequency/window (e.g., 1m, 2m, 5m, 1h, 1d).
- Outputs: NxN matrix, list of significant pairs, heatmap and network graph.

#### Session Analysis
- Purpose: computes correlations separated by market sessions (Asia, Europe, US) and compares results.
- Notes: because crypto markets run 24/7, statistical significance can be affected in short windows; interpret p-values with care.
- Outputs: tables by session, comparative charts and hourly maps.

#### Lead‑Lag Analysis (Cross-correlation and Causality)
- Purpose: identifies whether one asset tends to precede movements of another.
- Methods: cross-correlation (positive/negative lags) and Granger causality test.
- Typical use: detect leader assets (e.g., BTC → ETH) and quantify the most informative lag.
- Outputs: cross-correlation series by lag; Granger p-values per pair; recommended maximum relevant lag.

#### Individual Asset Analysis
- Purpose: detailed analysis of one asset against the universe or a selection.
- Metrics: average correlation, top correlates, rolling evolution, detected breaks and validated phases.
- Visualizations: timeline, histograms, session relations and network position.

#### Timeframe Comparison
- Purpose: compare how a set of assets behaves across selected timeframes.
- Use: detect whether relationships are stable at 1m vs 1h vs 1d, or whether different time scales show different regimes.

#### Impact Analysis
- Purpose: measure the immediate effect and the effect in the following candle(s) of a movement in one asset on another (similar to lead/lag but event-focused).
- Example: if BTC rises X% in a 2-minute candle, measure the probability and magnitude of ETH’s movement in the same candle and in the next.
- Applications: conditional trade execution, micro-arbitrage signals, confirmation of reactions in highly synchronized markets.

---

### Data requirements and preparation

- Expected format: time series of log returns (timestamp column + one return column per asset).
- If starting from prices, convert to log returns:
  ```python
  import numpy as np
  returns = np.log(prices).diff().dropna()
  ```
- Category file: `asset_categorized.csv` with at least: asset_symbol, category.
- Recommendation: clean data, remove extreme outliers in preprocessing, and ensure a consistent time index across all series to be compared.

---

### Basic commands and execution

1. Clone repo and enter:
   ```bash
   git clone https://github.com/DkySuarez/Correlation-EDA.git
   cd Correlation-EDA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. UI workflow:
   - Select dataset / upload CSV with returns.
   - Choose section (Correlation, Sessions, Lead-Lag, Impact, Individual).
   - Adjust parameters (method, window, timeframe, category filters).
   - Run and explore interactive charts; export reports as HTML/PDF if needed.

---

### Practical notes and interpretation guidance

- Always work with log returns so correlations are comparable and statistical assumptions hold.
- Prefer Spearman in markets with outliers (crypto) to avoid Pearson bias.
- Correlations in short windows can show apparent significance by chance; rely on p-values and phase-validation tests (minimum duration, internal stability, persistence) before operational decisions.
- Splitting by session reduces aggregation noise but also reduces sample size: balance resolution vs statistical power.
- Impact Analysis and Lead-Lag at minute resolution require clean high-frequency data (careful with latency, timestamps and candle alignment).
- Provided categories are based on Binance/CoinMarketCap and are for grouping analyses: validate categories if integrating new assets.

---

### Code structure and main modules

- data_loader/ : loading, validation and single optimized join of series  
- correlation_calculator/ : vectorized computation of correlation matrices and p-values  
- rolling_correlations/ : rolling windows, break detection and phase validation  
- session_analyzer/ : session separation and per-session statistics  
- leadlag/ : cross-correlation and Granger causality utilities  
- impact_analysis/ : algorithms to measure intra/extra-candle reaction  
- visualization/ : Streamlit components, Plotly charts, network graphs  
- app.py : Streamlit entrypoint that orchestrates sections and parameters

---

### Outputs, reports and export

- Interactive dashboard inside the Streamlit UI.  
- Exportable reports:
  - Interactive HTML (full dashboard).
  - PDF (executive summary and static tables).
  - CSV/Parquet with correlation matrices, p-values and the list of detected breaks.
- Logs and list of failed pairs for audit.

---
