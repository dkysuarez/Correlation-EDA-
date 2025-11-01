import numpy as np
import polars as pl
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import concurrent.futures
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class LeadLagAnalyzer:
    """Analiza relaciones lead-lag entre activos usando múltiples métodos"""

    def __init__(self):
        self.supported_methods = ['cross_correlation', 'granger_causality']

    def prepare_returns_data(self, asset1_data: pl.DataFrame, asset2_data: pl.DataFrame,
                             asset1_name: str, asset2_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara y alinea los datos de retornos para análisis
        """
        # Renombrar columnas
        df1 = asset1_data.rename({f"{asset1_name}_returns": "returns_1"})
        df2 = asset2_data.rename({f"{asset2_name}_returns": "returns_2"})

        # Unir por fecha (INNER JOIN para datos comunes)
        combined = df1.join(df2, on="open_time", how="inner")

        if combined.is_empty():
            raise ValueError("No hay datos temporales comunes entre los activos")

        # Extraer retornos como arrays numpy
        returns_1 = combined["returns_1"].drop_nulls().to_numpy()
        returns_2 = combined["returns_2"].drop_nulls().to_numpy()

        # Tomar longitud común mínima
        min_len = min(len(returns_1), len(returns_2))
        returns_1 = returns_1[:min_len]
        returns_2 = returns_2[:min_len]

        print(f"✅ Datos preparados: {min_len} períodos comunes")
        return returns_1, returns_2

    def cross_correlation_analysis(self, returns_1: np.ndarray, returns_2: np.ndarray,
                                   max_lag: int = 24) -> Dict:
        """
        Análisis de cross-correlation para encontrar lag óptimo
        """
        print(f"🔍 Ejecutando Cross-Correlation con max_lag={max_lag}")

        best_lag = 0
        best_correlation = 0
        correlations = []
        lags = list(range(-max_lag, max_lag + 1))

        for lag in lags:
            if lag >= 0:
                # Asset1 lidera: correlaciona asset1[t] con asset2[t+lag]
                x = returns_1[lag:]
                y = returns_2[:-lag] if lag > 0 else returns_2
            else:
                # Asset2 lidera: correlaciona asset1[t+lag] con asset2[t]
                x = returns_1[:lag] if lag < 0 else returns_1
                y = returns_2[-lag:]

            # Asegurar misma longitud
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            if min_len < 30:  # Mínimo de datos
                correlation = 0
            else:
                try:
                    correlation = pearsonr(x, y)[0]
                    if np.isnan(correlation):
                        correlation = 0
                except:
                    correlation = 0

            correlations.append(correlation)

            # Actualizar mejor lag
            if abs(correlation) > abs(best_correlation):
                best_correlation = correlation
                best_lag = lag

        # Interpretación
        if best_lag > 0:
            interpretation = f"Asset1 lidera por {best_lag} períodos"
            relationship = "asset1_leads"
        elif best_lag < 0:
            interpretation = f"Asset2 lidera por {abs(best_lag)} períodos"
            relationship = "asset2_leads"
        else:
            interpretation = "No hay lead-lag claro (lag ≈ 0)"
            relationship = "no_clear_lead"

        return {
            'optimal_lag': best_lag,
            'max_correlation': best_correlation,
            'correlations': correlations,
            'lags': lags,
            'interpretation': interpretation,
            'relationship': relationship,
            'method': 'cross_correlation'
        }

    def check_stationarity(self, returns: np.ndarray) -> bool:
        """
        Verifica estacionariedad usando Augmented Dickey-Fuller test
        """
        try:
            result = adfuller(returns)
            return result[1] <= 0.05  # p-value <= 0.05 → estacionario
        except:
            return False

    def granger_causality_analysis(self, returns_1: np.ndarray, returns_2: np.ndarray,
                                   max_lag: int = 12, significance_level: float = 0.05) -> Dict:
        """
        Análisis de Granger Causality para detectar causalidad estadística
        """
        print(f"🧠 Ejecutando Granger Causality con max_lag={max_lag}")

        # Verificar estacionariedad (requisito para Granger)
        stationary_1 = self.check_stationarity(returns_1)
        stationary_2 = self.check_stationarity(returns_2)

        if not stationary_1 or not stationary_2:
            print("⚠️  Los datos no son estacionarios - resultados pueden no ser confiables")

        # Preparar datos para Granger
        data = np.column_stack([returns_1, returns_2])

        # Test Granger en ambas direcciones
        try:
            # Asset1 → Asset2
            gc_12 = grangercausalitytests(data, maxlag=max_lag, verbose=False)

            # Asset2 → Asset1 (intercambiar columnas)
            gc_21 = grangercausalitytests(data[:, [1, 0]], maxlag=max_lag, verbose=False)
        except Exception as e:
            print(f"❌ Error en Granger test: {e}")
            return {
                'method': 'granger_causality',
                'error': str(e)
            }

        # Extraer p-values y encontrar lag óptimo
        p_values_12 = []
        p_values_21 = []

        for lag in range(1, max_lag + 1):
            p_values_12.append(gc_12[lag][0]['ssr_ftest'][1])
            p_values_21.append(gc_21[lag][0]['ssr_ftest'][1])

        # Encontrar lags significativos
        significant_lags_12 = [lag for lag, p in enumerate(p_values_12, 1) if p <= significance_level]
        significant_lags_21 = [lag for lag, p in enumerate(p_values_21, 1) if p <= significance_level]

        # Interpretación
        if significant_lags_12 and not significant_lags_21:
            interpretation = "Asset1 Granger-causa Asset2"
            relationship = "asset1_causes_asset2"
            optimal_lag = min(significant_lags_12) if significant_lags_12 else 0
        elif significant_lags_21 and not significant_lags_12:
            interpretation = "Asset2 Granger-causa Asset1"
            relationship = "asset2_causes_asset1"
            optimal_lag = min(significant_lags_21) if significant_lags_21 else 0
        elif significant_lags_12 and significant_lags_21:
            interpretation = "Relación bidireccional (feedback)"
            relationship = "bidirectional"
            optimal_lag = min(min(significant_lags_12), min(significant_lags_21))
        else:
            interpretation = "No hay causalidad Granger significativa"
            relationship = "no_causality"
            optimal_lag = 0

        return {
            'p_values_12': p_values_12,
            'p_values_21': p_values_21,
            'significant_lags_12': significant_lags_12,
            'significant_lags_21': significant_lags_21,
            'interpretation': interpretation,
            'relationship': relationship,
            'optimal_lag': optimal_lag,
            'method': 'granger_causality',
            'stationary_asset1': stationary_1,
            'stationary_asset2': stationary_2
        }

    def analyze_lead_lag(self, asset1_data: pl.DataFrame, asset2_data: pl.DataFrame,
                         asset1_name: str, asset2_name: str, method: str = 'cross_correlation',
                         **kwargs) -> Dict:
        """
        Método principal para análisis lead-lag
        """
        # Preparar datos
        returns_1, returns_2 = self.prepare_returns_data(
            asset1_data, asset2_data, asset1_name, asset2_name
        )

        if method == 'cross_correlation':
            max_lag = kwargs.get('max_lag', 24)
            return self.cross_correlation_analysis(returns_1, returns_2, max_lag)

        elif method == 'granger_causality':
            max_lag = kwargs.get('max_lag', 12)
            significance_level = kwargs.get('significance_level', 0.05)
            return self.granger_causality_analysis(
                returns_1, returns_2, max_lag, significance_level
            )

        else:
            raise ValueError(f"Método no soportado: {method}")

    def analyze_hub_relationships(self, hub_asset_data: pl.DataFrame, hub_asset_name: str,
                                  other_assets_data: Dict[str, pl.DataFrame],
                                  method: str = 'cross_correlation', max_workers: int = 4,
                                  **kwargs) -> pd.DataFrame:
        """
        Analiza relaciones lead-lag entre un activo hub y múltiples otros activos
        """
        print(f"🎯 Analizando hub: {hub_asset_name} vs {len(other_assets_data)} activos")

        results = []

        # Función para procesar cada par en paralelo
        def process_asset_pair(other_asset_name):
            try:
                other_asset_data = other_assets_data[other_asset_name]

                # Preparar datos
                returns_hub, returns_other = self.prepare_returns_data(
                    hub_asset_data, other_asset_data, hub_asset_name, other_asset_name
                )

                # Ejecutar análisis según método
                if method == 'cross_correlation':
                    result = self.cross_correlation_analysis(returns_hub, returns_other,
                                                             kwargs.get('max_lag', 24))
                    lead_direction = result['relationship']
                    optimal_lag = result['optimal_lag']
                    strength = abs(result['max_correlation'])

                else:  # granger_causality
                    result = self.granger_causality_analysis(returns_hub, returns_other,
                                                             kwargs.get('max_lag', 12))
                    lead_direction = result['relationship']
                    optimal_lag = result['optimal_lag']
                    # Usar número de lags significativos como fuerza
                    strength = len(result.get('significant_lags_12', [])) + \
                               len(result.get('significant_lags_21', []))

                # Determinar relación
                if lead_direction in ['asset1_leads', 'asset1_causes_asset2']:
                    relationship = f"{hub_asset_name} → {other_asset_name}"
                    hub_leads = True
                elif lead_direction in ['asset2_leads', 'asset2_causes_asset1']:
                    relationship = f"{other_asset_name} → {hub_asset_name}"
                    hub_leads = False
                else:
                    relationship = "No clara"
                    hub_leads = None

                return {
                    'hub_asset': hub_asset_name,
                    'other_asset': other_asset_name,
                    'relationship': relationship,
                    'hub_leads': hub_leads,
                    'optimal_lag': optimal_lag,
                    'strength': strength,
                    'method': method,
                    'success': True
                }

            except Exception as e:
                print(f"❌ Error analizando {hub_asset_name} vs {other_asset_name}: {str(e)}")
                return {
                    'hub_asset': hub_asset_name,
                    'other_asset': other_asset_name,
                    'relationship': f"Error: {str(e)}",
                    'hub_leads': None,
                    'optimal_lag': 0,
                    'strength': 0,
                    'method': method,
                    'success': False
                }

        # Procesar en paralelo para mejor performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_asset = {
                executor.submit(process_asset_pair, asset_name): asset_name
                for asset_name in other_assets_data.keys()
            }

            # Usar tqdm para barra de progreso
            for future in tqdm(concurrent.futures.as_completed(future_to_asset),
                               total=len(other_assets_data), desc="Analizando pares"):
                result = future.result()
                results.append(result)

        # Convertir a DataFrame
        df_results = pd.DataFrame(results)

        # Ordenar por fuerza (descendente)
        df_results = df_results.sort_values('strength', ascending=False)

        print(f"✅ Análisis de hub completado: {len(df_results)} pares procesados")
        return df_results

    def get_hub_summary_stats(self, hub_results: pd.DataFrame) -> Dict:
        """
        Calcula estadísticas resumen para el análisis de hub
        """
        successful_results = hub_results[hub_results['success'] == True]

        if successful_results.empty:
            return {
                'total_pairs': len(hub_results),
                'successful_pairs': 0,
                'hub_leads_count': 0,
                'hub_follows_count': 0,
                'no_clear_count': 0,
                'hub_lead_ratio': 0,
                'avg_strength': 0,
                'avg_lag': 0
            }

        hub_leads = successful_results[successful_results['hub_leads'] == True]
        hub_follows = successful_results[successful_results['hub_leads'] == False]
        no_clear = successful_results[successful_results['hub_leads'].isna()]

        total_successful = len(successful_results)

        return {
            'total_pairs': len(hub_results),
            'successful_pairs': total_successful,
            'hub_leads_count': len(hub_leads),
            'hub_follows_count': len(hub_follows),
            'no_clear_count': len(no_clear),
            'hub_lead_ratio': len(hub_leads) / total_successful if total_successful > 0 else 0,
            'avg_strength': successful_results['strength'].mean(),
            'avg_lag': successful_results['optimal_lag'].abs().mean()
        }