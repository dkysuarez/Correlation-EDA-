import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.correlation_calculator import CorrelationCalculator
from datetime import timedelta
import plotly.express as px
import streamlit as st


class SessionCorrelationAnalyzer:
    """Analiza correlaciones por sesiones de trading - VERSIÓN CORREGIDA"""

    def __init__(self):
        self.sessions = {
            "Asia": ("00:00", "08:00"),  # 00:00-08:00 UTC
            "Europa": ("08:00", "16:00"),  # 08:00-16:00 UTC
            "US": ("16:00", "24:00")  # 16:00-24:00 UTC
        }

    def filter_data_by_session(self, df: pl.DataFrame, session: str) -> pl.DataFrame:
        """
        Filtra datos por sesión de trading basado en hora UTC
        """
        if session not in self.sessions:
            raise ValueError(f"Sesión {session} no válida. Opciones: {list(self.sessions.keys())}")

        start_hour, end_hour = self.sessions[session]
        start_hour_int = int(start_hour.split(":")[0])
        end_hour_int = int(end_hour.split(":")[0])

        try:
            # Extraer hora UTC del open_time
            df_with_hour = df.with_columns([
                pl.col("open_time").dt.hour().alias("hour_utc")
            ])

            # Filtrar por rango de horas
            if start_hour_int < end_hour_int:
                # Sesión normal (ej: 08:00-16:00)
                df_filtered = df_with_hour.filter(
                    pl.col("hour_utc").is_between(start_hour_int, end_hour_int - 1)
                )
            else:
                # Sesión que cruza medianoche (ej: 16:00-24:00)
                df_filtered = df_with_hour.filter(
                    (pl.col("hour_utc") >= start_hour_int) | (pl.col("hour_utc") < end_hour_int)
                )

            result = df_filtered.drop("hour_utc")
            return result

        except Exception as e:
            # Devolver DataFrame vacío pero con misma estructura
            return df.clear()

    def calculate_session_correlations(self, assets_data: Dict[str, pl.DataFrame],
                                       method: str = 'Spearman') -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        VERSIÓN CORREGIDA - Con umbrales estadísticamente robustos
        """
        calculator = CorrelationCalculator()
        results = {}

        for session_name in self.sessions.keys():
            # Paso 1: Filtrar por sesión
            session_data = {}
            asset_names = []

            for asset, df_full in assets_data.items():
                df_session = self.filter_data_by_session(df_full, session_name)

                # ✅ CORRECCIÓN 1: Cambiar de >= 10 a >= 30 (mínimo estadístico)
                if not df_session.is_empty() and len(df_session) >= 30:
                    # Normalizar open_time a tipo consistente
                    df_normalized = df_session.with_columns([
                        pl.col("open_time").cast(pl.Datetime("ms")).alias("open_time")
                    ])
                    session_data[asset] = df_normalized
                    asset_names.append(asset)

            if len(asset_names) < 2:
                results[session_name] = (None, None, asset_names)
                continue

            # Paso 2: Encontrar rango temporal común
            all_dates_list = []
            for asset, df in session_data.items():
                dates = df['open_time'].to_list()
                all_dates_list.extend(dates)

            if not all_dates_list:
                results[session_name] = (None, None, asset_names)
                continue

            # Crear DataFrame con todas las fechas
            all_dates_df = pl.DataFrame({'open_time': all_dates_list}).sort('open_time')
            total_dates = len(all_dates_df)

            # Usar percentiles para encontrar rango con buena densidad
            start_idx = int(total_dates * 0.10)
            end_idx = int(total_dates * 0.90)

            if start_idx >= end_idx:
                start_idx = 0
                end_idx = total_dates - 1

            common_start = all_dates_df['open_time'][start_idx]
            common_end = all_dates_df['open_time'][end_idx]

            # Paso 3: Combinar datos
            # Encontrar el activo con más datos en el rango común
            base_asset = None
            max_rows = 0

            for asset in asset_names:
                df_asset = session_data[asset]
                df_filtered = df_asset.filter(
                    (pl.col('open_time') >= common_start) &
                    (pl.col('open_time') <= common_end)
                )
                if len(df_filtered) > max_rows:
                    max_rows = len(df_filtered)
                    base_asset = asset

            if not base_asset:
                results[session_name] = (None, None, asset_names)
                continue

            # Usar el activo base como referencia temporal
            base_df = session_data[base_asset].filter(
                (pl.col('open_time') >= common_start) &
                (pl.col('open_time') <= common_end)
            )

            base_df = base_df.with_columns([
                pl.col("open_time").cast(pl.Datetime("ms"))
            ])

            base_return_col = f"{base_asset}_returns"
            combined_df = base_df.rename({f"{base_asset}_returns": base_return_col})

            # Unir otros activos
            successful_joins = 0
            for asset in asset_names:
                if asset == base_asset:
                    continue

                df_asset = session_data[asset]
                df_filtered = df_asset.filter(
                    (pl.col('open_time') >= common_start) &
                    (pl.col('open_time') <= common_end)
                )

                if len(df_filtered) > 0:
                    df_filtered = df_filtered.with_columns([
                        pl.col("open_time").cast(pl.Datetime("ms"))
                    ])

                    asset_return_col = f"{asset}_returns"
                    df_to_join = df_filtered.rename({f"{asset}_returns": asset_return_col})

                    try:
                        rows_before = len(combined_df)
                        combined_df = combined_df.join(
                            df_to_join.select(['open_time', asset_return_col]),
                            on='open_time',
                            how='left'
                        )
                        rows_after = len(combined_df)

                        if rows_after == rows_before:
                            successful_joins += 1

                    except Exception:
                        continue

            if successful_joins < 1:
                results[session_name] = (None, None, asset_names)
                continue

            # Paso 4: Analizar calidad de datos
            available_data = {}
            valid_assets = []

            for asset in asset_names:
                return_col = f"{asset}_returns"
                if return_col in combined_df.columns:
                    non_null_count = combined_df[return_col].null_count()
                    available_count = len(combined_df) - non_null_count
                    available_data[asset] = available_count
                    coverage_pct = (available_count / len(combined_df)) * 100

                    # ✅ CORRECCIÓN 2: Cambiar de >= 10% a >= 50% (mínimo robusto)
                    # Esto previene correlaciones artificiales por falta de datos
                    if coverage_pct >= 50:
                        valid_assets.append(asset)

            if len(valid_assets) < 2:
                results[session_name] = (None, None, asset_names)
                continue

            # Paso 5: Calcular correlaciones
            try:
                valid_columns = ['open_time'] + [f"{asset}_returns" for asset in valid_assets]
                final_df = combined_df.select(valid_columns)

                corr_matrix, p_matrix = calculator.compute(final_df, method)
                results[session_name] = (corr_matrix.to_numpy(), p_matrix.to_numpy(), valid_assets)

            except Exception:
                results[session_name] = (None, None, valid_assets)

        return results

    def create_session_comparison_chart(self, session_results: Dict, metric: str = "mean") -> go.Figure:
        """
        Crea gráfico comparativo entre sesiones
        """
        sessions = list(self.sessions.keys())
        metrics_data = {}

        for session_name, (corr_matrix, p_matrix, asset_names) in session_results.items():
            if corr_matrix is None or len(corr_matrix) == 0:
                metrics_data[session_name] = 0
                continue

            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            corr_values = corr_matrix[mask]

            if len(corr_values) == 0:
                metrics_data[session_name] = 0
                continue

            if metric == "mean":
                value = np.mean(corr_values)
            elif metric == "max":
                value = np.max(corr_values)
            elif metric == "min":
                value = np.min(corr_values)
            elif metric == "positive_ratio":
                value = len(corr_values[corr_values > 0.1]) / len(corr_values) * 100
            elif metric == "strong_positive_ratio":
                value = len(corr_values[corr_values > 0.5]) / len(corr_values) * 100
            else:
                value = np.mean(corr_values)

            metrics_data[session_name] = value

        # Crear gráfico
        fig = go.Figure()

        colors = {'Asia': '#FF6B6B', 'Europa': '#4ECDC4', 'US': '#45B7D1'}

        for session in sessions:
            value = metrics_data.get(session, 0)
            fig.add_trace(go.Bar(
                name=session,
                x=[session],
                y=[value],
                marker_color=colors.get(session, '#999999'),
                text=[f'{value:.3f}'],
                textposition='auto',
            ))

        metric_titles = {
            "mean": "Correlación Promedio",
            "max": "Correlación Máxima",
            "min": "Correlación Mínima",
            "positive_ratio": "% Correlaciones Positivas",
            "strong_positive_ratio": "% Correlaciones Fuertes Positivas"
        }

        fig.update_layout(
            title=f"Comparación entre Sesiones - {metric_titles.get(metric, metric)}",
            xaxis_title="Sesión de Trading",
            yaxis_title=metric_titles.get(metric, "Valor"),
            template="plotly_dark",
            showlegend=False,
            height=400
        )

        return fig

    def get_session_stats(self, session_results: Dict) -> Dict:
        """
        Obtiene estadísticas detalladas por sesión
        """
        stats = {}

        for session_name, (corr_matrix, p_matrix, asset_names) in session_results.items():
            if corr_matrix is None:
                stats[session_name] = {
                    'status': 'NO_DATA',
                    'n_assets': len(asset_names),
                    'n_periods': 0,
                    'mean_correlation': 0,
                    'max_correlation': 0,
                    'min_correlation': 0,
                    'positive_ratio': 0
                }
                continue

            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            corr_values = corr_matrix[mask]

            if len(corr_values) == 0:
                stats[session_name] = {
                    'status': 'INSUFFICIENT_DATA',
                    'n_assets': len(asset_names),
                    'n_periods': 0,
                    'mean_correlation': 0,
                    'max_correlation': 0,
                    'min_correlation': 0,
                    'positive_ratio': 0
                }
                continue

            positive_count = len(corr_values[corr_values > 0.1])
            strong_positive_count = len(corr_values[corr_values > 0.5])

            stats[session_name] = {
                'status': 'SUCCESS',
                'n_assets': len(asset_names),
                'n_periods': len(corr_values),
                'mean_correlation': float(np.mean(corr_values)),
                'max_correlation': float(np.max(corr_values)),
                'min_correlation': float(np.min(corr_values)),
                'positive_ratio': float(positive_count / len(corr_values) * 100),
                'strong_positive_ratio': float(strong_positive_count / len(corr_values) * 100)
            }

        return stats

    def get_failed_sessions_info(self, session_results: Dict) -> List[Dict]:
        """
        Obtiene información sobre sesiones que fallaron
        """
        failed_info = []

        for session_name, (corr_matrix, p_matrix, assets) in session_results.items():
            if corr_matrix is None:
                if len(assets) >= 2:
                    failed_info.append({
                        'Sesión': session_name,
                        'Activos': len(assets),
                        'Razón': 'No se pudo calcular correlaciones',
                        'Detalle': 'Problemas en combinación de datos o falta de superposición temporal (verifica umbrales: mínimo 30 períodos y 50% cobertura)'
                    })
                else:
                    failed_info.append({
                        'Sesión': session_name,
                        'Activos': len(assets),
                        'Razón': 'Insuficientes activos',
                        'Detalle': f'Solo {len(assets)} activos válidos (se necesitan al menos 2 con mínimo 30 períodos cada uno)'
                    })

        return failed_info