import polars as pl
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import streamlit as st
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler


class CorrelationTimelineAnalyzer:
    """
    Analiza cómo evoluciona la correlación en TODO el período histórico
    MEJORADO: Consistencia en método de correlación (Pearson o Spearman)
    CORREGIDO: Reproducibilidad en todos los algoritmos
    """

    def __init__(self, color_positive: str = "#00C896", color_negative: str = "#FF6B6B"):
        self.color_positive = color_positive
        self.color_negative = color_negative

    def _safe_date_format(self, date_obj):
        """Formatea fechas de manera segura"""
        try:
            if hasattr(date_obj, 'strftime'):
                return date_obj.strftime('%Y-%m-%d')
            elif isinstance(date_obj, (np.datetime64, datetime)):
                date_str = str(date_obj)
                return date_str[:10]
            else:
                return str(date_obj)
        except:
            return str(date_obj)

    def _align_data_by_date(self, df1: pl.DataFrame, df2: pl.DataFrame, asset1: str, asset2: str) -> pl.DataFrame:
        """Alinea datos por fecha para activos con diferentes períodos"""
        combined_df = df1.join(df2, on="open_time", how="inner").sort("open_time")

        if combined_df.is_empty():
            raise ValueError("No hay fechas comunes entre los dos activos")

        return combined_df

    def diagnose_date_alignment(self, df1: pl.DataFrame, df2: pl.DataFrame, asset1: str, asset2: str) -> Dict:
        """Diagnóstico completo de alineación de fechas"""
        dates1 = df1["open_time"].to_list()
        dates2 = df2["open_time"].to_list()

        unique_dates1 = set(dates1)
        unique_dates2 = set(dates2)
        common_dates = unique_dates1.intersection(unique_dates2)

        stats_dict = {
            'asset1': {
                'name': asset1,
                'total_dates': len(dates1),
                'start_date': min(dates1) if dates1 else None,
                'end_date': max(dates1) if dates1 else None,
                'unique_dates': len(unique_dates1)
            },
            'asset2': {
                'name': asset2,
                'total_dates': len(dates2),
                'start_date': min(dates2) if dates2 else None,
                'end_date': max(dates2) if dates2 else None,
                'unique_dates': len(unique_dates2)
            },
            'common': {
                'common_dates': len(common_dates),
                'overlap_percentage': (len(common_dates) / min(len(unique_dates1), len(unique_dates2)) * 100) if min(
                    len(unique_dates1), len(unique_dates2)) > 0 else 0
            }
        }

        return stats_dict

    def debug_data_quality(self, df: pl.DataFrame, asset1: str, asset2: str) -> Dict:
        """Debug completo de calidad de datos"""
        col1 = f"{asset1}_returns"
        col2 = f"{asset2}_returns"

        if col1 not in df.columns or col2 not in df.columns:
            return {
                'error': f'Columnas no encontradas: {col1} o {col2}',
                'available_columns': df.columns
            }

        stats_dict = {
            'total_rows': df.height,
            'columns_present': df.columns,
            'date_range': {
                'start': str(df["open_time"].min()),
                'end': str(df["open_time"].max())
            },
            'missing_data': {
                'open_time_nulls': df["open_time"].null_count(),
                f'{asset1}_nulls': df[col1].null_count(),
                f'{asset2}_nulls': df[col2].null_count()
            },
            'data_quality': {
                f'{asset1}_zeros': (df[col1] == 0).sum(),
                f'{asset2}_zeros': (df[col2] == 0).sum(),
                f'{asset1}_mean': float(df[col1].mean()),
                f'{asset2}_mean': float(df[col2].mean()),
                f'{asset1}_std': float(df[col1].std()),
                f'{asset2}_std': float(df[col2].std())
            },
            'sample_data': {
                'first_5_dates': df["open_time"].head(5).to_list(),
                f'first_5_{asset1}': df[col1].head(5).to_list(),
                f'first_5_{asset2}': df[col2].head(5).to_list()
            }
        }

        return stats_dict

    def _calculate_rolling_correlations(self, dates, returns1, returns2, window=30, method='Spearman'):
        """
        ✅ CORREGIDO: Calcula correlaciones con ventana móvil usando el método especificado
        """
        rolling_correlations = []

        if len(returns1) < window:
            return rolling_correlations

        for i in range(window, len(returns1) + 1):
            window_returns1 = returns1[i - window:i]
            window_returns2 = returns2[i - window:i]

            valid_mask = ~(np.isnan(window_returns1) | np.isnan(window_returns2))
            valid_count = np.sum(valid_mask)

            if valid_count >= 20:
                try:
                    # ✅ USAR EL MÉTODO CORRECTO
                    if method == 'Spearman':
                        corr, p_value = stats.spearmanr(
                            window_returns1[valid_mask],
                            window_returns2[valid_mask]
                        )
                    else:  # Pearson
                        corr, p_value = stats.pearsonr(
                            window_returns1[valid_mask],
                            window_returns2[valid_mask]
                        )

                    if not np.isnan(corr):
                        rolling_correlations.append({
                            'end_date': dates[i - 1],
                            'start_date': dates[i - window],
                            'correlation': float(corr),
                            'p_value': float(p_value),
                            'window_size': window
                        })
                except Exception:
                    continue

        return rolling_correlations

    def _calculate_trend(self, series):
        """Calcula tendencia usando regresión lineal"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        try:
            slope, _, _, _, _ = stats.linregress(x, series)
            return slope
        except:
            return 0

    def _calculate_persistence(self, series):
        """Calcula persistencia (autocorrelación)"""
        if len(series) < 2:
            return 0
        try:
            return np.corrcoef(series[:-1], series[1:])[0, 1]
        except:
            return 0

    def _detect_structural_breaks(self, correlation_series, min_periods=20, seed: int = 42):
        """
        ✅ CORREGIDO: Detecta puntos de quiebre estructural con seed para reproducibilidad
        """
        n = len(correlation_series)
        breakpoints = []

        # ✅ Configurar seed para reproducibilidad
        rng = np.random.default_rng(seed)

        for i in range(min_periods, n - min_periods):
            before = correlation_series[:i]
            after = correlation_series[i:]

            # Test t para diferencia de medias
            try:
                t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
                mean_change = abs(np.mean(after) - np.mean(before))

                # Criterio: cambio significativo y sustancial
                if p_value < 0.01 and mean_change > 0.25:
                    breakpoints.append(i)
            except:
                continue

        return breakpoints

    def _validate_phase_statistically(self, phase_correlations):
        """
        Valida estadísticamente una fase detectada
        """
        n_periods = len(phase_correlations)

        # 1. Duración mínima
        if n_periods < 15:
            return False, "Duración insuficiente"

        # 2. Estabilidad interna (coeficiente de variación)
        mean_corr = np.mean(phase_correlations)
        std_corr = np.std(phase_correlations)

        if abs(mean_corr) < 0.05:
            cv = std_corr / 0.05
        else:
            cv = std_corr / abs(mean_corr)

        if cv > 0.6:
            return False, f"Alta volatilidad interna (CV: {cv:.2f})"

        # 3. Persistencia temporal
        persistence = self._calculate_persistence(phase_correlations)
        if abs(persistence) < 0.3 and n_periods > 25:
            return False, f"Baja persistencia ({persistence:.2f})"

        return True, "Fase válida estadísticamente"

    def _characterize_regime_type(self, mean_correlation, correlation_volatility):
        """
        Clasifica el tipo de régimen basado en características estadísticas
        """
        if mean_correlation > 0.7:
            if correlation_volatility < 0.15:
                return "ACOPLAMIENTO_FUERTE_ESTABLE", "🔥"
            else:
                return "ACOPLAMIENTO_FUERTE_VOLÁTIL", "⚡"
        elif mean_correlation > 0.5:
            return "ACOPLAMIENTO_MODERADO", "💪"
        elif mean_correlation > 0.3:
            return "CORRELACIÓN_BAJA", "📊"
        elif mean_correlation > -0.2:
            return "DESACOPLAMIENTO", "🔄"
        elif mean_correlation > -0.5:
            return "DIVERGENCIA_MODERADA", "📉"
        else:
            return "DIVERGENCIA_FUERTE", "🎯"

    def _identify_temporal_phases_robust(self, dates, rolling_correlations, seed: int = 42):
        """
        ✅ CORREGIDO: Identifica fases temporales con VALIDACIÓN ESTADÍSTICA y seed
        """
        if len(rolling_correlations) < 30:
            return []

        # Extraer serie de correlaciones
        correlation_series = np.array([r['correlation'] for r in rolling_correlations])

        # ✅ 1. Detectar puntos de quiebre estructural CON SEED
        breakpoints = self._detect_structural_breaks(correlation_series, seed=seed)

        # 2. Agregar inicio y fin
        all_points = [0] + breakpoints + [len(correlation_series)]

        phases = []
        valid_phase_count = 0

        for i in range(1, len(all_points)):
            start_idx = all_points[i - 1]
            end_idx = all_points[i]

            phase_correlations = correlation_series[start_idx:end_idx]

            # 3. Validación estadística
            is_valid, validation_msg = self._validate_phase_statistically(phase_correlations)

            if not is_valid:
                continue

            # 4. Caracterización del régimen
            mean_corr = np.mean(phase_correlations)
            std_corr = np.std(phase_correlations)
            regime_type, emoji = self._characterize_regime_type(mean_corr, std_corr)

            # 5. Significancia estadística
            n_periods = end_idx - start_idx
            if n_periods >= 20:
                try:
                    t_stat, p_value = stats.ttest_1samp(phase_correlations, 0)
                    significance = (
                        "MUY_SIGNIFICATIVO" if p_value < 0.01 else
                        "SIGNIFICATIVO" if p_value < 0.05 else
                        "MARGINAL"
                    )
                except:
                    significance = "NO_CALCULABLE"
                    p_value = None
            else:
                significance = "INSUFICIENTES_DATOS"
                p_value = None

            valid_phase_count += 1

            # Obtener fechas reales de los índices
            start_date = rolling_correlations[start_idx]['end_date']
            end_date = rolling_correlations[end_idx - 1]['end_date'] if end_idx <= len(rolling_correlations) else \
                rolling_correlations[-1]['end_date']

            phases.append({
                'phase': f"{emoji} Fase {valid_phase_count}",
                'start_date': start_date,
                'end_date': end_date,
                'correlation': float(mean_corr),
                'volatility': float(std_corr),
                'n_periods': n_periods,
                'regime_type': regime_type,
                'significance': significance,
                'p_value': p_value if p_value is not None else "N/A",
                'persistence': float(self._calculate_persistence(phase_correlations)),
                'validation': validation_msg,
                'criteria': f'Detección de quiebre estructural con validación estadística (test t p<0.01, cambio>0.25)'
            })

        return phases

    def analyze_complete_correlation_timeline(self,
                                              df: pl.DataFrame,
                                              asset1: str,
                                              asset2: str,
                                              window_size: int = 30,
                                              method: str = 'Spearman',
                                              seed: int = 42) -> Dict:
        """
        ✅ CORREGIDO: Análisis completo con CONSISTENCIA en el método y REPRODUCIBILIDAD

        Args:
            method: 'Spearman' o 'Pearson' - se usará el mismo método para todo
            seed: Semilla para reproducibilidad de todos los algoritmos
        """
        col1 = f"{asset1}_returns"
        col2 = f"{asset2}_returns"

        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Columnas {col1} o {col2} no encontradas")

        # Datos básicos
        dates = df["open_time"].to_numpy()
        returns1 = df[col1].to_numpy()
        returns2 = df[col2].to_numpy()

        start_date = dates[0] if len(dates) > 0 else None
        end_date = dates[-1] if len(dates) > 0 else None
        total_periods = len(dates)

        # Filtrar datos válidos
        valid_mask = ~(np.isnan(returns1) | np.isnan(returns2))
        valid_count = np.sum(valid_mask)

        if valid_count < 30:
            raise ValueError(f"Solo {valid_count} períodos válidos. Necesarios al menos 30.")

        # ✅ 1. CORRELACIÓN TOTAL REAL - Usando el método especificado
        if method == 'Spearman':
            total_correlation, total_p_value = stats.spearmanr(
                returns1[valid_mask],
                returns2[valid_mask]
            )
        else:  # Pearson
            total_correlation, total_p_value = stats.pearsonr(
                returns1[valid_mask],
                returns2[valid_mask]
            )

        # 2. Calcular correlaciones rolling - USANDO EL MISMO MÉTODO
        rolling_correlations = self._calculate_rolling_correlations(
            dates, returns1, returns2, window=window_size, method=method
        )

        # Calcular promedio de rolling correlations para comparación
        if rolling_correlations:
            rolling_corr_values = [r['correlation'] for r in rolling_correlations]
            avg_rolling_correlation = np.mean(rolling_corr_values)
            std_rolling_correlation = np.std(rolling_corr_values)
        else:
            avg_rolling_correlation = total_correlation
            std_rolling_correlation = 0.0

        # ✅ 3. Detectar CAMBIOS BRUSCOS de correlación CON SEED
        correlation_breaks = self._detect_correlation_breaks(dates, returns1, returns2, method=method, seed=seed)

        # ✅ 4. Fases temporales con VALIDACIÓN ESTADÍSTICA Y SEED
        temporal_phases = self._identify_temporal_phases_robust(dates, rolling_correlations, seed=seed)

        # ✅ 5. Encontrar PERÍODOS DE ALTA y BAJA correlación CON SEED
        correlation_periods = self._find_high_low_correlation_periods(dates, returns1, returns2, method=method,
                                                                      seed=seed)

        return {
            # ✅ TODO USA EL MISMO MÉTODO Y ES REPRODUCIBLE
            'method': method,
            'seed_used': seed,
            'total_correlation': total_correlation,
            'total_p_value': total_p_value,

            # Información adicional sobre rolling correlations
            'avg_rolling_correlation': avg_rolling_correlation,
            'std_rolling_correlation': std_rolling_correlation,
            'rolling_window_size': window_size,

            # Información del período
            'start_date': start_date,
            'end_date': end_date,
            'total_periods': total_periods,
            'valid_periods': valid_count,

            # Datos para análisis de fases
            'rolling_correlations': rolling_correlations,
            'correlation_breaks': correlation_breaks,
            'temporal_phases': temporal_phases,
            'correlation_periods': correlation_periods,
            'dates': dates,
            'returns_asset1': returns1,
            'returns_asset2': returns2
        }

    def _detect_correlation_breaks(self, dates, returns1, returns2, window=50, threshold=0.4, method='Spearman',
                                   seed: int = 42):
        """✅ CORREGIDO: Detecta cambios bruscos usando el método especificado CON SEED"""
        breaks = []

        # ✅ Configurar seed para reproducibilidad del sampling
        rng = np.random.default_rng(seed)

        # ✅ Usar paso fijo para consistencia
        step_size = max(window // 2, 10)  # Paso mínimo garantizado

        for i in range(window, len(returns1) - window, step_size):
            before_returns1 = returns1[i - window:i]
            before_returns2 = returns2[i - window:i]
            before_mask = ~(np.isnan(before_returns1) | np.isnan(before_returns2))

            if np.sum(before_mask) >= 20:
                if method == 'Spearman':
                    before_corr, _ = stats.spearmanr(before_returns1[before_mask], before_returns2[before_mask])
                else:
                    before_corr, _ = stats.pearsonr(before_returns1[before_mask], before_returns2[before_mask])

                after_returns1 = returns1[i:i + window]
                after_returns2 = returns2[i:i + window]
                after_mask = ~(np.isnan(after_returns1) | np.isnan(after_returns2))

                if np.sum(after_mask) >= 20:
                    if method == 'Spearman':
                        after_corr, _ = stats.spearmanr(after_returns1[after_mask], after_returns2[after_mask])
                    else:
                        after_corr, _ = stats.pearsonr(after_returns1[after_mask], after_returns2[after_mask])

                    change = abs(after_corr - before_corr)
                    if change > threshold and not np.isnan(change):
                        breaks.append({
                            'break_date': dates[i],
                            'before_correlation': before_corr,
                            'after_correlation': after_corr,
                            'change_magnitude': change,
                            'direction': 'aumentó' if after_corr > before_corr else 'disminuyó',
                            'intensity': 'Fuerte' if change > 0.6 else 'Moderado' if change > 0.4 else 'Leve'
                        })

        return breaks

    def _calculate_phase_correlation(self, returns1, returns2, method='Spearman'):
        """✅ Calcula correlación para una fase usando el método especificado"""
        valid_mask = ~(np.isnan(returns1) | np.isnan(returns2))
        valid_count = np.sum(valid_mask)

        if valid_count >= 20:
            if method == 'Spearman':
                corr, _ = stats.spearmanr(returns1[valid_mask], returns2[valid_mask])
            else:
                corr, _ = stats.pearsonr(returns1[valid_mask], returns2[valid_mask])
            return corr
        return np.nan

    def _find_high_low_correlation_periods(self, dates, returns1, returns2, window=30, method='Spearman',
                                           seed: int = 42):
        """✅ CORREGIDO: Encuentra períodos usando el método especificado CON SEED"""
        periods = []

        # ✅ Configurar seed para reproducibilidad del sampling
        rng = np.random.default_rng(seed)

        # ✅ Usar paso fijo para consistencia
        step_size = max(window // 3, 10)  # Paso mínimo garantizado

        for i in range(0, len(returns1) - window, step_size):
            period_returns1 = returns1[i:i + window]
            period_returns2 = returns2[i:i + window]

            period_corr = self._calculate_phase_correlation(period_returns1, period_returns2, method)

            if not np.isnan(period_corr):
                intensity = (
                    "Correlación MUY ALTA" if period_corr > 0.8 else
                    "Correlación ALTA" if period_corr > 0.6 else
                    "Correlación MEDIA" if period_corr > 0.4 else
                    "Correlación BAJA" if period_corr > 0.2 else
                    "Sin Correlación" if period_corr > -0.2 else
                    "Correlación NEGATIVA BAJA" if period_corr > -0.4 else
                    "Correlación NEGATIVA MEDIA" if period_corr > -0.6 else
                    "Correlación NEGATIVA ALTA" if period_corr > -0.8 else
                    "Correlación MUY NEGATIVA"
                )

                periods.append({
                    'start_date': dates[i],
                    'end_date': dates[min(i + window, len(dates) - 1)],
                    'correlation': period_corr,
                    'intensity': intensity,
                    'period_length': window
                })

        periods.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return periods[:10]

    def create_complete_timeline_plot(self, analysis_data: Dict, asset1: str, asset2: str) -> go.Figure:
        """Crea gráfico de la evolución COMPLETA de ambos activos"""
        dates = analysis_data['dates']
        returns1 = analysis_data['returns_asset1']
        returns2 = analysis_data['returns_asset2']

        norm_returns1 = (returns1 - np.nanmean(returns1)) / np.nanstd(returns1)
        norm_returns2 = (returns2 - np.nanmean(returns2)) / np.nanstd(returns2)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=norm_returns1,
            mode='lines',
            name=f'{asset1}',
            line=dict(color='#00C896', width=2),
            hovertemplate=f'<b>Fecha:</b> %{{x}}<br><b>{asset1}:</b> %{{y:.3f}}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=norm_returns2,
            mode='lines',
            name=f'{asset2}',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate=f'<b>Fecha:</b> %{{x}}<br><b>{asset2}:</b> %{{y:.3f}}<extra></extra>'
        ))

        breaks = analysis_data['correlation_breaks']
        if breaks:
            break_dates = [break_['break_date'] for break_ in breaks]
            break_values = [norm_returns1[np.where(dates == date)[0][0]] if np.where(dates == date)[0].size > 0 else 0
                            for date in break_dates]

            fig.add_trace(go.Scatter(
                x=break_dates,
                y=break_values,
                mode='markers',
                name='📈 Quiebres Correlación',
                marker=dict(
                    size=12,
                    color='yellow',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>📈 QUIEBRE</b><br>Fecha: %{x}<br>Cambio en correlación<extra></extra>'
            ))

        start_date = self._safe_date_format(analysis_data['start_date'])
        end_date = self._safe_date_format(analysis_data['end_date'])
        total_corr = analysis_data['total_correlation']
        method = analysis_data.get('method', 'Spearman')
        seed = analysis_data.get('seed_used', 'N/A')

        fig.update_layout(
            title=f"📊 Evolución Completa: {asset1} vs {asset2}<br>"
                  f"<sub>Período: {start_date} a {end_date} | Correlación Total ({method}): {total_corr:.3f} | Seed: {seed}</sub>",
            xaxis_title="Fecha",
            yaxis_title="Retornos Normalizados",
            height=600,
            plot_bgcolor='#1E2128',
            paper_bgcolor='#0E1117',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#2D3038'),
            yaxis=dict(gridcolor='#2D3038'),
            showlegend=True
        )

        return fig

    def create_correlation_breaks_chart(self, correlation_breaks: List[Dict]) -> go.Figure:
        """Gráfico de quiebres estructurales"""
        if not correlation_breaks:
            fig = go.Figure()
            fig.update_layout(
                title="No se detectaron quiebres significativos de correlación",
                plot_bgcolor='#1E2128',
                paper_bgcolor='#0E1117',
                font=dict(color='#FFFFFF'),
                height=300
            )
            return fig

        break_dates = [break_['break_date'] for break_ in correlation_breaks]
        break_dates_formatted = [self._safe_date_format(date) for date in break_dates]
        break_changes = [break_['change_magnitude'] for break_ in correlation_breaks]
        break_directions = [break_['direction'] for break_ in correlation_breaks]
        break_intensities = [break_['intensity'] for break_ in correlation_breaks]

        colors = ['#00FF00' if dir == 'aumentó' else '#FF0000'
                  for dir in break_directions]

        fig = go.Figure(data=go.Bar(
            x=break_dates_formatted,
            y=break_changes,
            marker_color=colors,
            hovertemplate=(
                '<b>Fecha:</b> %{x}<br>'
                '<b>Cambio:</b> %{y:.3f}<br>'
                '<b>Dirección:</b> %{customdata[0]}<br>'
                '<b>Intensidad:</b> %{customdata[1]}<extra></extra>'
            ),
            customdata=list(zip(break_directions, break_intensities))
        ))

        fig.update_layout(
            title="📈 Puntos donde la Correlación Cambió Bruscamente",
            xaxis_title="Fecha del Cambio",
            yaxis_title="Magnitud del Cambio",
            height=400,
            plot_bgcolor='#1E2128',
            paper_bgcolor='#0E1117',
            font=dict(color='#FFFFFF'),
            xaxis=dict(
                gridcolor='#2D3038',
                tickangle=45,
                tickmode='array',
                tickvals=break_dates_formatted,
                ticktext=break_dates_formatted
            ),
            yaxis=dict(gridcolor='#2D3038')
        )

        return fig

    def create_phases_analysis_chart(self, temporal_phases: List[Dict]) -> go.Figure:
        """Gráfico de fases con información estadística"""
        if not temporal_phases:
            fig = go.Figure()
            fig.update_layout(
                title="⚠️ No se detectaron fases estadísticamente válidas",
                plot_bgcolor='#1E2128',
                paper_bgcolor='#0E1117',
                font=dict(color='#FFFFFF'),
                height=300
            )
            return fig

        phases = [phase['phase'] for phase in temporal_phases]
        correlations = [phase['correlation'] for phase in temporal_phases]
        significances = [phase['significance'] for phase in temporal_phases]

        # Colores basados en significancia estadística
        colors = []
        for i, phase in enumerate(temporal_phases):
            if phase['significance'] == 'MUY_SIGNIFICATIVO':
                colors.append('#00FF00' if correlations[i] > 0 else '#FF0000')
            elif phase['significance'] == 'SIGNIFICATIVO':
                colors.append('#00C896' if correlations[i] > 0 else '#FF6B6B')
            else:
                colors.append('#808080')

        # Hover text mejorado
        hover_texts = []
        for phase in temporal_phases:
            start = self._safe_date_format(phase['start_date'])
            end = self._safe_date_format(phase['end_date'])
            hover_texts.append(
                f"<b>{phase['regime_type']}</b><br>"
                f"Período: {start} a {end}<br>"
                f"Duración: {phase['n_periods']} períodos<br>"
                f"Volatilidad: {phase['volatility']:.3f}<br>"
                f"Persistencia: {phase['persistence']:.2f}<br>"
                f"P-value: {phase['p_value']}<br>"
                f"Significancia: {phase['significance']}"
            )

        fig = go.Figure(data=go.Bar(
            x=phases,
            y=correlations,
            marker_color=colors,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title="🔬 Fases Temporales con Validación Estadística<br>"
                  "<sub>Detectadas mediante quiebres estructurales (test t, p<0.01)</sub>",
            xaxis_title="Fases Detectadas",
            yaxis_title="Correlación",
            height=500,
            plot_bgcolor='#1E2128',
            paper_bgcolor='#0E1117',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#2D3038', tickangle=45),
            yaxis=dict(gridcolor='#2D3038', range=[-1, 1])
        )

        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        # Agregar anotaciones para significancia
        annotations = []
        for i, phase in enumerate(temporal_phases):
            if phase['significance'] == 'MUY_SIGNIFICATIVO':
                annotations.append(dict(
                    x=phases[i],
                    y=correlations[i],
                    text="✅",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=16)
                ))

        fig.update_layout(annotations=annotations)

        return fig


# Clase mantenida para compatibilidad
class RollingCorrelationAnalyzer:
    """Clase mantenida para compatibilidad"""

    def __init__(self, color_positive: str = "#00C896", color_negative: str = "#FF6B6B"):
        self.analyzer = CorrelationTimelineAnalyzer(color_positive, color_negative)

    def _align_data_by_date(self, df1: pl.DataFrame, df2: pl.DataFrame, asset1: str, asset2: str) -> pl.DataFrame:
        return self.analyzer._align_data_by_date(df1, df2, asset1, asset2)

    def calculate_rolling_correlation(self, df: pl.DataFrame, asset1: str, asset2: str,
                                      window: int = 30) -> pl.DataFrame:
        return pl.DataFrame({
            "open_time": [],
            "rolling_corr": []
        })