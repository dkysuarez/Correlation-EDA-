import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
import pandas as pd


class MultiTimeframeVisualizer:
    """Visualizaciones para análisis multi-timeframe"""

    def __init__(self):
        self.color_scale = [
            [0.0, "#FF0000"],  # Rojo (correlación negativa fuerte)
            [0.3, "#FF6B6B"],  # Rojo claro
            [0.5, "#2D3038"],  # Gris (neutral)
            [0.7, "#4ECDC4"],  # Verde claro
            [1.0, "#00C896"]  # Verde (correlación positiva fuerte)
        ]
        self.background_color = "#0E1117"
        self.plot_bgcolor = "#1E2128"
        self.text_color = "#FFFFFF"
        self.grid_color = "#2D3038"

    def create_timeframe_comparison_chart(self, results: Dict[str, Dict],
                                          asset1: str, asset2: str) -> go.Figure:
        """Crea gráfico de barras comparando correlaciones por timeframe"""
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        correlations = []
        p_values = []
        periods = []
        statuses = []

        for tf in timeframes:
            data = results.get(tf, {})
            corr = data.get('correlation', 0)
            p_val = data.get('p_value', 1)
            period = data.get('periods', 0)
            status = data.get('status', 'NO_DATA')

            correlations.append(corr)
            p_values.append(p_val)
            periods.append(period)
            statuses.append(status)

        # Crear figura
        fig = go.Figure()

        # Barras de correlación
        colors = ['#00C896' if x >= 0 else '#FF6B6B' for x in correlations]

        fig.add_trace(go.Bar(
            x=timeframes,
            y=correlations,
            marker_color=colors,
            marker_line=dict(color='#2D3038', width=1),
            hovertemplate=(
                    "Timeframe: %{x}<br>" +
                    "Correlación: %{y:.3f}<br>" +
                    "P-value: %{customdata[0]:.3e}<br>" +
                    "Períodos: %{customdata[1]}<br>" +
                    "Estado: %{customdata[2]}<br>" +
                    "<extra></extra>"
            ),
            customdata=list(zip(p_values, periods, statuses)),
            name="Correlación"
        ))

        # Línea de cero
        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        fig.update_layout(
            title=f"Comparación de Correlaciones por Timeframe<br>{asset1} vs {asset2}",
            xaxis_title="Timeframe",
            yaxis_title="Correlación (Spearman)",
            template="plotly_dark",
            height=500,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color),
            xaxis=dict(gridcolor=self.grid_color),
            yaxis=dict(gridcolor=self.grid_color, range=[-1, 1])
        )

        return fig

    def create_correlation_heatmap_matrix(self, all_results: Dict[str, Dict]) -> go.Figure:
        """Crea matriz de heatmaps para múltiples pares de activos"""
        # Esta función sería para comparar múltiples pares (futura expansión)
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        # Para un solo par, mostramos heatmap de evolución temporal
        correlations = [all_results.get(tf, {}).get('correlation', 0) for tf in timeframes]

        fig = go.Figure(data=go.Heatmap(
            z=[correlations],  # Una fila para este par
            x=timeframes,
            y=[f"{list(all_results.keys())[0] if all_results else 'Pair'}"] if all_results else [''],
            colorscale=self.color_scale,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate=(
                    "Timeframe: %{x}<br>" +
                    "Correlación: %{z:.3f}<br>" +
                    "<extra></extra>"
            )
        ))

        fig.update_layout(
            title="Heatmap de Correlación por Timeframe",
            xaxis_title="Timeframe",
            template="plotly_dark",
            height=300,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
        )

        return fig

    def create_consistency_radar_chart(self, results: Dict[str, Dict],
                                       consistency_analysis: Dict) -> go.Figure:
        """Crea gráfico radar mostrando consistencia entre timeframes"""
        timeframes = ["15m", "30m", "1h", "4h", "1d"]

        # Datos para el radar
        correlations = []
        intensities = []

        for tf in timeframes:
            data = results.get(tf, {})
            corr = data.get('correlation', 0)
            correlations.append(abs(corr))  # Usamos valor absoluto para radar

            # Convertir intensidad a numérico
            intensity_map = {
                "MUY_FUERTE": 1.0, "FUERTE": 0.8, "MODERADA": 0.6,
                "DEBIL": 0.4, "MUY_DEBIL": 0.2, "NO_DATA": 0.0
            }
            intensity = intensity_map.get(data.get('intensity', 'NO_DATA'), 0.0)
            intensities.append(intensity)

        fig = go.Figure()

        # Traza de correlaciones absolutas
        fig.add_trace(go.Scatterpolar(
            r=correlations + [correlations[0]],  # Cerrar el círculo
            theta=timeframes + [timeframes[0]],
            fill='toself',
            name='Correlación Absoluta',
            line=dict(color='#00C896'),
            fillcolor='rgba(0, 200, 150, 0.3)'
        ))

        # Traza de intensidad
        fig.add_trace(go.Scatterpolar(
            r=intensities + [intensities[0]],
            theta=timeframes + [timeframes[0]],
            fill='toself',
            name='Intensidad',
            line=dict(color='#4ECDC4'),
            fillcolor='rgba(78, 205, 196, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Análisis de Consistencia - Gráfico Radar<br>" +
                  f"Score: {consistency_analysis.get('consistency_score', 0):.2f}",
            template="plotly_dark",
            height=500,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color),
            showlegend=True
        )

        return fig

    def create_timeframe_trend_analysis(self, results: Dict[str, Dict]) -> go.Figure:
        """Crea gráfico de línea mostrando tendencia de correlaciones por timeframe"""
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        correlations = [results.get(tf, {}).get('correlation', 0) for tf in timeframes]
        periods = [results.get(tf, {}).get('periods', 0) for tf in timeframes]

        fig = go.Figure()

        # Línea principal de correlaciones
        fig.add_trace(go.Scatter(
            x=timeframes,
            y=correlations,
            mode='lines+markers+text',
            line=dict(color='#00C896', width=3),
            marker=dict(size=10, color='#00C896'),
            text=[f"{corr:.3f}" for corr in correlations],
            textposition="top center",
            name="Correlación",
            hovertemplate=(
                    "Timeframe: %{x}<br>" +
                    "Correlación: %{y:.3f}<br>" +
                    "Períodos: %{customdata}<br>" +
                    "<extra></extra>"
            ),
            customdata=periods
        ))

        # Área sombreada para correlación positiva/negativa
        x_combined = timeframes + timeframes[::-1]
        y_positive = correlations + [0] * len(timeframes)
        y_negative = [0] * len(timeframes) + correlations[::-1]

        fig.add_trace(go.Scatter(
            x=x_combined,
            y=[max(0, y) for y in y_positive],
            fill='toself',
            fillcolor='rgba(0, 200, 150, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Correlación Positiva',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=x_combined,
            y=[min(0, y) for y in y_negative],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Correlación Negativa',
            hoverinfo='skip'
        ))

        # Línea de cero
        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        fig.update_layout(
            title="Tendencia de Correlación por Timeframe",
            xaxis_title="Timeframe",
            yaxis_title="Correlación",
            template="plotly_dark",
            height=500,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color),
            xaxis=dict(gridcolor=self.grid_color),
            yaxis=dict(gridcolor=self.grid_color, range=[-1, 1])
        )

        return fig

    def create_summary_dashboard(self, results: Dict[str, Dict],
                                 consistency_analysis: Dict,
                                 asset1: str, asset2: str) -> go.Figure:
        """Crea dashboard resumen con múltiples visualizaciones - CORREGIDO"""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Crear subplots CON TIPOS CORRECTOS
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Comparación por Timeframe',
                'Tendencia Temporal',
                'Análisis de Consistencia',
                'Heatmap de Correlaciones'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "polar"}, {"type": "heatmap"}]  # Cambiado de "domain" a "heatmap"
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Gráfico de barras (arriba izquierda)
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        correlations = [results.get(tf, {}).get('correlation', 0) for tf in timeframes]
        colors = ['#00C896' if x >= 0 else '#FF6B6B' for x in correlations]

        fig.add_trace(
            go.Bar(x=timeframes, y=correlations, marker_color=colors,
                   name="Correlación", showlegend=False),
            row=1, col=1
        )

        # 2. Gráfico de tendencia (arriba derecha)
        fig.add_trace(
            go.Scatter(x=timeframes, y=correlations, mode='lines+markers',
                       line=dict(color='#00C896', width=3),
                       marker=dict(size=8, color='#00C896'),
                       name="Tendencia", showlegend=False),
            row=1, col=2
        )

        # 3. Radar chart (abajo izquierda)
        correlations_abs = [abs(c) for c in correlations]
        fig.add_trace(
            go.Scatterpolar(r=correlations_abs + [correlations_abs[0]],
                            theta=timeframes + [timeframes[0]],
                            fill='toself',
                            line=dict(color='#00C896'),
                            fillcolor='rgba(0, 200, 150, 0.3)',
                            name="Consistencia", showlegend=False),
            row=2, col=1
        )

        # 4. Heatmap simple (abajo derecha) - EN LUGAR del texto
        heatmap_data = [correlations]  # Una fila con todas las correlaciones
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=timeframes,
                y=[f"{asset1} vs {asset2}"],
                colorscale=self.color_scale,
                zmin=-1,
                zmax=1,
                showscale=True,
                colorbar=dict(title="Correlación"),
                hovertemplate=(
                        "Timeframe: %{x}<br>" +
                        "Correlación: %{z:.3f}<br>" +
                        "<extra></extra>"
                )
            ),
            row=2, col=2
        )

        # Actualizar layout
        fig.update_layout(
            title_text=f"Dashboard Multi-Timeframe: {asset1} vs {asset2}",
            height=700,
            template="plotly_dark",
            showlegend=False,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
        )

        # Actualizar ejes específicos
        fig.update_xaxes(title_text="Timeframe", row=1, col=1)
        fig.update_yaxes(title_text="Correlación", range=[-1, 1], row=1, col=1)

        fig.update_xaxes(title_text="Timeframe", row=1, col=2)
        fig.update_yaxes(title_text="Correlación", range=[-1, 1], row=1, col=2)

        fig.update_polars(radialaxis_range=[0, 1], row=2, col=1)

        # Añadir métricas como anotaciones en lugar de subplot
        success_count = consistency_analysis.get('successful_timeframes', 0)
        avg_corr = consistency_analysis.get('average_correlation', 0)
        consistency_score = consistency_analysis.get('consistency_score', 0)

        # Añadir anotaciones con las métricas
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=f"<b>MÉTRICAS:</b><br>• Timeframes válidos: {success_count}/5<br>• Corr. promedio: {avg_corr:.3f}<br>• Consistencia: {consistency_score:.2f}",
            showarrow=False,
            bgcolor="rgba(30, 33, 40, 0.8)",
            bordercolor="#00C896",
            borderwidth=1,
            borderpad=10,
            align="left"
        )

        return fig