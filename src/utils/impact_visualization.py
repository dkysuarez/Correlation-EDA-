import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List
from core.impact_analyzer import ImpactResult, CascadeResult

# Colores
COLOR_POSITIVE = '#00C896'
COLOR_NEGATIVE = '#FF6B6B'
COLOR_NEUTRAL = '#4ECDC4'
BG_COLOR = '#0E1117'
PLOT_BG_COLOR = '#1E2128'
GRID_COLOR = '#2D3038'


def create_beta_comparison_chart(results: Dict[str, ImpactResult],
                                 reference_asset: str) -> go.Figure:
    """
    Gráfico de barras comparando betas con confidence intervals
    """
    assets = list(results.keys())
    betas = [results[asset].beta for asset in assets]
    ci_lowers = [results[asset].beta_ci_lower for asset in assets]
    ci_uppers = [results[asset].beta_ci_upper for asset in assets]

    # Calcular errores para error bars
    errors_minus = [betas[i] - ci_lowers[i] for i in range(len(betas))]
    errors_plus = [ci_uppers[i] - betas[i] for i in range(len(betas))]

    colors = [COLOR_POSITIVE if b > 1.0 else COLOR_NEUTRAL if b > 0.5 else COLOR_NEGATIVE
              for b in betas]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=assets,
        y=betas,
        error_y=dict(
            type='data',
            symmetric=False,
            array=errors_plus,
            arrayminus=errors_minus,
            color='white',
            thickness=2
        ),
        marker_color=colors,
        text=[f"{b:.2f}" for b in betas],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Beta: %{y:.3f}<br>'
            'CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>'
            '<extra></extra>'
        ),
        customdata=list(zip(ci_lowers, ci_uppers))
    ))

    # Línea de referencia en 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="yellow",
                  annotation_text="Beta = 1.0 (igual sensibilidad que referencia)")

    fig.update_layout(
        title=f"Sensibilidad (Beta) respecto a {reference_asset}<br>"
              "<sub>Barras de error = 95% Confidence Interval (Bootstrap)</sub>",
        xaxis_title="Activos",
        yaxis_title="Beta (sensibilidad)",
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500,
        showlegend=False
    )

    return fig


def create_scatter_with_regression(ref_returns: np.ndarray,
                                   target_returns: np.ndarray,
                                   asset_name: str,
                                   beta: float,
                                   r_squared: float,
                                   seed: int = 42) -> go.Figure:
    """
    Scatter plot con línea de regresión - CORREGIDO CON SEED

    Args:
        seed: Semilla para reproducibilidad del subsampling
    """
    # Filtrar NaN
    valid_mask = ~(np.isnan(ref_returns) | np.isnan(target_returns))
    ref_clean = ref_returns[valid_mask]
    target_clean = target_returns[valid_mask]

    # ✅ CORRECCIÓN: Subsample con seed para reproducibilidad
    if len(ref_clean) > 5000:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ref_clean), 5000, replace=False)
        ref_clean = ref_clean[indices]
        target_clean = target_clean[indices]

    fig = go.Figure()

    # Scatter
    fig.add_trace(go.Scatter(
        x=ref_clean,
        y=target_clean,
        mode='markers',
        marker=dict(
            size=4,
            color=target_clean,
            colorscale='RdYlGn',
            opacity=0.5,
            showscale=False
        ),
        name='Observaciones',
        hovertemplate='Ref: %{x:.2%}<br>Target: %{y:.2%}<extra></extra>'
    ))

    # Línea de regresión
    x_range = np.array([ref_clean.min(), ref_clean.max()])
    y_pred = beta * x_range

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='yellow', width=3),
        name=f'Beta = {beta:.2f}',
        hovertemplate='Predicción: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Relación de Returns: {asset_name}<br>"
              f"<sub>Beta = {beta:.3f} | R² = {r_squared:.3f} | Seed: {seed}</sub>",
        xaxis_title="Cambio en Referencia (%)",
        yaxis_title=f"Cambio en {asset_name} (%)",
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500
    )

    return fig


def create_lag_response_heatmap(conditional_results: Dict,
                                event_type: str,
                                assets: List[str],
                                max_lag: int) -> go.Figure:
    """
    Heatmap de respuestas por lag

    Args:
        conditional_results: Resultados condicionales por activo
        event_type: 'up_events' o 'down_events'
        assets: Lista de nombres de activos
        max_lag: Número máximo de lags
    """
    # Preparar matriz
    matrix = []

    for asset in assets:
        row = []
        for lag in range(max_lag + 1):
            lag_key = f'lag_{lag}'
            try:
                mean_response = conditional_results[asset][event_type][lag_key]['mean']
                row.append(mean_response * 100)  # Convertir a porcentaje
            except:
                row.append(0.0)
        matrix.append(row)

    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'Lag {i}' for i in range(max_lag + 1)],
        y=assets,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.2f}%" for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate=(
            'Activo: %{y}<br>'
            'Lag: %{x}<br>'
            'Respuesta: %{z:.2f}%<br>'
            '<extra></extra>'
        )
    ))

    event_label = "Subidas" if event_type == 'up_events' else "Bajadas"

    fig.update_layout(
        title=f"Respuesta Media por Lag - {event_label}<br>"
              "<sub>Colores: Verde = respuesta positiva, Rojo = negativa</sub>",
        xaxis_title="Lag (velas después del evento)",
        yaxis_title="Activos",
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=400
    )

    return fig


def create_persistence_chart(results: Dict[str, ImpactResult]) -> go.Figure:
    """
    Gráfico de barras comparando persistencia en subidas vs bajadas
    """
    assets = list(results.keys())
    persistence_up = [results[asset].persistence_up for asset in assets]
    persistence_down = [results[asset].persistence_down for asset in assets]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Subidas',
        x=assets,
        y=persistence_up,
        marker_color=COLOR_POSITIVE,
        text=[f"{p:.1f}%" for p in persistence_up],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Persistencia: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Bajadas',
        x=assets,
        y=persistence_down,
        marker_color=COLOR_NEGATIVE,
        text=[f"{p:.1f}%" for p in persistence_down],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Persistencia: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Persistencia de Dirección tras Eventos<br>"
              "<sub>% de velas que mantienen la dirección esperada en lag 0</sub>",
        xaxis_title="Activos",
        yaxis_title="Persistencia (%)",
        barmode='group',
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500
    )

    return fig


def create_rolling_beta_chart(dates: np.ndarray,
                              rolling_betas: np.ndarray,
                              asset_name: str,
                              overall_beta: float) -> go.Figure:
    """
    Timeline de beta rolling
    """
    fig = go.Figure()

    # Rolling beta
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_betas,
        mode='lines',
        name='Rolling Beta',
        line=dict(color=COLOR_POSITIVE, width=2),
        hovertemplate='Fecha: %{x}<br>Beta: %{y:.3f}<extra></extra>'
    ))

    # Beta promedio
    fig.add_hline(
        y=overall_beta,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Beta Promedio: {overall_beta:.3f}"
    )

    # Beta = 1.0 referencia
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="white",
        annotation_text="Beta = 1.0"
    )

    fig.update_layout(
        title=f"Evolución de Beta (Rolling) - {asset_name}<br>"
              "<sub>Muestra cómo cambia la sensibilidad en el tiempo</sub>",
        xaxis_title="Fecha",
        yaxis_title="Beta",
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500
    )

    return fig


def create_cascade_waterfall(cascade_results: List[CascadeResult]) -> go.Figure:
    """
    Gráfico de cascada mostrando orden de propagación
    """
    assets = [r.asset_name for r in cascade_results]
    response_times = [r.response_time_avg for r in cascade_results]
    correlations = [r.max_correlation for r in cascade_results]

    # Colores según velocidad
    colors = [COLOR_POSITIVE if rt < 2 else COLOR_NEUTRAL if rt < 4 else COLOR_NEGATIVE
              for rt in response_times]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=assets,
        y=response_times,
        marker_color=colors,
        text=[f"#{r.cascade_order}<br>{rt:.1f} velas"
              for r, rt in zip(cascade_results, response_times)],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Orden: #%{customdata[0]}<br>'
            'Tiempo respuesta: %{y:.2f} velas<br>'
            'Correlación máx: %{customdata[1]:.3f}<br>'
            'Lag óptimo: %{customdata[2]}<br>'
            '<extra></extra>'
        ),
        customdata=[[r.cascade_order, r.max_correlation, r.optimal_lag]
                    for r in cascade_results]
    ))

    fig.update_layout(
        title="Orden de Propagación (Cascade Effect)<br>"
              "<sub>Tiempo promedio de respuesta - Verde=rápido, Rojo=lento</sub>",
        xaxis_title="Activos (ordenados por velocidad)",
        yaxis_title="Tiempo de Respuesta (velas)",
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500
    )

    return fig


def create_asymmetry_chart(results: Dict[str, ImpactResult]) -> go.Figure:
    """
    Gráfico comparando respuestas en subidas vs bajadas (asimetría)
    """
    assets = list(results.keys())
    mean_up = [abs(results[asset].mean_response_up) * 100 for asset in assets]
    mean_down = [abs(results[asset].mean_response_down) * 100 for asset in assets]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Respuesta a Subidas',
        x=assets,
        y=mean_up,
        marker_color=COLOR_POSITIVE,
        text=[f"{m:.2f}%" for m in mean_up],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Respuesta a Bajadas',
        x=assets,
        y=mean_down,
        marker_color=COLOR_NEGATIVE,
        text=[f"{m:.2f}%" for m in mean_down],
        textposition='outside'
    ))

    fig.update_layout(
        title="Asimetría de Respuestas<br>"
              "<sub>Magnitud promedio de respuesta a eventos de referencia</sub>",
        xaxis_title="Activos",
        yaxis_title="Magnitud de Respuesta (%)",
        barmode='group',
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white'),
        height=500
    )

    return fig


def create_pattern_detection_summary(results: Dict[str, ImpactResult],
                                     reference_asset: str) -> go.Figure:
    """
    Dashboard de patrones detectados
    """
    from plotly.subplots import make_subplots

    assets = list(results.keys())

    # Preparar datos
    overreaction_scores = [results[asset].overreaction_score for asset in assets]
    asymmetry_ratios = [results[asset].asymmetry_ratio for asset in assets]
    betas = [results[asset].beta for asset in assets]
    persistence_avg = [(results[asset].persistence_up + results[asset].persistence_down) / 2
                       for asset in assets]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Overreaction Score',
            'Asimetría (Bajadas/Subidas)',
            'Beta vs Persistencia',
            'Clasificación de Patrones'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    # 1. Overreaction Score
    colors_over = [COLOR_NEGATIVE if s > 1.5 else COLOR_NEUTRAL if s > 1.2 else COLOR_POSITIVE
                   for s in overreaction_scores]
    fig.add_trace(
        go.Bar(x=assets, y=overreaction_scores, marker_color=colors_over,
               name='Overreaction', showlegend=False),
        row=1, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", row=1, col=1)

    # 2. Asimetría
    colors_asym = [COLOR_NEGATIVE if r > 1.2 else COLOR_NEUTRAL if r > 0.8 else COLOR_POSITIVE
                   for r in asymmetry_ratios]
    fig.add_trace(
        go.Bar(x=assets, y=asymmetry_ratios, marker_color=colors_asym,
               name='Asimetría', showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=1.0, line_dash="dash", row=1, col=2)

    # 3. Beta vs Persistencia
    fig.add_trace(
        go.Scatter(
            x=betas,
            y=persistence_avg,
            mode='markers+text',
            text=assets,
            textposition='top center',
            marker=dict(size=12, color=COLOR_NEUTRAL),
            name='Activos',
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Clasificación
    pattern_labels = []
    for asset in assets:
        r = results[asset]
        if r.overreaction_score > 1.5:
            pattern_labels.append('Overreaction')
        elif r.asymmetry_ratio > 1.2:
            pattern_labels.append('Asimetría Bajista')
        elif r.beta > 1.3:
            pattern_labels.append('Alta Sensibilidad')
        elif r.persistence_up > 70 and r.persistence_down > 70:
            pattern_labels.append('Alta Persistencia')
        else:
            pattern_labels.append('Normal')

    pattern_counts = pd.Series(pattern_labels).value_counts()
    fig.add_trace(
        go.Bar(x=pattern_counts.index, y=pattern_counts.values,
               marker_color=COLOR_POSITIVE, showlegend=False),
        row=2, col=2
    )

    fig.update_layout(
        title_text=f"Dashboard de Patrones Detectados - Referencia: {reference_asset}",
        height=800,
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color='white')
    )

    fig.update_xaxes(title_text="Activos", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Activos", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Beta", row=2, col=1)
    fig.update_yaxes(title_text="Persistencia (%)", row=2, col=1)
    fig.update_xaxes(title_text="Tipo de Patrón", row=2, col=2)
    fig.update_yaxes(title_text="Cantidad", row=2, col=2)

    return fig


def create_multi_timeframe_heatmap(z_data: np.ndarray, assets: List[str],
                                   max_lag: int, event_type: str = "Eventos") -> go.Figure:
    """
    Heatmap simplificado para análisis multi-timeframe

    Args:
        z_data: Matriz de datos [n_assets x max_lag]
        assets: Lista de nombres de activos
        max_lag: Número máximo de lags
        event_type: Descripción del tipo de eventos
    """

    if z_data.size == 0 or not assets:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>⚠️ No hay datos para el heatmap</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor=BG_COLOR
        )
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=list(range(1, max_lag + 1)),
        y=assets,
        colorscale='RdBu',
        zmid=0,
        hoverinfo='z',
        hovertemplate=(
            'Activo: %{y}<br>'
            'Lag: %{x}<br>'
            'Retorno: %{z:.4f}<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title=f"Respuesta a {event_type}",
        xaxis_title="Lag (períodos posteriores)",
        yaxis_title="Activos",
        height=400 + len(assets) * 20,
        template="plotly_dark",
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR
    )

    return fig