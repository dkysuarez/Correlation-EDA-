import plotly.graph_objects as go
import numpy as np
import polars as pl
from typing import List, Dict
import pandas as pd

# Try to import igraph for optimized network graphs
try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    import networkx as nx

# Constants
COLOR_SCALE = [
    [0.0, '#FF0000'],
    [0.3, '#FF6B6B'],
    [0.5, '#2D3038'],
    [0.7, '#4ECDC4'],
    [1.0, '#00C896']
]
BAR_COLORS_POSITIVE = '#00C896'
BAR_COLORS_NEGATIVE = '#FF6B6B'
BACKGROUND_COLOR = '#0E1117'
PLOT_BACKGROUND_COLOR = '#1E2128'
GRID_COLOR = '#2D3038'
TEXT_COLOR = '#FFFFFF'


# =============================================================================
# HELPER FUNCTIONS - CORREGIDAS
# =============================================================================

def _create_empty_figure(message: str, height: int = 400) -> go.Figure:
    """Crea figura vacía con mensaje informativo"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>⚠️ {message}</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="white"),
        bordercolor="#FF6B6B",
        borderwidth=2,
        borderpad=10,
        bgcolor="#1E2128"
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        plot_bgcolor=PLOT_BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    return fig


def _add_responsive_config(fig: go.Figure) -> go.Figure:
    """Agrega configuración responsive - CORREGIDA Y SIMPLIFICADA"""
    # En Streamlit, la responsividad se maneja automáticamente con use_container_width=True
    # Solo agregamos configuración básica sin causar conflictos
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    return fig


def _get_correlation_intensity(corr_value: float) -> str:
    """Clasifica la intensidad de la correlación"""
    abs_corr = abs(corr_value)
    if abs_corr > 0.7:
        return "🔥 Muy Fuerte"
    elif abs_corr > 0.5:
        return "💪 Fuerte"
    elif abs_corr > 0.3:
        return "📊 Moderada"
    elif abs_corr > 0.1:
        return "📈 Débil"
    else:
        return "📉 Muy Débil"


def _is_statistically_significant(p_value: float) -> str:
    """Determina si un p-valor es estadísticamente significativo"""
    if p_value < 0.01:
        return "✅ Muy Significativo (p < 0.01)"
    elif p_value < 0.05:
        return "✅ Significativo (p < 0.05)"
    elif p_value < 0.1:
        return "⚠️ Marginalmente Significativo (p < 0.1)"
    else:
        return "❌ No Significativo (p ≥ 0.1)"


# =============================================================================
# MAIN VISUALIZATION FUNCTIONS - CORREGIDAS
# =============================================================================

def create_heatmap(corr_matrix, p_matrix, asset_names, n_assets_to_show):
    """Crea heatmap de correlaciones con manejo de errores"""
    # Validación de entrada
    if (corr_matrix is None or p_matrix is None or
            corr_matrix.size == 0 or len(asset_names) == 0):
        return _create_empty_figure("No hay datos para mostrar el heatmap")

    if n_assets_to_show < 2:
        return _create_empty_figure("Se necesitan al menos 2 activos para el heatmap")

    try:
        corr_subset = corr_matrix[:n_assets_to_show, :n_assets_to_show]
        p_subset = p_matrix[:n_assets_to_show, :n_assets_to_show]
        asset_subset = asset_names[:n_assets_to_show]

        # Crear texto hover con más información estadística
        hover_text = [[
            f"<b>{asset_subset[i]} vs {asset_subset[j]}</b><br>"
            f"Correlación: {corr_subset[i][j]:.3f}<br>"
            f"P-valor: {p_subset[i][j]:.3e}<br>"
            f"Significancia: {_is_statistically_significant(p_subset[i][j])}<br>"
            f"Intensidad: {_get_correlation_intensity(corr_subset[i][j])}"
            for j in range(n_assets_to_show)
        ] for i in range(n_assets_to_show)]

        fig = go.Figure(data=go.Heatmap(
            z=corr_subset,
            x=asset_subset,
            y=asset_subset,
            colorscale=COLOR_SCALE,
            zmin=-1,
            zmax=1,
            text=hover_text,
            texttemplate="",  # No mostrar texto en celdas, solo hover
            hoverinfo="text",
            showscale=True,
            hovertemplate='%{text}<extra></extra>'
        ))

        fig.update_layout(
            title="Matriz de Correlaciones",
            height=600,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(tickangle=45, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error creando heatmap: {str(e)}")


def create_top_pairs_bar(corr_matrix, p_matrix, asset_names, n_pairs_to_show):
    """Crea gráfico de barras de top pares con p-values mejorados"""
    # Validación de entrada
    if (corr_matrix is None or p_matrix is None or
            len(asset_names) < 2 or n_pairs_to_show < 1):
        empty_fig = _create_empty_figure("Datos insuficientes para mostrar pares")
        return empty_fig, []

    try:
        pairs = []
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                pairs.append({
                    'pair': f"{asset_names[i]}-{asset_names[j]}",
                    'asset1': asset_names[i],
                    'asset2': asset_names[j],
                    'correlation': corr_matrix[i, j],
                    'p_value': p_matrix[i, j]
                })

        pairs_df = pl.DataFrame(pairs)

        if pairs_df.is_empty():
            empty_fig = _create_empty_figure("No se encontraron pares válidos")
            return empty_fig, []

        top_pairs = pairs_df.sort("correlation", descending=True).head(n_pairs_to_show)

        # Preparar datos para hover con más información
        hover_texts = []
        customdata_list = []

        for pair in top_pairs.to_dicts():
            significance = _is_statistically_significant(pair['p_value'])
            intensity = _get_correlation_intensity(pair['correlation'])

            hover_texts.append(
                f"<b>{pair['pair']}</b><br>"
                f"Correlación: {pair['correlation']:.3f}<br>"
                f"P-valor: {pair['p_value']:.3e}<br>"
                f"Significancia: {significance}<br>"
                f"Intensidad: {intensity}"
            )
            customdata_list.append([pair['p_value'], significance])

        fig = go.Figure(data=go.Bar(
            x=top_pairs["pair"].to_list(),
            y=top_pairs["correlation"].to_list(),
            marker_color=[BAR_COLORS_POSITIVE if x >= 0 else BAR_COLORS_NEGATIVE
                          for x in top_pairs["correlation"]],
            marker_line=dict(color='#2D3038', width=1),
            hovertemplate='%{customdata[1]}<extra></extra>',
            customdata=customdata_list,
            text=hover_texts
        ))

        fig.update_layout(
            title=f"Top {n_pairs_to_show} Pares por Correlación",
            xaxis_title="Pares de Activos",
            yaxis_title="Correlación",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(tickangle=45, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, range=[-1, 1])
        )

        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        # Preparar datos para display
        display_data = []
        for pair in top_pairs.to_dicts():
            correlation = pair['correlation']
            intensity = _get_correlation_intensity(correlation)
            significance = _is_statistically_significant(pair['p_value'])

            display_data.append({
                'Par': pair['pair'],
                'Activo 1': pair['asset1'],
                'Activo 2': pair['asset2'],
                'Correlación': f"{correlation:.3f}",
                'P-valor': f"{pair['p_value']:.3e}",
                'Intensidad': intensity,
                'Significancia': significance
            })

        return _add_responsive_config(fig), display_data

    except Exception as e:
        empty_fig = _create_empty_figure(f"Error creando gráfico de pares: {str(e)}")
        return empty_fig, []


def create_distribution_histogram(corr_values, correlation_method):
    """Crea histograma de distribución de correlaciones"""
    # Validación de entrada
    if corr_values is None or len(corr_values) == 0:
        return _create_empty_figure("No hay datos para el histograma")

    try:
        # Filtrar valores no finitos
        corr_values_clean = np.array(corr_values)
        corr_values_clean = corr_values_clean[np.isfinite(corr_values_clean)]

        if len(corr_values_clean) == 0:
            return _create_empty_figure("No hay valores válidos para el histograma")

        fig = go.Figure(data=[go.Histogram(
            x=corr_values_clean,
            nbinsx=30,
            marker_color=BAR_COLORS_POSITIVE,
            marker_line=dict(color='#2D3038', width=1),
            opacity=0.8,
            hovertemplate=(
                "Rango: %{x} <br>"
                "Frecuencia: %{y} pares<br>"
                "<extra></extra>"
            )
        )])

        # Calcular estadísticas
        mean_val = np.mean(corr_values_clean)
        median_val = np.median(corr_values_clean)
        std_val = np.std(corr_values_clean)

        # Agregar líneas de referencia
        fig.add_vline(x=mean_val, line_dash="dash", line_color="yellow",
                      annotation_text=f"Media: {mean_val:.3f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="orange",
                      annotation_text=f"Mediana: {median_val:.3f}")
        fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=2)

        # Agregar áreas de significancia
        fig.add_vrect(x0=-0.1, x1=0.1,
                      fillcolor="gray", opacity=0.2, line_width=0,
                      annotation_text="Débil")
        fig.add_vrect(x0=0.5, x1=1.0,
                      fillcolor="green", opacity=0.1, line_width=0,
                      annotation_text="Fuerte +")
        fig.add_vrect(x0=-1.0, x1=-0.5,
                      fillcolor="red", opacity=0.1, line_width=0,
                      annotation_text="Fuerte -")

        fig.update_layout(
            title=f"Distribución de Correlaciones ({correlation_method})<br>"
                  f"μ={mean_val:.3f}, σ={std_val:.3f}, n={len(corr_values_clean):,} pares",
            xaxis_title="Valor de Correlación",
            yaxis_title="Frecuencia (Número de Pares)",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor=GRID_COLOR, range=[-1, 1]),
            yaxis=dict(gridcolor=GRID_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error creando histograma: {str(e)}")


def create_asset_correlations_bar(corr_matrix, p_matrix, asset_names, selected_asset, n_corrs_to_show):
    """Crea gráfico de barras de correlaciones para un activo específico"""
    # Validación de entrada
    if (corr_matrix is None or p_matrix is None or
            not asset_names or selected_asset not in asset_names):
        empty_fig = _create_empty_figure("Datos insuficientes o activo no encontrado")
        return empty_fig, []

    try:
        asset_idx = asset_names.index(selected_asset)
        asset_correlations = corr_matrix[asset_idx]
        asset_p_values = p_matrix[asset_idx]

        other_correlations = []
        other_p_values = []
        other_assets = []

        for i in range(len(asset_names)):
            if i != asset_idx:
                other_correlations.append(asset_correlations[i])
                other_p_values.append(asset_p_values[i])
                other_assets.append(asset_names[i])

        if not other_assets:
            empty_fig = _create_empty_figure(f"No hay otros activos para comparar con {selected_asset}")
            return empty_fig, []

        asset_data = list(zip(other_assets, other_correlations, other_p_values))
        asset_data.sort(key=lambda x: abs(x[1]), reverse=True)
        top_n = asset_data[:n_corrs_to_show]

        assets_top = [x[0] for x in top_n]
        corrs_top = [x[1] for x in top_n]
        p_vals_top = [x[2] for x in top_n]

        colors = [BAR_COLORS_POSITIVE if x >= 0 else BAR_COLORS_NEGATIVE for x in corrs_top]

        # Preparar hover information
        hover_texts = []
        customdata_list = []

        for asset, corr, p_val in top_n:
            significance = _is_statistically_significant(p_val)
            intensity = _get_correlation_intensity(corr)

            hover_texts.append(
                f"<b>{asset}</b><br>"
                f"Correlación con {selected_asset}: {corr:.3f}<br>"
                f"P-valor: {p_val:.3e}<br>"
                f"Significancia: {significance}<br>"
                f"Intensidad: {intensity}"
            )
            customdata_list.append([p_val, significance])

        fig = go.Figure(data=go.Bar(
            x=assets_top,
            y=corrs_top,
            marker_color=colors,
            marker_line=dict(color='#2D3038', width=1),
            hovertemplate='%{customdata[1]}<extra></extra>',
            customdata=customdata_list,
            text=hover_texts
        ))

        fig.update_layout(
            title=f"Top {n_corrs_to_show} Correlaciones para {selected_asset}",
            xaxis_title="Activos",
            yaxis_title=f"Correlación con {selected_asset}",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(tickangle=45, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, range=[-1, 1])
        )

        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        # Preparar datos para display
        display_data = []
        for asset, corr, p_val in top_n:
            intensity = _get_correlation_intensity(corr)
            significance = _is_statistically_significant(p_val)

            display_data.append({
                'Activo': asset,
                'Correlación': f"{corr:.3f}",
                'P-valor': f"{p_val:.3e}",
                'Intensidad': intensity,
                'Significancia': significance
            })

        return _add_responsive_config(fig), display_data

    except Exception as e:
        empty_fig = _create_empty_figure(f"Error creando gráfico de activo: {str(e)}")
        return empty_fig, []


def create_network_graph(corr_matrix, asset_names, corr_threshold, seed: int = 42):
    """Crea gráfico de red de correlaciones con optimización para grandes datasets - CORREGIDA"""
    # Validación de entrada
    if (corr_matrix is None or not asset_names or len(asset_names) < 2):
        return _create_empty_figure("Datos insuficientes para crear red")

    try:
        # Usar igraph para datasets grandes (>50 activos) si está disponible
        if len(asset_names) > 50 and IGRAPH_AVAILABLE:
            return _create_network_igraph(corr_matrix, asset_names, corr_threshold, seed)
        else:
            return _create_network_nx(corr_matrix, asset_names, corr_threshold, seed)

    except Exception as e:
        return _create_empty_figure(f"Error creando red: {str(e)}")


def _create_network_nx(corr_matrix, asset_names, corr_threshold, seed: int = 42):
    """Versión usando NetworkX (para datasets pequeños) - CORREGIDA"""
    G = nx.Graph()
    for asset in asset_names:
        G.add_node(asset)

    # Agregar edges basados en correlación
    for i in range(len(asset_names)):
        for j in range(i + 1, len(asset_names)):
            corr = corr_matrix[i, j]
            if abs(corr) > corr_threshold:
                G.add_edge(asset_names[i], asset_names[j], weight=corr,
                           correlation=corr, abs_correlation=abs(corr))

    if len(G.edges()) == 0:
        return _create_empty_figure(
            f"No hay conexiones fuertes (umbral: {corr_threshold}). "
            f"Intenta reducir el umbral."
        )

    # ✅ CORRECCIÓN: Layout con seed parametrizable
    pos = nx.spring_layout(G, seed=seed, k=1 / np.sqrt(len(G.nodes())), iterations=50)

    # Separar edges positivos y negativos
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    # Trazas para edges positivos
    edge_x_pos, edge_y_pos = [], []
    for edge in positive_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x_pos.extend([x0, x1, None])
        edge_y_pos.extend([y0, y1, None])

    edge_trace_pos = go.Scatter(
        x=edge_x_pos, y=edge_y_pos,
        line=dict(width=1.5, color=BAR_COLORS_POSITIVE),
        hoverinfo='none',
        mode='lines',
        name='Correlación Positiva'
    )

    # Trazas para edges negativos
    edge_x_neg, edge_y_neg = [], []
    for edge in negative_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x_neg.extend([x0, x1, None])
        edge_y_neg.extend([y0, y1, None])

    edge_trace_neg = go.Scatter(
        x=edge_x_neg, y=edge_y_neg,
        line=dict(width=1.5, color=BAR_COLORS_NEGATIVE),
        hoverinfo='none',
        mode='lines',
        name='Correlación Negativa'
    )

    # Traza para nodos
    node_x, node_y, node_text, node_hover = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Calcular grado y conexiones para hover
        degree = G.degree(node)
        connections = [n for n in G.neighbors(node)]
        node_hover.append(
            f"<b>{node}</b><br>"
            f"Grado: {degree}<br>"
            f"Conexiones: {', '.join(connections[:5])}{'...' if len(connections) > 5 else ''}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        hovertext=node_hover,
        hoverinfo='text',
        marker=dict(
            size=15,
            color=BAR_COLORS_POSITIVE,
            line=dict(width=2, color='#2D3038')
        ),
        name='Activos'
    )

    fig = go.Figure(data=[edge_trace_pos, edge_trace_neg, node_trace],
                    layout=go.Layout(
                        title=f"Red de Correlaciones (Umbral: {corr_threshold})<br>"
                              f"{len(G.nodes())} nodos, {len(G.edges())} conexiones",
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=60),
                        height=600,
                        plot_bgcolor=PLOT_BACKGROUND_COLOR,
                        paper_bgcolor=BACKGROUND_COLOR,
                        font=dict(color=TEXT_COLOR),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    ))

    return _add_responsive_config(fig)


def _create_network_igraph(corr_matrix, asset_names, corr_threshold, seed: int = 42):
    """Versión optimizada usando igraph (para datasets grandes) - CORREGIDA"""
    # Crear grafo con igraph
    G = ig.Graph()
    G.add_vertices(len(asset_names))
    G.vs["name"] = asset_names

    # Agregar edges
    edges = []
    edge_weights = []
    edge_correlations = []

    for i in range(len(asset_names)):
        for j in range(i + 1, len(asset_names)):
            corr = corr_matrix[i, j]
            if abs(corr) > corr_threshold:
                edges.append((i, j))
                edge_weights.append(abs(corr))
                edge_correlations.append(corr)

    if not edges:
        return _create_empty_figure(
            f"No hay conexiones fuertes (umbral: {corr_threshold}). "
            f"Intenta reducir el umbral."
        )

    G.add_edges(edges)
    G.es["weight"] = edge_weights
    G.es["correlation"] = edge_correlations

    # ✅ CORRECCIÓN: Layout con seed para igraph
    try:
        # igraph usa random seed global, así que lo configuramos
        import random
        random.seed(seed)
        np.random.seed(seed)

        layout = G.layout_fruchterman_reingold(weights=edge_weights, niter=50)
    except:
        # Fallback si hay problemas con la seed
        layout = G.layout_fruchterman_reingold(weights=edge_weights, niter=50)

    # Convertir a coordenadas para plotly
    node_x = [coord[0] for coord in layout]
    node_y = [coord[1] for coord in layout]

    # Separar edges positivos y negativos
    positive_edges = [(G.es[edge].source, G.es[edge].target)
                      for edge in range(len(G.es)) if G.es[edge]["correlation"] >= 0]
    negative_edges = [(G.es[edge].source, G.es[edge].target)
                      for edge in range(len(G.es)) if G.es[edge]["correlation"] < 0]

    # Trazas para edges
    edge_traces = []

    # Edges positivos
    edge_x_pos, edge_y_pos = [], []
    for edge in positive_edges:
        x0, y0 = node_x[edge[0]], node_y[edge[0]]
        x1, y1 = node_x[edge[1]], node_y[edge[1]]
        edge_x_pos.extend([x0, x1, None])
        edge_y_pos.extend([y0, y1, None])

    if edge_x_pos:
        edge_traces.append(go.Scatter(
            x=edge_x_pos, y=edge_y_pos,
            line=dict(width=1, color=BAR_COLORS_POSITIVE),
            hoverinfo='none',
            mode='lines',
            name='Correlación Positiva'
        ))

    # Edges negativos
    edge_x_neg, edge_y_neg = [], []
    for edge in negative_edges:
        x0, y0 = node_x[edge[0]], node_y[edge[0]]
        x1, y1 = node_x[edge[1]], node_y[edge[1]]
        edge_x_neg.extend([x0, x1, None])
        edge_y_neg.extend([y0, y1, None])

    if edge_x_neg:
        edge_traces.append(go.Scatter(
            x=edge_x_neg, y=edge_y_neg,
            line=dict(width=1, color=BAR_COLORS_NEGATIVE),
            hoverinfo='none',
            mode='lines',
            name='Correlación Negativa'
        ))

    # Traza para nodos
    node_hover = []
    for node in range(len(asset_names)):
        neighbors = G.neighbors(node)
        node_hover.append(
            f"<b>{asset_names[node]}</b><br>"
            f"Grado: {len(neighbors)}<br>"
            f"Conexiones: {', '.join([asset_names[n] for n in neighbors[:3]])}{'...' if len(neighbors) > 3 else ''}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=asset_names,
        textposition="middle center",
        hovertext=node_hover,
        hoverinfo='text',
        marker=dict(
            size=12,
            color=BAR_COLORS_POSITIVE,
            line=dict(width=1.5, color='#2D3038')
        ),
        name='Activos'
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=f"Red de Correlaciones (Umbral: {corr_threshold})<br>"
                              f"{len(asset_names)} nodos, {len(edges)} conexiones (igraph)",
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=60),
                        height=600,
                        plot_bgcolor=PLOT_BACKGROUND_COLOR,
                        paper_bgcolor=BACKGROUND_COLOR,
                        font=dict(color=TEXT_COLOR),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    ))

    return _add_responsive_config(fig)


# =============================================================================
# LEAD-LAG VISUALIZATION FUNCTIONS - CORREGIDAS COMPLETAMENTE
# =============================================================================

def plot_cross_correlation(results: Dict) -> go.Figure:
    """
    Gráfico de cross-correlation vs lags - CORREGIDO
    """
    if not results or 'lags' not in results or 'correlations' not in results:
        return _create_empty_figure("No hay datos de cross-correlation")

    try:
        lags = results['lags']
        correlations = results['correlations']
        optimal_lag = results.get('optimal_lag', 0)

        fig = go.Figure()

        # Línea de correlaciones
        fig.add_trace(go.Scatter(
            x=lags,
            y=correlations,
            mode='lines+markers',
            name='Correlación',
            line=dict(color='#00C896', width=2),
            marker=dict(size=4),
            hovertemplate=(
                "Lag: %{x}<br>"
                "Correlación: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))

        # Línea vertical en lag óptimo
        if optimal_lag != 0:
            fig.add_vline(
                x=optimal_lag,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Lag óptimo: {optimal_lag}",
                annotation_position="top right"
            )

        # Línea en cero
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            title="Cross-Correlation Analysis",
            xaxis_title="Lag (períodos)",
            yaxis_title="Correlación",
            template="plotly_dark",
            height=500,
            showlegend=False
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en cross-correlation: {str(e)}")


def plot_granger_results(results: Dict) -> go.Figure:
    """Gráfico de p-values de Granger Causality - CORREGIDO"""
    if not results or 'p_values_12' not in results or 'p_values_21' not in results:
        return _create_empty_figure("No hay datos de Granger causality")

    try:
        lags = list(range(1, len(results['p_values_12']) + 1))
        p_values_12 = results['p_values_12']
        p_values_21 = results['p_values_21']
        significance_level = 0.05

        fig = go.Figure()

        # Asset1 → Asset2
        fig.add_trace(go.Scatter(
            x=lags,
            y=p_values_12,
            mode='lines+markers',
            name='Asset1 → Asset2',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate=(
                "Lag: %{x}<br>"
                "p-value: %{y:.3e}<br>"
                "<extra></extra>"
            )
        ))

        # Asset2 → Asset1
        fig.add_trace(go.Scatter(
            x=lags,
            y=p_values_21,
            mode='lines+markers',
            name='Asset2 → Asset1',
            line=dict(color='#4ECDC4', width=2),
            hovertemplate=(
                "Lag: %{x}<br>"
                "p-value: %{y:.3e}<br>"
                "<extra></extra>"
            )
        ))

        # Línea de significancia
        fig.add_hline(
            y=significance_level,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Significancia (p={significance_level})",
            annotation_position="bottom right"
        )

        # Destacar lags significativos
        for lag in results.get('significant_lags_12', []):
            fig.add_vline(x=lag, line_dash="dot", line_color="#FF6B6B", opacity=0.5)

        for lag in results.get('significant_lags_21', []):
            fig.add_vline(x=lag, line_dash="dot", line_color="#4ECDC4", opacity=0.5)

        fig.update_layout(
            title="Granger Causality Analysis",
            xaxis_title="Lag",
            yaxis_title="p-value",
            yaxis_type="log",
            template="plotly_dark",
            height=500
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en Granger causality: {str(e)}")


def create_lead_lag_summary(results: Dict, asset1_name: str, asset2_name: str) -> go.Figure:
    """Crear resumen visual de resultados lead-lag - CORREGIDO"""
    if not results:
        return _create_empty_figure("No hay resultados de lead-lag")

    try:
        method = results['method']

        if method == 'cross_correlation':
            title = f"Cross-Correlation: {asset1_name} vs {asset2_name}"
            interpretation = results.get('interpretation', 'No disponible')
            correlation = results.get('max_correlation', 0)

            fig = go.Figure()

            fig.add_annotation(
                text=f"<b>{interpretation}</b><br>Correlación: {correlation:.3f}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white"),
                align="center",
                bordercolor="#00C896",
                borderwidth=2,
                borderpad=10,
                bgcolor="#1E2128"
            )

        else:  # granger_causality
            title = f"Granger Causality: {asset1_name} vs {asset2_name}"
            interpretation = results.get('interpretation', 'No disponible')

            fig = go.Figure()

            fig.add_annotation(
                text=f"<b>{interpretation}</b>",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white"),
                align="center",
                bordercolor="#4ECDC4",
                borderwidth=2,
                borderpad=10,
                bgcolor="#1E2128"
            )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=200,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en resumen lead-lag: {str(e)}")


def plot_hub_network(hub_results: Dict, corr_threshold: float = 0.5, seed: int = 42) -> go.Figure:
    """Gráfico de red para hubs de correlación - CORREGIDO"""
    if not hub_results or 'hubs' not in hub_results:
        return _create_empty_figure("No hay datos de hubs")

    try:
        hubs = hub_results['hubs']

        if not hubs:
            return _create_empty_figure("No se encontraron hubs significativos")

        # Crear grafo de hubs
        G = nx.Graph()

        for hub in hubs:
            G.add_node(hub['asset'], size=hub['hub_score'], connections=hub['n_connections'])
            for connection in hub.get('strong_connections', []):
                if abs(connection['correlation']) > corr_threshold:
                    G.add_edge(hub['asset'], connection['asset'],
                               weight=abs(connection['correlation']),
                               correlation=connection['correlation'])

        if len(G.nodes()) == 0:
            return _create_empty_figure("No hay nodos en la red de hubs")

        # ✅ CORRECCIÓN: Layout con seed parametrizable
        pos = nx.spring_layout(G, seed=seed, k=1, iterations=50)

        # Edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#666666'),
            hoverinfo='none',
            mode='lines'
        )

        # Nodes
        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_size.append(G.nodes[node]['size'] * 30 + 10)
            node_color.append(G.nodes[node]['size'])

            connections = G.nodes[node]['connections']
            node_text.append(
                f"<b>{node}</b><br>"
                f"Hub Score: {G.nodes[node]['size']:.3f}<br>"
                f"Conexiones: {connections}"
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Hub Score"),
                line=dict(width=2, color='#2D3038')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"Red de Hubs de Correlación (Umbral: {corr_threshold})",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=60),
                            height=600,
                            plot_bgcolor=PLOT_BACKGROUND_COLOR,
                            paper_bgcolor=BACKGROUND_COLOR,
                            font=dict(color=TEXT_COLOR),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en red de hubs: {str(e)}")


def plot_hub_strength_analysis(hub_results: Dict) -> go.Figure:
    """Análisis de fuerza de hubs - CORREGIDO"""
    if not hub_results or 'hubs' not in hub_results:
        return _create_empty_figure("No hay datos de hubs")

    try:
        hubs = hub_results['hubs']

        if not hubs:
            return _create_empty_figure("No se encontraron hubs")

        assets = [hub['asset'] for hub in hubs]
        hub_scores = [hub['hub_score'] for hub in hubs]
        connections = [hub['n_connections'] for hub in hubs]

        fig = go.Figure(data=[
            go.Bar(name='Hub Score', x=assets, y=hub_scores, marker_color='#00C896'),
            go.Bar(name='Conexiones', x=assets, y=connections, marker_color='#4ECDC4')
        ])

        fig.update_layout(
            title="Análisis de Fuerza de Hubs",
            xaxis_title="Activos",
            yaxis_title="Score / Conexiones",
            barmode='group',
            template="plotly_dark",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(tickangle=45, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en análisis de hubs: {str(e)}")


def create_hub_summary_dashboard(hub_results: Dict) -> go.Figure:
    """Dashboard resumen de hubs - CORREGIDO"""
    if not hub_results or 'hubs' not in hub_results:
        return _create_empty_figure("No hay datos de hubs")

    try:
        hubs = hub_results['hubs']
        summary = hub_results.get('summary', {})

        if not hubs:
            return _create_empty_figure("No se encontraron hubs significativos")

        # Crear figura con múltiples subplots
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top Hubs por Score',
                'Distribución de Conexiones',
                'Correlación Score vs Conexiones',
                'Resumen de Hubs'
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "domain"}]
            ]
        )

        # Top Hubs
        top_hubs = sorted(hubs, key=lambda x: x['hub_score'], reverse=True)[:10]
        fig.add_trace(
            go.Bar(x=[h['asset'] for h in top_hubs],
                   y=[h['hub_score'] for h in top_hubs],
                   marker_color='#00C896'),
            row=1, col=1
        )

        # Distribución de conexiones
        connections = [h['n_connections'] for h in hubs]
        fig.add_trace(
            go.Histogram(x=connections, nbinsx=15, marker_color='#4ECDC4'),
            row=1, col=2
        )

        # Scatter plot
        fig.add_trace(
            go.Scatter(x=[h['hub_score'] for h in hubs],
                       y=[h['n_connections'] for h in hubs],
                       mode='markers',
                       marker=dict(size=8, color='#FF6B6B'),
                       text=[h['asset'] for h in hubs],
                       hovertemplate='<b>%{text}</b><br>Score: %{x}<br>Conexiones: %{y}<extra></extra>'),
            row=2, col=1
        )

        # Pie chart de hubs por categoría de fuerza
        strong_hubs = len([h for h in hubs if h['hub_score'] > 0.7])
        medium_hubs = len([h for h in hubs if 0.4 <= h['hub_score'] <= 0.7])
        weak_hubs = len([h for h in hubs if h['hub_score'] < 0.4])

        fig.add_trace(
            go.Pie(labels=['Fuertes', 'Medios', 'Débiles'],
                   values=[strong_hubs, medium_hubs, weak_hubs],
                   marker=dict(colors=['#00C896', '#4ECDC4', '#FF6B6B'])),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Dashboard de Análisis de Hubs",
            height=700,
            template="plotly_dark",
            showlegend=True,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en dashboard de hubs: {str(e)}")


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN PARA ANÁLISIS TEMPORAL - CORREGIDAS
# =============================================================================

def create_correlation_timeline_plot(timeline_data: Dict, asset1: str, asset2: str) -> go.Figure:
    """Crea gráfico de línea temporal de correlaciones - CORREGIDO"""
    if not timeline_data or 'timeline' not in timeline_data:
        return _create_empty_figure("No hay datos de timeline")

    try:
        timeline = timeline_data['timeline']
        dates = [point['date'] for point in timeline]
        correlations = [point['correlation'] for point in timeline]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=correlations,
            mode='lines',
            name='Correlación',
            line=dict(color='#00C896', width=2),
            hovertemplate=(
                "Fecha: %{x}<br>"
                "Correlación: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))

        # Línea de cero
        fig.add_hline(y=0, line_dash="dot", line_color="white")

        fig.update_layout(
            title=f"Evolución Temporal de Correlación: {asset1} vs {asset2}",
            xaxis_title="Fecha",
            yaxis_title="Correlación",
            template="plotly_dark",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en timeline: {str(e)}")


def create_volatility_plot(volatility_data: Dict) -> go.Figure:
    """Crea gráfico de volatilidad comparativa - CORREGIDO"""
    if not volatility_data:
        return _create_empty_figure("No hay datos de volatilidad")

    try:
        fig = go.Figure()

        for asset, data in volatility_data.items():
            if 'dates' in data and 'volatility' in data:
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['volatility'],
                    mode='lines',
                    name=asset,
                    hovertemplate=(
                        f"Asset: {asset}<br>"
                        "Fecha: %{x}<br>"
                        "Volatilidad: %{y:.4f}<br>"
                        "<extra></extra>"
                    )
                ))

        fig.update_layout(
            title="Comparación de Volatilidad entre Activos",
            xaxis_title="Fecha",
            yaxis_title="Volatilidad (Desviación Estándar)",
            template="plotly_dark",
            height=500,
            plot_bgcolor=PLOT_BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR)
        )

        return _add_responsive_config(fig)

    except Exception as e:
        return _create_empty_figure(f"Error en gráfico de volatilidad: {str(e)}")