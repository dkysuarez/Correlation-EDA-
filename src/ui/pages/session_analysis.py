import streamlit as st
import numpy as np
import polars as pl
from core.data_loader import BatchCorrelationLoader
from ui.components import render_sidebar
from utils.visualization import create_heatmap, create_distribution_histogram
from utils.session_analyzer import SessionCorrelationAnalyzer
import plotly.graph_objects as go
import pandas as pd


def render():
    st.markdown("## 🌍 Análisis de Correlaciones por Sesiones de Trading")

    # Reutilizar sidebar existente
    selected_assets, correlation_method, timeframe, batch_size, load_button = render_sidebar()

    if 'session_data_loaded' not in st.session_state:
        st.session_state.session_data_loaded = False
        st.session_state.session_results = None
        st.session_state.session_asset_names = None
        st.session_state.session_failed_pairs = []

    if load_button and selected_assets:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Cargando activos para análisis por sesiones...")
            progress_bar.progress(30)

            # Cargar datos manteniendo todas las fechas
            loader = BatchCorrelationLoader()
            assets_data = loader.load_assets_for_session_analysis(selected_assets, timeframe)
            progress_bar.progress(60)

            if len(assets_data) >= 2:
                status_text.text(f"📊 Calculando correlaciones por sesión ({correlation_method})...")
                progress_bar.progress(80)

                # Calcular correlaciones por sesión
                analyzer = SessionCorrelationAnalyzer()
                session_results = analyzer.calculate_session_correlations(assets_data, correlation_method)
                progress_bar.progress(100)

                # Extraer información de pares fallidos
                failed_pairs_info = []
                for session_name, (corr_matrix, p_matrix, assets) in session_results.items():
                    if corr_matrix is None and len(assets) >= 2:
                        failed_pairs_info.append({
                            'Sesión': session_name,
                            'Activos': len(assets),
                            'Razón': 'No se pudo calcular matriz de correlación',
                            'Detalle': 'Problemas en combinación de datos temporales'
                        })

                st.session_state.update({
                    'session_results': session_results,
                    'session_asset_names': list(assets_data.keys()),
                    'session_data_loaded': True,
                    'session_correlation_method': correlation_method,
                    'session_failed_pairs': failed_pairs_info
                })

                status_text.text("✅ Análisis por sesiones completado")
                st.success(f"Procesados {len(assets_data)} activos en 3 sesiones con {correlation_method}")

                # Mostrar resumen rápido
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Activos", len(assets_data))
                with col2:
                    st.metric("Sesiones", 3)
                with col3:
                    # ✅ CORRECCIÓN: Calcular pares únicos correctamente
                    n_assets = len(assets_data)
                    unique_pairs = n_assets * (n_assets - 1) // 2
                    total_pairs_all_sessions = unique_pairs * 3
                    st.metric("Pares Totales", f"{total_pairs_all_sessions:,}")

                    # Mostrar desglose en tooltip
                    st.caption(f"({unique_pairs} pares × 3 sesiones)")

            else:
                st.error("No hay suficientes activos cargados. Selecciona al menos 2 activos válidos.")

        except Exception as e:
            st.error(f"Error en procesamiento por sesiones: {str(e)}")

    if st.session_state.get('session_data_loaded', False) and st.session_state.get('session_results'):
        session_results = st.session_state.session_results
        asset_names = st.session_state.session_asset_names
        correlation_method = st.session_state.session_correlation_method
        failed_pairs = st.session_state.get('session_failed_pairs', [])

        st.markdown("---")
        st.markdown("## 📊 Resultados por Sesión")

        # Resumen de sesiones exitosas
        successful_sessions = []
        failed_sessions = []

        for session_name, (corr_matrix, p_matrix, assets) in session_results.items():
            if corr_matrix is not None:
                successful_sessions.append(session_name)
            else:
                failed_sessions.append(session_name)

        # Estadísticas generales
        st.markdown("### 📈 Resumen General")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sesiones Exitosas", len(successful_sessions))
        with col2:
            st.metric("Sesiones Fallidas", len(failed_sessions))
        with col3:
            total_pairs = len(asset_names) * (len(asset_names) - 1) // 2
            st.metric("Pares por Sesión", total_pairs)
        with col4:
            if successful_sessions:
                st.metric("Tasa de Éxito", f"{(len(successful_sessions) / 3) * 100:.1f}%")

        # Mostrar sesiones fallidas si las hay
        if failed_sessions:
            st.warning(f"⚠️ {len(failed_sessions)} sesión(es) sin datos: {', '.join(failed_sessions)}")
            st.info("💡 Esto puede deberse a falta de superposición temporal en los datos para esa sesión")

        # Tabs de visualización
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔥 Heatmaps por Sesión",
            "📊 Comparativa entre Sesiones",
            "📈 Distribuciones",
            "🔍 Análisis por Activo",
            "⚠️ Sesiones Problemáticas"
        ])

        with tab1:
            st.markdown("### 🎨 Matrices de Correlación por Sesión")

            # Solo mostrar sesiones exitosas en el selector
            available_sessions = [s for s in session_results.keys()
                                  if session_results[s][0] is not None]

            if available_sessions:
                session_to_show = st.selectbox(
                    "Seleccionar sesión para visualizar:",
                    options=available_sessions,
                    key="session_heatmap_selector"
                )

                if session_to_show in session_results:
                    corr_matrix, p_matrix, assets = session_results[session_to_show]

                    if corr_matrix is not None and len(assets) >= 2:
                        # Manejo seguro del slider
                        n_assets = len(assets)
                        if n_assets > 2:
                            n_assets_to_show = st.slider(
                                "Número de activos a mostrar:",
                                min_value=2,
                                max_value=n_assets,
                                value=min(20, n_assets),
                                step=1,
                                key="session_heatmap_slider"
                            )
                        else:
                            n_assets_to_show = n_assets
                            st.info(f"📊 Mostrando los {n_assets} activos disponibles")

                        fig = create_heatmap(corr_matrix, p_matrix, assets, n_assets_to_show)
                        fig.update_layout(title=f"Correlaciones - Sesión {session_to_show} ({correlation_method})")
                        st.plotly_chart(fig, use_container_width=True)

                        # Estadísticas de la sesión seleccionada
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        corr_values = corr_matrix[mask]

                        if len(corr_values) > 0:
                            st.markdown("#### 📊 Estadísticas de la Sesión")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                mean_corr = np.mean(corr_values)
                                st.metric("Correlación Media", f"{mean_corr:.3f}")
                            with col2:
                                st.metric("Correlación Máxima", f"{np.max(corr_values):.3f}")
                            with col3:
                                st.metric("Correlación Mínima", f"{np.min(corr_values):.3f}")
                            with col4:
                                positive_ratio = len(corr_values[corr_values > 0.1]) / len(corr_values) * 100
                                st.metric("% Positivas", f"{positive_ratio:.1f}%")

                            # Interpretación de la correlación
                            if abs(mean_corr) < 0.1:
                                st.warning(
                                    "📉 **CORRELACIÓN MUY DÉBIL**: Los activos prácticamente no se mueven juntos en esta sesión")
                            elif abs(mean_corr) < 0.3:
                                st.info("📊 **CORRELACIÓN DÉBIL**: Relación leve entre los activos")
                            elif abs(mean_corr) < 0.5:
                                st.success("💪 **CORRELACIÓN MODERADA**: Relación significativa entre activos")
                            else:
                                st.success("🔥 **CORRELACIÓN FUERTE**: Alta relación entre los movimientos de activos")
                    else:
                        st.warning(f"No hay datos suficientes para la sesión {session_to_show}")
            else:
                st.error("❌ No hay sesiones con datos suficientes para mostrar")
                st.info("💡 Prueba con activos que tengan mayor superposición temporal")

        with tab2:
            st.markdown("### 📊 Comparativa entre Sesiones de Trading")

            if len(successful_sessions) >= 2:
                metric = st.selectbox(
                    "Métrica a comparar:",
                    options=["mean", "max", "min", "positive_ratio", "strong_positive_ratio"],
                    format_func=lambda x: {
                        "mean": "Correlación Promedio",
                        "max": "Correlación Máxima",
                        "min": "Correlación Mínima",
                        "positive_ratio": "% Correlaciones Positivas",
                        "strong_positive_ratio": "% Correlaciones Fuertes Positivas"
                    }[x],
                    key="session_metric_selector"
                )

                analyzer = SessionCorrelationAnalyzer()
                fig = analyzer.create_session_comparison_chart(session_results, metric)
                st.plotly_chart(fig, use_container_width=True)

                # Tabla comparativa detallada
                st.markdown("#### 📋 Datos Comparativos Detallados")
                comparison_data = []
                for session_name in successful_sessions:
                    corr_matrix, p_matrix, assets = session_results[session_name]

                    if corr_matrix is not None:
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        corr_values = corr_matrix[mask]

                        if len(corr_values) > 0:
                            comparison_data.append({
                                'Sesión': session_name,
                                'Activos': len(assets),
                                'Pares Válidos': len(corr_values),
                                'Corr. Media': f"{np.mean(corr_values):.3f}",
                                'Corr. Máx': f"{np.max(corr_values):.3f}",
                                'Corr. Mín': f"{np.min(corr_values):.3f}",
                                '% Positivas': f"{(len(corr_values[corr_values > 0.1]) / len(corr_values)) * 100:.1f}%",
                                '% Fuertes Pos': f"{(len(corr_values[corr_values > 0.5]) / len(corr_values)) * 100:.1f}%"
                            })

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    # Análisis comparativo
                    st.markdown("#### 🎯 Insights Comparativos")
                    if len(comparison_data) > 1:
                        best_session = max(comparison_data, key=lambda x: float(x['Corr. Media']))
                        worst_session = min(comparison_data, key=lambda x: float(x['Corr. Media']))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(
                                f"**Mejor sesión**: {best_session['Sesión']} (Corr: {best_session['Corr. Media']})")
                        with col2:
                            st.warning(
                                f"**Peor sesión**: {worst_session['Sesión']} (Corr: {worst_session['Corr. Media']})")
                else:
                    st.warning("No hay datos comparativos disponibles")
            else:
                st.warning("Se necesitan al menos 2 sesiones exitosas para la comparativa")

        with tab3:
            st.markdown("### 📈 Distribución de Correlaciones por Sesión")

            available_sessions_dist = [s for s in session_results.keys()
                                       if session_results[s][0] is not None]

            if available_sessions_dist:
                session_for_dist = st.selectbox(
                    "Seleccionar sesión:",
                    options=available_sessions_dist,
                    key="session_dist_selector"
                )

                if session_for_dist in session_results:
                    corr_matrix, p_matrix, assets = session_results[session_for_dist]

                    if corr_matrix is not None:
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        corr_values = corr_matrix[mask]

                        if len(corr_values) > 0:
                            fig = create_distribution_histogram(corr_values,
                                                                f"{correlation_method} - {session_for_dist}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No hay valores de correlación para mostrar en {session_for_dist}")
                    else:
                        st.warning(f"No hay matriz de correlación para {session_for_dist}")
            else:
                st.warning("No hay sesiones con datos para mostrar distribuciones")

        with tab4:
            st.markdown("### 🔍 Análisis Detallado por Activo y Sesión")

            selected_asset = st.selectbox(
                "Seleccionar activo:",
                options=asset_names,
                key="session_asset_analyzer"
            )

            if selected_asset:
                # Crear tabla de correlaciones por sesión
                analysis_data = []
                for session_name in successful_sessions:
                    corr_matrix, p_matrix, assets = session_results[session_name]

                    if corr_matrix is not None and selected_asset in assets:
                        asset_idx = assets.index(selected_asset)

                        # Obtener correlaciones de este activo con otros
                        other_correlations = []
                        for i in range(len(assets)):
                            if i != asset_idx:
                                other_correlations.append(corr_matrix[asset_idx, i])

                        if other_correlations:
                            mean_corr = np.mean(other_correlations)
                            analysis_data.append({
                                'Sesión': session_name,
                                'Corr. Promedio': f"{mean_corr:.3f}",
                                'Corr. Máxima': f"{np.max(other_correlations):.3f}",
                                'Corr. Mínima': f"{np.min(other_correlations):.3f}",
                                'Activos Relacionados': len(other_correlations),
                                'Interpretación': (
                                    "MUY DÉBIL" if abs(mean_corr) < 0.1 else
                                    "DÉBIL" if abs(mean_corr) < 0.3 else
                                    "MODERADA" if abs(mean_corr) < 0.5 else
                                    "FUERTE"
                                )
                            })

                if analysis_data:
                    analysis_df = pd.DataFrame(analysis_data)
                    st.dataframe(analysis_df, use_container_width=True)

                    # Gráfico de comportamiento por sesión
                    fig = go.Figure()

                    sessions = [item['Sesión'] for item in analysis_data]
                    avg_corrs = [float(item['Corr. Promedio']) for item in analysis_data]

                    fig.add_trace(go.Scatter(
                        x=sessions,
                        y=avg_corrs,
                        mode='lines+markers',
                        name='Correlación Promedio',
                        line=dict(color='#00C896', width=3),
                        marker=dict(size=10),
                        hovertemplate='<b>%{x}</b><br>Correlación: %{y:.3f}<extra></extra>'
                    ))

                    fig.update_layout(
                        title=f"Comportamiento de {selected_asset} por Sesión",
                        xaxis_title="Sesión de Trading",
                        yaxis_title="Correlación Promedio",
                        template="plotly_dark",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Análisis del patrón
                    if len(analysis_data) >= 2:
                        max_session = max(analysis_data, key=lambda x: float(x['Corr. Promedio']))
                        min_session = min(analysis_data, key=lambda x: float(x['Corr. Promedio']))

                        st.info(
                            f"**Patrón detectado**: Mayor correlación en **{max_session['Sesión']}** ({max_session['Corr. Promedio']}), menor en **{min_session['Sesión']}** ({min_session['Corr. Promedio']})")
                else:
                    st.warning(f"No hay datos de correlación para {selected_asset} en ninguna sesión")

        with tab5:
            st.markdown("### ⚠️ Sesiones y Pares Problemáticos")

            if failed_pairs:
                st.warning(f"Se encontraron {len(failed_pairs)} sesión(es) con problemas:")

                failed_df = pd.DataFrame(failed_pairs)
                st.dataframe(failed_df, use_container_width=True)

                st.markdown("#### 💡 Posibles Soluciones:")
                st.info("""
                1. **Verifica la superposición temporal** entre activos
                2. **Prueba con un timeframe diferente** (1h, 4h, 1d)
                3. **Selecciona activos con períodos similares**
                4. **Aumenta el número de activos** para tener más opciones de combinación
                """)
            else:
                st.success("✅ Todas las sesiones se procesaron correctamente")

    else:
        st.info("""
        🌍 **Sistema de Análisis por Sesiones de Trading**

        **Para comenzar:**
        1. Selecciona los activos en el sidebar (mínimo 2)
        2. Configura método de correlación y timeframe  
        3. Haz click en **"🚀 Cargar y Analizar"**

        **📊 El sistema analizará correlaciones en 3 sesiones:**
        - **Asia**: 00:00-08:00 UTC
        - **Europa**: 08:00-16:00 UTC  
        - **US**: 16:00-24:00 UTC

        **🎯 Insights que podrás descubrir:**
        - ¿En qué sesión las correlaciones son más fuertes?
        - ¿Qué activos se mueven juntos en horario asiático vs US?
        - Patrones temporales en las relaciones entre activos
        - Sesiones problemáticas y posibles causas

        **💡 Tip**: Para mejores resultados, selecciona activos con buena superposición temporal
        """)


if __name__ == "__main__":
    render()