# =====================================================
# app.py – Proyecto Aurelion
# =====================================================
import pandas as pd
import streamlit as st
import numpy as np
import os
from scripts import graficos
from scriptsML import regression, classification, clustering


# -----------------------------------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------------------------------
st.set_page_config(page_title="Proyecto Aurelion", page_icon="🌎", layout="wide")

# -----------------------------------------------------
# ESTILOS PERSONALIZADOS
# -----------------------------------------------------
st.markdown("""
    <style>
    /* Reducción del espacio superior general */
    .st-emotion-cache-1pxn4lb { /* Este selector puede variar. Revisa el dev tools si no funciona. */
        padding-top: 0rem; 
    }
    
    /* Alternativa más genérica (puede requerir prueba y error) */
    div.stApp > section.main {
        padding-top: 0rem;
    }

    body {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: "Inter", sans-serif;
    }
    .titulo {
        text-align: center;
        font-size: 2.9em;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 0.2em;      /* REDUCIDO de 0.9em */
        margin-bottom: 0.1em;   /* REDUCIDO de 0.2em */
    }
    .subtitulo {
        font-size: 1.4em;
        font-weight: 600;
        color: #2563EB;
        margin-top: 0.5em;      /* REDUCIDO de 1.2em */
        margin-bottom: 0.1em;   /* REDUCIDO de 0.3em */
    }
    .descripcion {
        font-size: 1.05em;
        color: #475569;
        margin-bottom: 0.5em;   /* REDUCIDO de 1em para comprimir el texto */
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------
csv_path = "./Base de datos/ventas_completas.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.error(f"⚠️ No se encontró `{csv_path}`.")
    st.stop()

st.markdown('<div class="titulo">📚 Proyecto Aurelion</div>', unsafe_allow_html=True)
st.subheader("Consulta interactiva de documentación y análisis")

# -----------------------------------------------------
# MENÚ LATERAL PRINCIPAL
# -----------------------------------------------------
menu = st.sidebar.radio(
    "Selecciona una sección:",
    ("SPRINT 1 - Docs", "SPRINT 2 - Análisis","SPRINT 3 - Machine Learning" ,"SPRINT 4 - Power BI","Salir")
)

# -----------------------------------------------------
# FUNCIÓN PARA MOSTRAR MARKDOWN
# -----------------------------------------------------
def mostrar_markdown(nombre_archivo):
    try:
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()
            st.markdown(contenido)
    except FileNotFoundError:
        st.error(f"⚠️ No se encontró el archivo `{nombre_archivo}`.")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")

# =====================================================
# 🧾 SPRINT 1 - DOCUMENTACIÓN
# =====================================================
if menu == "SPRINT 1 - Docs":
    st.markdown('<div class="titulo">📊 SPRINT 1 - Documentación</div>', unsafe_allow_html=True)
    st.markdown('<p class="descripcion">Este módulo agrupa la documentación inicial del Proyecto Aurelion.</p>', unsafe_allow_html=True)
    st.divider()
    
    opcion = st.selectbox(
        "Selecciona el archivo que deseas consultar:",
        ("documentacion.md", "dataset.md", "pseudocodigo.md", "copilot.md", "diagrama.drawio.png")
    )

    if opcion.endswith(".md"):
        mostrar_markdown(opcion)

    elif opcion.endswith(".png"):
        st.markdown('<div class="subtitulo">🧩 Diagrama de Flujo del Proceso</div>', unsafe_allow_html=True)
        if os.path.exists(opcion):
            st.image(opcion, caption="Diagrama de Flujo del Proceso", use_container_width=True)
        else:
            st.error(f"⚠️ No se encontró la imagen `{opcion}`.")

# =====================================================
# 📊 SPRINT 2 - ANÁLISIS DE DATOS
# =====================================================
elif menu == "SPRINT 2 - Análisis":
    st.markdown('<div class="titulo">📊 SPRINT 2 - Análisis de Datos</div>', unsafe_allow_html=True)
    st.markdown('<p class="descripcion">Este módulo agrupa el análisis exploratorio, las visualizaciones generales y las métricas de negocio del Proyecto Aurelion.</p>', unsafe_allow_html=True)
    st.divider()

    seccion = st.selectbox(
        "📂 Selecciona una parte del análisis:",
        (
            "📄 Documentación actualizada",
            "🧾 Datos limpios (CSV)",
            "🔍 Análisis exploratorio (EDA)",
            "📈 Visualizaciones generales",
            "💼 Métricas de negocio",
            "🧠 Resultados y conclusión"
        )
    )

    # -----------------------------------------------------
    # 📄 DOCUMENTACIÓN ACTUALIZADA
    # -----------------------------------------------------
    if seccion.startswith("📄"):
        mostrar_markdown("documentacion.md")

    # -----------------------------------------------------
    # 🧾 DATOS LIMPIOS
    # -----------------------------------------------------
    elif seccion.startswith("🧾"):
        if os.path.exists(csv_path):
            st.success("✅ Dataset cargado correctamente.")
            st.dataframe(df.head(50), use_container_width=True)
            st.info(f"Filas totales: {len(df):,}")
        else:
            st.error("⚠️ No se encontró el dataset limpio.")

    # -----------------------------------------------------
    # 🔍 ANÁLISIS EXPLORATORIO (EDA)
    # -----------------------------------------------------
    elif seccion.startswith("🔍"):
        st.markdown('<div class="subtitulo">🔍 Análisis Exploratorio de Datos</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">En esta sección se examinan distribuciones, outliers y relaciones entre variables.</p>', unsafe_allow_html=True)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            with st.expander("📊 Estadísticas descriptivas detalladas", expanded=True):
                graficos.analisis_estadistico(df)

            st.divider()
            st.markdown('<p class="descripcion">📈 Distribuciones y relaciones principales.</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.caption("💰 Distribución del importe total")
                st.pyplot(graficos.viz_importe_hist_kde(df))

                st.caption("📦 Comparación de importes por categoría")
                st.pyplot(graficos.plot_box_importe_by_categoria(df))

            with col2:
                st.caption("🧮 Relación entre cantidad e importe")
                st.pyplot(graficos.plot_scatter_cantidad_importe(df))

                st.caption("🕒 Tendencia del importe a lo largo del tiempo")
                st.pyplot(graficos.viz_tiempo_linea(df))

            with st.expander("📆 Mapa de calor de ventas por mes y año"):
                st.caption("Representa la concentración de ventas en el calendario.")
                st.pyplot(graficos.viz_heatmap_ventas_calendario(df))

        else:
            st.error("⚠️ No se encontró el dataset limpio.")

    # -----------------------------------------------------
    # 📈 VISUALIZACIONES GENERALES
    # -----------------------------------------------------
    elif seccion.startswith("📈"):
        st.markdown('<div class="subtitulo">📈 Visualizaciones Generales</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Visualizaciones globales del comportamiento de ventas, categorías, ubicaciones y medios de pago.</p>', unsafe_allow_html=True)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            with st.expander("📅 Importe promedio por día", expanded=True):
                st.caption("Promedio diario de importe de ventas.")
                st.pyplot(graficos.grafico_importe_promedio_por_dia(df))

            with st.expander("📊 Categorías más vendidas por mes"):
                st.caption("Muestra qué categorías lideraron las ventas en cada mes.")
                st.pyplot(graficos.grafico_categoria_mas_vendida_por_mes(df))

            with st.expander("🌍 Ventas por ciudad"):
                st.caption("Distribución geográfica de las ventas registradas.")
                st.pyplot(graficos.grafico_ventas_por_ciudad(df))

            with st.expander("💳 Distribución de medios de pago"):
                st.caption("Proporción de cada medio de pago utilizado por los clientes.")
                st.pyplot(graficos.grafico_medio_pago(df))

            with st.expander("🗺️ Mapa interactivo de ventas"):
                st.caption("Mapa de ubicaciones de ventas según la ciudad o región.")
                graficos.grafico_mapa_ventas(df)

        else:
            st.error("⚠️ No se encontró el dataset limpio.")

    # -----------------------------------------------------
    # 💼 MÉTRICAS DE NEGOCIO
    # -----------------------------------------------------
    elif seccion.startswith("💼"):
        st.markdown('<div class="subtitulo">💼 Métricas de Negocio (KPI)</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Indicadores clave de rendimiento para evaluar ingresos, márgenes y fidelidad del cliente.</p>', unsafe_allow_html=True)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            with st.expander("💰 Ticket promedio por medio de pago", expanded=True):
                st.caption("Compara el ticket promedio entre distintos medios de pago.")
                st.pyplot(graficos.grafico_ticket_promedio_por_medio(df))

            with st.expander("📦 Ticket promedio por categoría"):
                st.caption("Evalúa qué categorías generan ventas más altas en promedio.")
                st.pyplot(graficos.biz_ticket_promedio_vs_categoria(df))

            with st.expander("🏆 Top clientes por ingresos"):
                st.caption("Clientes que generaron mayores ingresos totales.")
                st.pyplot(graficos.biz_top_clientes_ingresos(df))

            with st.expander("🔁 Frecuencia de compra por cliente"):
                st.caption("Mide la recurrencia de compras de cada cliente.")
                st.pyplot(graficos.biz_frecuencia_compra_por_cliente(df))

            with st.expander("📈 Margen estimado por categoría"):
                st.caption("Margen de beneficio aproximado considerando un 25% por unidad.")
                st.pyplot(graficos.biz_margen_aprox_por_categoria(df))

            with st.expander("🕒 Cohorte por mes de alta y retención"):
                st.caption("Analiza la retención de clientes según su mes de alta.")
                st.pyplot(graficos.biz_cohorte_mes_alta_retencion(df))

        else:
            st.error("⚠️ No se encontró el dataset limpio.")

    # -----------------------------------------------------
    # 🧠 RESULTADOS Y CONCLUSIÓN
    # -----------------------------------------------------
    elif seccion.startswith("🧠"):
        st.markdown('<div class="subtitulo">🧠 Resultados y Conclusión</div>', unsafe_allow_html=True)
        mostrar_markdown("conclusion.md")
# =====================================================
# 📊 SPRINT 3 - MACHINE LEARNING
# =====================================================
elif menu == "SPRINT 3 - Machine Learning":
    st.markdown('<div class="titulo">🤖 SPRINT 3 - Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<p class="descripcion">Este módulo agrupa los modelos de Machine Learning desarrollados para el Proyecto Aurelion.</p>', unsafe_allow_html=True)
    st.divider()

    opcion_ml = st.selectbox(
        "Selecciona el modelo que deseas consultar:",
        ("Predicción de Ventas por Producto", "Clasificación de Medios de Pago", "Segmentación de Clientes (RFM)")
    )

    if opcion_ml == "Predicción de Ventas por Producto":
        st.markdown('<div class="subtitulo">📦 Predicción de Ventas por Producto</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Predice cuántas unidades se venderán de cada producto el próximo mes. Útil para gestión de inventario y planificación de compras.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Información del dataset
            df['fecha'] = pd.to_datetime(df['fecha'])
            st.info(f"📊 Tu dataset contiene **{df['id_producto'].nunique()} productos únicos** en **{df['fecha'].dt.to_period('M').nunique()} meses**")
            # Después de cargar tus datos
            result = regression.run_pipeline(df, test_size=0.2)

            print("=== DIAGNÓSTICO ===")
            print(f"Total observaciones: {len(result['df_productos'])}")
            print(f"\nTrain size: {len(result['X_train'])}")
            print(f"Test size: {len(result['X_test'])}")

            print(f"\n=== LAGS EN TRAIN ===")
            print(result['X_train'].describe())

            print(f"\n=== LAGS EN TEST ===")
            print(result['X_test'].describe())

            print(f"\n=== TARGET (y_train) ===")
            print(result['y_train'].describe())

            print(f"\n=== TARGET (y_test) ===")
            print(result['y_test'].describe())

            print(f"\n=== COEFICIENTES DEL MODELO ===")
            print(f"Intercepto: {result['model'].intercept_}")
            print(f"Coeficientes: {result['model'].coef_}")

            print(f"\n=== MUESTRA DE PREDICCIONES ===")
            print(pd.DataFrame({
                'y_real': result['y_test'][:10].values,
                'y_pred': result['y_pred'][:10]
            }))
            # Control de test size
            test_size = st.slider("Porcentaje de datos para Test", 
                                min_value=10, max_value=40, value=20, step=5,
                                help="% de observaciones que se reservan para validar el modelo") / 100
            
            st.divider()
            
            # Ejecutar pipeline
            try:
                with st.spinner("🔄 Entrenando modelo de predicción por producto..."):
                    result = regression.run_pipeline(df, test_size=test_size)
                
                st.success(f"✅ Modelo entrenado con **{len(result['X_train'])} observaciones** (producto-mes)!")
                
                # 1. Análisis exploratorio
                st.markdown("### 📊 Análisis Exploratorio de Productos")
                st.pyplot(result["fig_exploracion"])
                
                with st.expander("ℹ️ ¿Qué muestran estos gráficos?"):
                    st.markdown("""
                    - **Top 10 Productos:** Los productos más vendidos históricamente
                    - **Ventas por Categoría:** Qué categorías tienen mejor desempeño
                    - **Distribución de Ventas:** Cómo se distribuyen las ventas mensuales
                    - **Precio vs Ventas:** Relación entre precio unitario y cantidad vendida
                    """)
                
                st.divider()
                
                # 2. Métricas del modelo
                st.markdown("### 📈 Rendimiento del Modelo")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{result['metrics']['R2']:.3f}", 
                            help="Porcentaje de variabilidad explicada")
                with col2:
                    st.metric("MAE", f"{result['metrics']['MAE']:.1f} unid.", 
                            help="Error promedio en unidades")
                with col3:
                    st.metric("RMSE", f"{result['metrics']['RMSE']:.1f} unid.",
                            help="Error cuadrático medio")
                with col4:
                    st.metric("MAPE", f"{result['metrics']['MAPE']:.1f}%",
                            help="Error porcentual promedio")
                
                st.divider()
                
                # 3. Gráficos de evaluación
                st.markdown("### 🎯 Evaluación de Predicciones")
                st.pyplot(result["fig_predicciones"])
                
                with st.expander("ℹ️ Cómo interpretar estos gráficos"):
                    st.markdown("""
                    - **Real vs Predicho:** Los puntos cerca de la línea roja indican buenas predicciones
                    - **Distribución de Errores:** Centrada en 0 indica que el modelo no tiene sesgo
                    - **Top Errores:** Muestra los productos más difíciles de predecir
                    """)
                
                st.divider()
                
                # 4. Importancia de variables
                st.markdown("### 🔍 Variables Más Importantes")
                
                # Mostrar top 10 features
                df_coef_top = result["coef_df"].head(10).copy()
                df_coef_top["Coeficiente"] = df_coef_top["Coeficiente"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df_coef_top, use_container_width=True, hide_index=True)
                
                with st.expander("ℹ️ ¿Qué significan estas variables?"):
                    st.markdown("""
                    **Variables de historial:**
                    - **Ventas del mes anterior (lag_1):** El mejor predictor. Si un producto vendió bien el mes pasado, probablemente venda bien este mes
                    - **Promedio móvil 3 meses:** Suaviza fluctuaciones aleatorias
                    - **Tendencia:** Captura si el producto está en crecimiento o declive
                    
                    **Variables de contexto:**
                    - **Mes del año:** Captura estacionalidad (ej: heladeras en verano)
                    - **Promedio de la categoría:** Productos de categorías populares venden más
                    - **Meses activos:** Productos con más historial son más predecibles
                    
                    **Variables de producto:**
                    - **Precio unitario:** Productos más caros suelen vender menos unidades
                    - **Categoría:** Algunas categorías son más demandadas que otras
                    """)
                
                st.divider()
                
                # 5. Predicciones para el próximo mes
                st.markdown("### 🔮 Predicciones para el Próximo Mes")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    categorias = ["Todas"] + sorted(result["df_futuro"]["categoria"].unique().tolist())
                    cat_filtro = st.selectbox("Filtrar por Categoría", categorias)
                
                with col2:
                    top_n = st.slider("Mostrar Top N productos", min_value=10, max_value=50, value=20, step=5)
                
                # Aplicar filtros
                df_pred_display = result["df_futuro"].copy()
                if cat_filtro != "Todas":
                    df_pred_display = df_pred_display[df_pred_display["categoria"] == cat_filtro]
                
                df_pred_display = df_pred_display.head(top_n)
                
                # Formatear para display
                df_pred_display["cantidad_predicha"] = df_pred_display["cantidad_predicha"].apply(lambda x: f"{x:.1f}")
                df_pred_display["precio_unitario"] = df_pred_display["precio_unitario"].apply(lambda x: f"${x:.2f}")
                df_pred_display["ingresos_estimados"] = df_pred_display["ingresos_estimados"].apply(lambda x: f"${x:,.2f}")
                
                df_pred_display = df_pred_display[[
                    "nombre_producto", "categoria", "mes_predicho", 
                    "cantidad_predicha", "precio_unitario", "ingresos_estimados"
                ]]
                
                df_pred_display.rename(columns={
                    "nombre_producto": "Producto",
                    "categoria": "Categoría",
                    "mes_predicho": "Mes",
                    "cantidad_predicha": "Unidades Predichas",
                    "precio_unitario": "Precio Unit.",
                    "ingresos_estimados": "Ingresos Estimados"
                }, inplace=True)
                
                st.dataframe(df_pred_display, use_container_width=True, hide_index=True)
                
                # Resumen de predicciones
                total_unidades = result["df_futuro"]["cantidad_predicha"].sum()
                total_ingresos = result["df_futuro"]["ingresos_estimados"].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📦 Total Unidades Predichas", f"{total_unidades:,.0f}")
                with col2:
                    st.metric("💰 Ingresos Estimados", f"${total_ingresos:,.2f}")
                
                st.divider()
                
                # 6. Interpretación textual
                st.markdown(result["texto_interpretacion"])
                
                st.divider()
                
                # Distribución de resultados (al final)
                st.markdown("### 📊 Distribución de Predicciones del Modelo")

                st.markdown("""
                Estos gráficos permiten analizar si el modelo está capturando la forma real
                de los datos o si está sesgado hacia ciertos valores.
                """)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Distribución Real vs Predicha")
                    st.pyplot(result["fig_hist"])

                with col2:
                    st.markdown("#### Curvas de Densidad (KDE)")
                    st.pyplot(result["fig_kde"])

                st.divider()
                
                # 7. Descargas
                st.markdown("### 💾 Descargar Resultados")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_pred = result["df_futuro"].to_csv(index=False)
                    st.download_button(
                        label="📥 Predicciones por Producto",
                        data=csv_pred,
                        file_name="predicciones_productos.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_coef = result["coef_df"].to_csv(index=False)
                    st.download_button(
                        label="📥 Importancia de Variables",
                        data=csv_coef,
                        file_name="importancia_variables.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    csv_metrics = pd.DataFrame([result["metrics"]]).to_csv(index=False)
                    st.download_button(
                        label="📥 Métricas del Modelo",
                        data=csv_metrics,
                        file_name="metricas_modelo.csv",
                        mime="text/csv"
                    )
                
            except ValueError as ve:
                st.error(str(ve))
                st.stop()
            except Exception as e:
                st.error(f"❌ Error inesperado:")
                st.exception(e)
                st.stop()
        
        else:
            st.warning("⚠️ Por favor, carga los datos primero en la sección de inicio.")
            

    elif opcion_ml == "Clasificación de Medios de Pago":
        st.markdown('<div class="subtitulo">💳 Clasificación de Medio de Pago</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Predice qué medio de pago usará un cliente según características de la compra. Útil para personalización y optimización de pagos.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Información del dataset
            df['fecha'] = pd.to_datetime(df['fecha'])
            medios_pago = df['medio_pago'].nunique()
            st.info(f"📊 Tu dataset contiene **{len(df)} transacciones** con **{medios_pago} medios de pago** diferentes")
            
            # Control de test size
            test_size = st.slider("Porcentaje de datos para Test", 
                                min_value=10, max_value=40, value=20, step=5,
                                help="% de transacciones que se reservan para validar el modelo") / 100
            
            st.divider()
            
            # Ejecutar pipeline
            try:
                with st.spinner("🔄 Entrenando modelo de clasificación..."):
                    result = classification.run_pipeline(df, test_size=test_size)
                
                st.success(f"✅ Modelo entrenado con **{len(result['X_train'])} transacciones**!")
                
                # 1. Análisis exploratorio
                st.markdown("### 📊 Análisis Exploratorio")
                st.pyplot(result["fig_exploracion"])
                
                with st.expander("ℹ️ ¿Qué muestran estos gráficos?"):
                    st.markdown("""
                    - **Distribución de Medios de Pago:** Frecuencia de uso de cada método
                    - **Medio de Pago por Categoría:** Qué productos se compran con cada método
                    - **Importe Promedio:** Cuánto gastan en promedio con cada medio de pago
                    - **Uso por Día:** Patrones de uso según día de la semana
                    """)
                
                st.divider()
                
                # 2. Métricas del modelo
                st.markdown("### 📈 Rendimiento del Modelo")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{result['metrics']['accuracy']:.1%}", 
                            help="Porcentaje de predicciones correctas")
                with col2:
                    st.metric("Precision", f"{result['metrics']['precision']:.1%}", 
                            help="Precisión ponderada por clase")
                with col3:
                    st.metric("Recall", f"{result['metrics']['recall']:.1%}",
                            help="Cobertura ponderada por clase")
                with col4:
                    st.metric("F1-Score", f"{result['metrics']['f1_score']:.1%}",
                            help="Balance entre precisión y recall")
                
                st.divider()
                
                # 3. Matriz de confusión
                st.markdown("### 🎯 Matriz de Confusión")
                st.pyplot(result["fig_matriz"])
                
                with st.expander("ℹ️ Cómo interpretar la matriz"):
                    st.markdown("""
                    La matriz muestra qué tan bien predice el modelo cada medio de pago:
                    - **Diagonal principal (azul oscuro):** Predicciones correctas
                    - **Fuera de la diagonal:** Confusiones del modelo
                    - Los valores están normalizados (0 a 1) para facilitar la comparación
                    
                    **Ejemplo:** Si en la fila "Efectivo" y columna "Tarjeta" hay 0.20, significa que el 20% de las compras reales en efectivo fueron predichas incorrectamente como tarjeta.
                    """)
                
                st.divider()
                
                # 4. Importancia de variables
                st.markdown("### 🔍 Variables Más Importantes")
                st.pyplot(result["fig_importancia"])
                
                with st.expander("ℹ️ ¿Qué significa la importancia?"):
                    st.markdown("""
                    **Interpretación:**
                    - Las variables más arriba son las más útiles para predecir el medio de pago
                    - La importancia mide cuánto aporta cada variable a la precisión del modelo
                    
                    **Variables típicamente importantes:**
                    - **Monto de la compra:** Compras grandes suelen usar tarjeta
                    - **Historial del cliente:** Clientes frecuentes tienen patrones establecidos
                    - **Categoría:** Algunos productos se asocian más con ciertos métodos de pago
                    """)
                
                st.divider()
                
                # 5. Distribución de predicciones
                st.markdown("### 📊 Distribución Real vs Predicha")
                st.pyplot(result["fig_distribucion"])
                
                with st.expander("ℹ️ ¿Qué muestra este gráfico?"):
                    st.markdown("""
                    Compara la distribución de medios de pago en el test set:
                    - **Izquierda (azul):** Distribución real de los datos de test
                    - **Derecha (naranja):** Distribución de las predicciones del modelo
                    
                    Si ambas son similares, el modelo está capturando bien los patrones.
                    Si son muy diferentes, puede haber sesgo hacia ciertas clases.
                    """)
                
                st.divider()
                
                # 6. Rendimiento por clase
                st.markdown("### 📋 Rendimiento por Medio de Pago")
                
                # Crear DataFrame con métricas por clase
                report_data = []
                for clase in result['classes']:
                    if clase in result['metrics']['report']:
                        clase_metrics = result['metrics']['report'][clase]
                        report_data.append({
                            "Medio de Pago": clase,
                            "Precision": f"{clase_metrics['precision']:.1%}",
                            "Recall": f"{clase_metrics['recall']:.1%}",
                            "F1-Score": f"{clase_metrics['f1-score']:.1%}",
                            "Casos en Test": int(clase_metrics['support'])
                        })
                
                df_report = pd.DataFrame(report_data)
                st.dataframe(df_report, use_container_width=True, hide_index=True)
                
                with st.expander("ℹ️ Definición de métricas"):
                    st.markdown("""
                    - **Precision:** De todas las predicciones de este medio de pago, ¿cuántas fueron correctas?
                    - **Recall:** De todos los casos reales de este medio de pago, ¿cuántos detectó el modelo?
                    - **F1-Score:** Promedio armónico entre precision y recall
                    - **Casos en Test:** Cantidad de transacciones reales con este medio de pago en el test set
                    """)
                
                st.divider()
                
                # 7. Interpretación textual
                st.markdown(result["texto_interpretacion"])
                
                st.divider()
            
                # 8. NUEVA SECCIÓN: Prueba con nuevas transacciones
                st.markdown("### 🧪 Prueba el Modelo con Nuevas Transacciones")
                
                st.markdown("""
                Simula una nueva transacción y observa qué medio de pago predice el modelo.
                Ajusta los parámetros para ver cómo cambian las predicciones.
                """)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    importe_nuevo = st.number_input("💰 Importe ($)", min_value=0.0, max_value=50000.0, value=5000.0, step=100.0)
                    cantidad_nuevo = st.number_input("📦 Cantidad", min_value=1, max_value=50, value=2)
                    categoria_nuevo = st.selectbox("🏷️ Categoría", options=result["encoders"]["categoria"].classes_)
                
                with col2:
                    ciudad_nuevo = st.selectbox("🌆 Ciudad", options=result["encoders"]["ciudad"].classes_)
                    mes_nuevo = st.slider("📅 Mes", min_value=1, max_value=12, value=6)
                    dia_semana_nuevo = st.selectbox("📆 Día de la semana", 
                                                    options=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
                                                    index=2)
                
                with col3:
                    es_fin_semana_nuevo = st.checkbox("🎉 Es fin de semana", value=False)
                    compras_previas_nuevo = st.number_input("🛒 Compras previas del cliente", min_value=0, max_value=100, value=5)
                    gasto_promedio_nuevo = st.number_input("📊 Gasto promedio del cliente ($)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0)
                
                # Mapear día de la semana
                dias_map = {
                    "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
                    "Viernes": 4, "Sábado": 5, "Domingo": 6
                }
                dia_semana_numero = dias_map[dia_semana_nuevo]
                
                # Determinar rango de precio
                if importe_nuevo < result["df_clasificacion"]["importe"].quantile(0.33):
                    rango_precio_nuevo = "bajo"
                elif importe_nuevo < result["df_clasificacion"]["importe"].quantile(0.67):
                    rango_precio_nuevo = "medio"
                else:
                    rango_precio_nuevo = "alto"
                
                if st.button("🔮 Predecir Medio de Pago", type="primary"):
                    try:
                        # Crear diccionario de transacción
                        nueva_transaccion = {
                            "importe": importe_nuevo,
                            "cantidad": cantidad_nuevo,
                            "categoria": categoria_nuevo,
                            "ciudad": ciudad_nuevo,
                            "mes": mes_nuevo,
                            "dia_semana": dia_semana_numero,
                            "es_fin_semana": 1 if es_fin_semana_nuevo else 0,
                            "rango_precio": rango_precio_nuevo,
                            "compras_previas_cliente": compras_previas_nuevo,
                            "gasto_promedio_cliente": gasto_promedio_nuevo
                        }
                        
                        # Hacer predicción
                        prediccion = classification.predecir_nueva_transaccion(
                            result["model"],
                            result["encoders"],
                            nueva_transaccion
                        )
                        
                        st.success(f"### 🎯 Medio de Pago Predicho: **{prediccion['medio_pago_predicho']}**")
                        
                        # Mostrar probabilidades
                        st.markdown("#### 📊 Probabilidades por Medio de Pago")
                        
                        # Ordenar probabilidades de mayor a menor
                        probs_ordenadas = sorted(prediccion["probabilidades"].items(), key=lambda x: x[1], reverse=True)
                        
                        for medio, prob in probs_ordenadas:
                            # Barra de progreso visual
                            col_a, col_b = st.columns([1, 4])
                            with col_a:
                                st.write(f"**{medio}**")
                            with col_b:
                                st.progress(prob)
                                st.caption(f"{prob*100:.1f}%")
                        
                        # Explicación
                        st.info(f"""
                        💡 **Interpretación:** El modelo predice con **{max(prediccion['probabilidades'].values())*100:.1f}%** 
                        de confianza que el cliente usará **{prediccion['medio_pago_predicho']}** para esta transacción.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error al hacer la predicción: {str(e)}")
                
                st.divider()
                
                # 9. NUEVA SECCIÓN: Análisis de errores
                st.markdown("### 🔍 Análisis de Errores del Modelo")
                
                if result["df_errores"] is not None and len(result["df_errores"]) > 0:
                    st.markdown(f"""
                    El modelo cometió **{len(result['df_errores'])} errores** de **{len(result['y_test'])} predicciones** 
                    en el conjunto de test (**{len(result['df_errores'])/len(result['y_test'])*100:.1f}%** de error).
                    """)
                    
                    # Mostrar tabla de errores
                    st.dataframe(result["df_errores"], use_container_width=True, hide_index=True)
                    
                    # Análisis de patrones de error
                    st.markdown("#### 🔎 Patrones de Error Detectados")
                    
                    # Confusiones más comunes
                    errores_por_tipo = result["df_errores"].groupby(["Real", "Predicho"]).size().reset_index(name="Frecuencia")
                    errores_por_tipo = errores_por_tipo.sort_values("Frecuencia", ascending=False).head(5)
                    
                    st.markdown("**Top 5 Confusiones Más Frecuentes:**")
                    for _, row in errores_por_tipo.iterrows():
                        st.write(f"- Confunde **{row['Real']}** con **{row['Predicho']}**: {row['Frecuencia']} veces")
                    
                    with st.expander("💡 ¿Por qué ocurren estos errores?"):
                        st.markdown("""
                        **Posibles razones:**
                        
                        1. **Clases similares:** Algunos medios de pago pueden tener características similares (ej: efectivo vs débito para compras pequeñas)
                        
                        2. **Datos insuficientes:** Con pocas transacciones, el modelo no puede aprender bien todos los patrones
                        
                        3. **Factores no capturados:** Puede haber variables importantes que no están en el modelo (ej: promociones especiales, preferencias personales)
                        
                        4. **Comportamiento impredecible:** Los clientes no siempre actúan de forma predecible
                        
                        **Para mejorar:**
                        - Recopilar más transacciones históricas
                        - Agregar features adicionales (hora del día, dispositivo usado, etc.)
                        - Usar técnicas de balanceo si hay clases minoritarias
                        """)
                else:
                    st.success("🎉 ¡Perfecto! El modelo clasificó correctamente todas las transacciones del test set.")
                
                st.divider()
                
                # 8. Descargas
                st.markdown("### 💾 Descargar Resultados")
            
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_report = df_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Reporte por Clase",
                        data=csv_report,
                        file_name="reporte_clasificacion.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_metrics = pd.DataFrame([{
                        "Accuracy": result['metrics']['accuracy'],
                        "Precision": result['metrics']['precision'],
                        "Recall": result['metrics']['recall'],
                        "F1-Score": result['metrics']['f1_score']
                    }]).to_csv(index=False)
                    st.download_button(
                        label="📥 Métricas Generales",
                        data=csv_metrics,
                        file_name="metricas_clasificacion.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    if result["df_errores"] is not None:
                        csv_errores = result["df_errores"].to_csv(index=False)
                        st.download_button(
                            label="📥 Análisis de Errores",
                            data=csv_errores,
                            file_name="errores_clasificacion.csv",
                            mime="text/csv"
                        )
                
            except ValueError as ve:
                st.error(str(ve))
                st.stop()
            except Exception as e:
                st.error(f"❌ Error inesperado:")
                st.exception(e)
                st.stop()
        
    elif opcion_ml == "Segmentación de Clientes (RFM)":
        st.markdown('<div class="subtitulo">👥 Segmentación de Clientes (RFM)</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Agrupa clientes según su comportamiento de compra: Recency (reciente), Frequency (frecuencia) y Monetary (valor). Útil para estrategias de marketing personalizadas.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Información del dataset
            df['fecha'] = pd.to_datetime(df['fecha'])
            n_clientes = df['id_cliente'].nunique()
            st.info(f"📊 Tu dataset contiene **{n_clientes} clientes únicos** para segmentar")
            
            # Control de número de clusters
            n_clusters = st.slider("Número de Segmentos (Clusters)", 
                                min_value=2, max_value=min(8, n_clientes), value=4,
                                help="Cantidad de grupos en los que se dividirán los clientes")
            
            st.divider()
            
            # Ejecutar pipeline
            try:
                with st.spinner("🔄 Analizando comportamiento de clientes..."):
                    result = clustering.run_pipeline(df, n_clusters=n_clusters)
                
                st.success(f"✅ Segmentación completada! Se identificaron **{n_clusters} grupos** de clientes")
                
                # 1. Método del Codo y Silhouette
                st.markdown("### 📉 Determinación del Número Óptimo de Clusters")
                st.pyplot(result["fig_elbow"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}",
                            help="Mide qué tan bien definidos están los clusters (0.5+ es bueno)")
                with col2:
                    # Encontrar K óptimo según Silhouette
                    idx_max = np.argmax(result["metricas_k"]["silhouette_scores"])
                    k_optimo = result["metricas_k"]["k_values"][idx_max]
                    st.metric("K Óptimo Sugerido", f"{k_optimo} clusters",
                            help="Basado en el máximo Silhouette Score")
                
                with st.expander("ℹ️ ¿Cómo interpretar estos gráficos?"):
                    st.markdown("""
                    **Método del Codo (Elbow):**
                    - Busca el "codo" donde la inercia deja de disminuir significativamente
                    - Ese punto indica un buen equilibrio entre número de clusters y calidad
                    
                    **Silhouette Score:**
                    - Mide qué tan similares son los clientes dentro del mismo cluster vs otros clusters
                    - **> 0.7:** Excelente separación
                    - **0.5 - 0.7:** Buena separación
                    - **< 0.5:** Clusters pueden solaparse
                    
                    💡 **Recomendación:** Usa el K donde el Silhouette Score es máximo
                    """)
                
                st.divider()
                
                # 2. Visualización 2D
                st.markdown("### 🎯 Visualización de Segmentos")
                st.pyplot(result["fig_2d"])
                
                with st.expander("ℹ️ ¿Qué muestra este gráfico?"):
                    st.markdown("""
                    Este gráfico usa **PCA** (Análisis de Componentes Principales) para reducir las 3 dimensiones RFM a 2D.
                    
                    - Cada punto = un cliente
                    - Colores = diferentes segmentos
                    - Distancia entre puntos = similaridad de comportamiento
                    
                    **Clusters bien definidos:** Los puntos del mismo color están juntos y separados de otros colores
                    """)
                
                st.divider()
                
                # 3. Características de clusters
                st.markdown("### 📊 Características de cada Segmento")
                st.pyplot(result["fig_caracteristicas"])
                
                with st.expander("ℹ️ Interpretación de métricas RFM"):
                    st.markdown("""
                    **Recency (Recencia):**
                    - Días desde la última compra
                    - ✅ **Menor es mejor** (cliente más activo)
                    
                    **Frequency (Frecuencia):**
                    - Cantidad de compras realizadas
                    - ✅ **Mayor es mejor** (cliente más leal)
                    
                    **Monetary (Monetario):**
                    - Total gastado en todas las compras
                    - ✅ **Mayor es mejor** (cliente más valioso)
                    """)
                
                st.divider()
                
                # 4. Distribución de clientes
                st.markdown("### 📊 Distribución de Clientes por Segmento")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.pyplot(result["fig_distribucion"])
                
                with col2:
                    st.markdown("#### 📋 Resumen por Segmento")
                    for cluster_nombre in sorted(result["df_rfm"]["Cluster_Nombre"].unique()):
                        df_cluster = result["df_rfm"][result["df_rfm"]["Cluster_Nombre"] == cluster_nombre]
                        porcentaje = len(df_cluster) / len(result["df_rfm"]) * 100
                        st.metric(
                            cluster_nombre,
                            f"{len(df_cluster)} clientes",
                            f"{porcentaje:.1f}%"
                        )
                
                st.divider()
                
                # 5. Tabla de clientes por segmento
                st.markdown("### 👥 Detalle de Clientes por Segmento")
                
                # Filtro por cluster
                clusters_disponibles = ["Todos"] + sorted(result["df_rfm"]["Cluster_Nombre"].unique().tolist())
                cluster_seleccionado = st.selectbox("Filtrar por Segmento", clusters_disponibles)
                
                # Aplicar filtro
                if cluster_seleccionado == "Todos":
                    df_display = result["df_rfm"].copy()
                else:
                    df_display = result["df_rfm"][result["df_rfm"]["Cluster_Nombre"] == cluster_seleccionado].copy()
                
                # Formatear para display
                df_display_formatted = df_display[[
                    "nombre_cliente", "email", "ciudad", "Cluster_Nombre",
                    "Recency", "Frequency", "Monetary"
                ]].copy()
                
                df_display_formatted["Monetary"] = df_display_formatted["Monetary"].apply(lambda x: f"${x:,.2f}")
                
                df_display_formatted.rename(columns={
                    "nombre_cliente": "Cliente",
                    "email": "Email",
                    "ciudad": "Ciudad",
                    "Cluster_Nombre": "Segmento",
                    "Recency": "Días (última compra)",
                    "Frequency": "# Compras",
                    "Monetary": "Total Gastado"
                }, inplace=True)
                
                st.dataframe(df_display_formatted, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # 6. Interpretación textual
                st.markdown(result["texto_interpretacion"])
                
                st.divider()
                
                # 7. Descargas
                st.markdown("### 💾 Descargar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_clientes = result["df_rfm"][[
                        "id_cliente", "nombre_cliente", "email", "ciudad",
                        "Cluster", "Cluster_Nombre", "Recency", "Frequency", "Monetary"
                    ]].to_csv(index=False)
                    st.download_button(
                        label="📥 Clientes Segmentados",
                        data=csv_clientes,
                        file_name="segmentacion_clientes.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Resumen por cluster
                    resumen_clusters = result["df_rfm"].groupby("Cluster_Nombre").agg({
                        "id_cliente": "count",
                        "Recency": "mean",
                        "Frequency": "mean",
                        "Monetary": ["mean", "sum"]
                    }).reset_index()
                    
                    resumen_clusters.columns = [
                        "Segmento", "Cantidad Clientes", "Recency Promedio",
                        "Frequency Promedio", "Monetary Promedio", "Monetary Total"
                    ]
                    
                    csv_resumen = resumen_clusters.to_csv(index=False)
                    st.download_button(
                        label="📥 Resumen por Segmento",
                        data=csv_resumen,
                        file_name="resumen_segmentos.csv",
                        mime="text/csv"
                    )
                
            except ValueError as ve:
                st.error(str(ve))
                st.stop()
            except Exception as e:
                st.error(f"❌ Error inesperado:")
                st.exception(e)
                st.stop()
        
        
        else:
            st.warning("⚠️ Por favor, carga los datos primero en la sección de inicio.")

# =====================================================
# SPRINT 4 - Power BI
# =====================================================

# =====================================================
# SPRINT 4 - POWER BI
# =====================================================
elif menu == "SPRINT 4 - Power BI":
    st.markdown('<div class="titulo">📊 SPRINT 4 - Tienda Aurelion Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="descripcion">Dashboards interactivos con análisis avanzados de ventas, clientes, productos y medios de pago.</p>', unsafe_allow_html=True)
    st.divider()
    
    # Selector de sección del dashboard
    seccion_dashboard = st.selectbox(
        "📂 Selecciona una sección del dashboard:",
        (
            "📊 General",
            "👥 Clientes", 
            "📦 Productos",
            "💳 Medios de Pago",
            "🔗 Dashboard Completo (Power BI Embed)"
        )
    )
    
    # =====================================================
    # OPCIÓN 1: DASHBOARD COMPLETO EMBEBIDO
    # =====================================================
    if seccion_dashboard == "🔗 Dashboard Completo (Power BI Embed)":
        st.markdown('<div class="subtitulo">🔗 Dashboard Completo de Power BI</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Visualiza el dashboard completo e interactivo publicado en Power BI Service.</p>', unsafe_allow_html=True)
        
        # URL de tu reporte de Power BI (debes reemplazarla con tu URL real)
        powerbi_embed_url = "https://app.powerbi.com/view?r=YOUR_EMBED_URL_HERE"
        
        with st.expander("ℹ️ ¿Cómo configurar la URL de Power BI?", expanded=False):
            st.markdown("""
            ### Pasos para obtener la URL de incrustación:
            
            1. **Publica tu reporte en Power BI Service:**
               - Abre tu archivo `AurelionVentas_v2.pbix` en Power BI Desktop
               - Ve a `Archivo > Publicar > Publicar en Power BI`
               - Selecciona tu área de trabajo
            
            2. **Obtén el enlace de incrustación:**
               - Ve a [app.powerbi.com](https://app.powerbi.com)
               - Abre tu reporte "Tienda Aurelion"
               - Haz clic en `Archivo > Insertar informe > Sitio web o portal`
               - Marca la opción "Habilitar contenido en modo de pantalla completa"
               - Copia la URL que aparece
            
            3. **Configura la URL en el código:**
               - En el archivo `app.py`, busca la variable `powerbi_embed_url`
               - Reemplaza `YOUR_EMBED_URL_HERE` con tu URL real
               - La URL debe verse así: `https://app.powerbi.com/view?r=XXXXXXXXXX`
            
            ### Alternativa - Sin publicar:
            Si no quieres publicar en Power BI Service, puedes usar las secciones individuales abajo que replican tu dashboard con Streamlit.
            """)
        
        st.divider()
        
        # Verificar si la URL está configurada
        if "YOUR_EMBED_URL" not in powerbi_embed_url:
            # Iframe para embeber Power BI
            iframe_html = f"""
            <iframe 
                width="100%" 
                height="900" 
                src="{powerbi_embed_url}" 
                frameborder="0" 
                allowFullScreen="true"
                style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            </iframe>
            """
            st.markdown(iframe_html, unsafe_allow_html=True)
            
            # Controles adicionales
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refrescar Dashboard"):
                    st.rerun()
            with col2:
                st.link_button("🔗 Abrir en Power BI", powerbi_embed_url)
            with col3:
                st.download_button(
                    label="📥 Descargar PDF",
                    data=open("AurelionVentas_v2.pdf", "rb").read() if os.path.exists("AurelionVentas_v2.pdf") else b"",
                    file_name="AurelionVentas_Dashboard.pdf",
                    mime="application/pdf",
                    disabled=not os.path.exists("AurelionVentas_v2.pdf")
                )
        else:
            st.warning("⚠️ La URL de Power BI no ha sido configurada.")
            st.info("👆 Por favor, sigue las instrucciones arriba para configurar tu URL de Power BI, o selecciona una de las secciones individuales en el menú desplegable.")
            
            # Mostrar preview del PDF si existe
            if os.path.exists("AurelionVentas_v2.pdf"):
                st.markdown("### 📄 Vista Previa del Dashboard (PDF)")
                with open("AurelionVentas_v2.pdf", "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
    
    # =====================================================
    # OPCIÓN 2: SECCIÓN GENERAL
    # =====================================================
    elif seccion_dashboard == "📊 General":
        st.markdown('<div class="subtitulo">📊 Dashboard General</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Vista general de las métricas principales de la tienda.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Calcular métricas principales
            total_ventas = df['importe'].sum()
            total_cantidad = df['cantidad'].sum()
            ticket_promedio = df['importe'].mean()
            
            # KPIs principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Ventas", f"$ {total_ventas/1_000_000:.2f} mill.", 
                         help="Suma total de ventas del período")
            with col2:
                st.metric("📦 Cantidad Vendidos", f"{total_cantidad/1_000:.0f} mil",
                         help="Total de unidades vendidas")
            with col3:
                st.metric("🎯 Ticket Promedio", f"$ {ticket_promedio/1_000:.2f} mil",
                         help="Importe promedio por transacción")
            
            st.divider()
            
            # Gráficos principales
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📈 Evolución temporal de las ventas")
                # Preparar datos
                df['fecha'] = pd.to_datetime(df['fecha'])
                df_temporal = df.groupby(df['fecha'].dt.to_period('M'))['importe'].sum().reset_index()
                df_temporal['fecha'] = df_temporal['fecha'].astype(str)
                
                # Gráfico con Streamlit nativo
                st.line_chart(
                    df_temporal.set_index('fecha')['importe'],
                    height=400
                )
            
            with col2:
                st.markdown("#### 📊 Cantidad por categoría")
                df_categoria = df.groupby('categoria')['cantidad'].sum().sort_values(ascending=True)
                st.bar_chart(df_categoria, height=400, horizontal=True)
            
            st.divider()
            
            # Selector de medio de pago
            st.markdown("#### 💳 Filtrar por Medio de Pago")
            medios_disponibles = ["Todos"] + sorted(df['medio_pago'].unique().tolist())
            medio_seleccionado = st.selectbox("Selecciona un medio de pago:", medios_disponibles)
            
            if medio_seleccionado != "Todos":
                df_filtrado = df[df['medio_pago'] == medio_seleccionado]
                st.info(f"📊 Mostrando datos filtrados por: **{medio_seleccionado}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Ventas (filtrado)", f"$ {df_filtrado['importe'].sum()/1_000:.2f} mil")
                with col2:
                    st.metric("Cantidad (filtrada)", f"{df_filtrado['cantidad'].sum()}")
                with col3:
                    st.metric("Ticket Promedio (filtrado)", f"$ {df_filtrado['importe'].mean()/1_000:.2f} mil")
        else:
            st.error("⚠️ No se encontró el dataset. Por favor, verifica la ruta del archivo.")
    
    # =====================================================
    # OPCIÓN 3: SECCIÓN CLIENTES
    # =====================================================
    elif seccion_dashboard == "👥 Clientes":
        st.markdown('<div class="subtitulo">👥 Análisis de Clientes</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Comportamiento y análisis detallado de clientes.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Métricas de clientes
            clientes_totales = df['id_cliente'].nunique()
            ticket_promedio_cliente = df.groupby('id_cliente')['importe'].sum().mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👥 Clientes Totales", f"{clientes_totales}", 
                         help="Cantidad de clientes únicos")
            with col2:
                st.metric("💰 Ticket Promedio por Cliente", f"$ {ticket_promedio_cliente/1_000:.2f} mil",
                         help="Gasto promedio por cliente")
            
            st.divider()
            
            # Top 10 clientes
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🏆 Top 10 clientes (según importe)")
                df_top_clientes = df.groupby('nombre_cliente')['importe'].sum().sort_values(ascending=False).head(10)
                
                # Formatear para display
                df_top_display = df_top_clientes.reset_index()
                df_top_display.columns = ['Cliente', 'Importe Total']
                df_top_display['Importe Total'] = df_top_display['Importe Total'].apply(lambda x: f"$ {x/1_000:.2f} mil")
                
                st.dataframe(df_top_display, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🌍 Importe por ciudad")
                df_ciudad = df.groupby('ciudad')['importe'].sum().sort_values(ascending=False)
                st.bar_chart(df_ciudad, height=400)
            
            st.divider()
            
            # Análisis de frecuencia
            st.markdown("#### 📊 Frecuencia de compras vs. importe por cliente")
            
            df_clientes_analisis = df.groupby('id_cliente').agg({
                'importe': 'sum',
                'fecha': 'count'  # Frecuencia de compras
            }).reset_index()
            df_clientes_analisis.columns = ['id_cliente', 'Total_Ventas', 'Frecuencia_Compras']
            
            # Gráfico de dispersión
            st.scatter_chart(
                df_clientes_analisis,
                x='Frecuencia_Compras',
                y='Total_Ventas',
                height=400
            )
            
            # Filtros adicionales
            st.divider()
            st.markdown("#### 🔍 Filtros Avanzados")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                categorias = ["Todas"] + sorted(df['categoria'].unique().tolist())
                cat_filtro = st.selectbox("Categoría:", categorias)
            with col2:
                meses = ["Todos"] + sorted(df['fecha'].dt.month_name().unique().tolist())
                mes_filtro = st.selectbox("Mes:", meses)
            with col3:
                ciudades = ["Todas"] + sorted(df['ciudad'].unique().tolist())
                ciudad_filtro = st.selectbox("Ciudad:", ciudades)
            
            # Aplicar filtros
            df_filtrado = df.copy()
            if cat_filtro != "Todas":
                df_filtrado = df_filtrado[df_filtrado['categoria'] == cat_filtro]
            if mes_filtro != "Todos":
                df_filtrado = df_filtrado[df_filtrado['fecha'].dt.month_name() == mes_filtro]
            if ciudad_filtro != "Todas":
                df_filtrado = df_filtrado[df_filtrado['ciudad'] == ciudad_filtro]
            
            if len(df_filtrado) > 0:
                st.success(f"✅ {len(df_filtrado)} transacciones encontradas con los filtros aplicados")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ventas (filtrado)", f"$ {df_filtrado['importe'].sum()/1_000:.2f} mil")
                with col2:
                    st.metric("Clientes únicos", f"{df_filtrado['id_cliente'].nunique()}")
                with col3:
                    st.metric("Ticket promedio", f"$ {df_filtrado['importe'].mean()/1_000:.2f} mil")
            else:
                st.warning("⚠️ No se encontraron resultados con los filtros seleccionados")
        
        else:
            st.error("⚠️ No se encontró el dataset.")
    
    # =====================================================
    # OPCIÓN 4: SECCIÓN PRODUCTOS
    # =====================================================
    elif seccion_dashboard == "📦 Productos":
        st.markdown('<div class="subtitulo">📦 Productos y Categorías</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Análisis de productos más vendidos y rendimiento por categoría.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Top 5 productos
            st.markdown("#### 🏆 Top 5 productos más vendidos (por importe)")
            
            df_top_productos = df.groupby('nombre_producto')['importe'].sum().sort_values(ascending=False).head(5)
            
            # Gráfico de barras horizontales
            st.bar_chart(df_top_productos, horizontal=True, height=300)
            
            # Tabla detallada
            with st.expander("📋 Ver detalle de productos"):
                df_productos_detalle = df.groupby('nombre_producto').agg({
                    'importe': 'sum',
                    'cantidad': 'sum',
                    'categoria': 'first'
                }).sort_values('importe', ascending=False).head(10)
                
                df_productos_detalle['importe'] = df_productos_detalle['importe'].apply(lambda x: f"$ {x/1_000:.2f} mil")
                df_productos_detalle.columns = ['Total Ventas', 'Cantidad Vendida', 'Categoría']
                st.dataframe(df_productos_detalle, use_container_width=True)
            
            st.divider()
            
            # Evolución por categoría
            st.markdown("#### 📈 Evolución de ventas por categoría")
            
            df['fecha'] = pd.to_datetime(df['fecha'])
            df_evol_cat = df.groupby([df['fecha'].dt.to_period('M'), 'categoria'])['importe'].sum().reset_index()
            df_evol_cat['fecha'] = df_evol_cat['fecha'].astype(str)
            
            # Pivot para el gráfico
            df_pivot = df_evol_cat.pivot(index='fecha', columns='categoria', values='importe')
            st.line_chart(df_pivot, height=400)
            
            st.divider()
            
            # Mapa de calor
            st.markdown("#### 🗺️ Mapa de calor: Ventas por ciudad y categoría")
            
            df_heatmap = df.groupby(['ciudad', 'categoria'])['importe'].sum().reset_index()
            df_heatmap_pivot = df_heatmap.pivot(index='ciudad', columns='categoria', values='importe')
            
            # Formatear para mostrar
            df_heatmap_display = df_heatmap_pivot.copy()
            df_heatmap_display['Total'] = df_heatmap_display.sum(axis=1)
            
            # Agregar fila de totales
            df_heatmap_display.loc['Total'] = df_heatmap_display.sum()
            
            # Formatear valores
            df_heatmap_formatted = df_heatmap_display.applymap(lambda x: f"$ {x:,.0f}")
            
            st.dataframe(
                df_heatmap_formatted,
                use_container_width=True,
                height=300
            )
            
            # Filtro por medio de pago
            st.divider()
            st.markdown("#### 💳 Filtrar por medio de pago")
            
            medios = ["Todos"] + sorted(df['medio_pago'].unique().tolist())
            medio_seleccionado = st.radio("Selecciona:", medios, horizontal=True)
            
            if medio_seleccionado != "Todos":
                df_filtrado = df[df['medio_pago'] == medio_seleccionado]
                
                st.info(f"📊 Productos más vendidos con **{medio_seleccionado}**")
                
                df_prod_filtrado = df_filtrado.groupby('nombre_producto')['importe'].sum().sort_values(ascending=False).head(5)
                st.bar_chart(df_prod_filtrado, horizontal=True, height=300)
        
        else:
            st.error("⚠️ No se encontró el dataset.")
    
    # =====================================================
    # OPCIÓN 5: SECCIÓN MEDIOS DE PAGO
    # =====================================================
    elif seccion_dashboard == "💳 Medios de Pago":
        st.markdown('<div class="subtitulo">💳 Análisis de Medios de Pago</div>', unsafe_allow_html=True)
        st.markdown('<p class="descripcion">Distribución y comportamiento de los diferentes medios de pago.</p>', unsafe_allow_html=True)
        
        if df is not None:
            # Participación de medios de pago
            st.markdown("#### 💰 Participación de medios de pago en las ventas")
            
            df_medios = df.groupby('medio_pago')['importe'].sum().sort_values(ascending=False)
            
            col1, col2, col3, col4 = st.columns(4)
            medios_list = df_medios.head(4).items()
            
            for col, (medio, valor) in zip([col1, col2, col3, col4], medios_list):
                with col:
                    st.metric(
                        f"{medio.capitalize()}",
                        f"$ {valor/1_000:.2f} mil",
                        help=f"Total vendido con {medio}"
                    )
            
            # Gráfico de torta (pie chart simulado con bar chart)
            st.bar_chart(df_medios, height=300)
            
            st.divider()
            
            # Evolución temporal
            st.markdown("#### 📈 Evolución de ventas por medio de pago")
            
            df['fecha'] = pd.to_datetime(df['fecha'])
            df_evol_medio = df.groupby([df['fecha'].dt.month, 'medio_pago'])['importe'].sum().reset_index()
            df_evol_medio.columns = ['Mes', 'Medio de Pago', 'Total Ventas']
            
            df_evol_pivot = df_evol_medio.pivot(index='Mes', columns='Medio de Pago', values='Total Ventas')
            st.line_chart(df_evol_pivot, height=400)
            
            st.divider()
            
            # Cantidad por medio de pago y categoría
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Cantidad por medio de pago y categoría")
                
                df_medio_cat = df.groupby(['medio_pago', 'categoria'])['cantidad'].sum().reset_index()
                df_medio_cat_pivot = df_medio_cat.pivot(index='medio_pago', columns='categoria', values='cantidad')
                
                st.bar_chart(df_medio_cat_pivot, height=400)
            
            with col2:
                st.markdown("#### 📅 Cantidad vendida por mes y medio de pago")
                
                df_mes_medio = df.groupby([df['fecha'].dt.month, 'medio_pago'])['cantidad'].sum().reset_index()
                df_mes_medio.columns = ['Mes', 'Medio de Pago', 'Cantidad']
                
                df_mes_medio_pivot = df_mes_medio.pivot(index='Mes', columns='Medio de Pago', values='Cantidad')
                st.bar_chart(df_mes_medio_pivot, height=400, horizontal=True)
            
            st.divider()
            
            # Análisis detallado por medio de pago
            st.markdown("#### 🔍 Análisis Detallado por Medio de Pago")
            
            medio_analizar = st.selectbox(
                "Selecciona un medio de pago para análisis detallado:",
                sorted(df['medio_pago'].unique())
            )
            
            if medio_analizar:
                df_medio_detail = df[df['medio_pago'] == medio_analizar]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Total Ventas", f"$ {df_medio_detail['importe'].sum()/1_000:.2f} mil")
                with col2:
                    st.metric("📦 Cantidad Vendida", f"{df_medio_detail['cantidad'].sum()}")
                with col3:
                    st.metric("🎯 Ticket Promedio", f"$ {df_medio_detail['importe'].mean()/1_000:.2f} mil")
                with col4:
                    st.metric("🔢 Transacciones", f"{len(df_medio_detail)}")
                
                # Productos más vendidos con este medio
                st.markdown(f"##### 🏆 Top productos vendidos con **{medio_analizar}**")
                
                df_prod_medio = df_medio_detail.groupby('nombre_producto')['importe'].sum().sort_values(ascending=False).head(5)
                st.bar_chart(df_prod_medio, horizontal=True, height=300)
        
        else:
            st.error("⚠️ No se encontró el dataset.")
    
    st.divider()
    
    # Footer con información adicional
    with st.expander("📚 Información del Dashboard"):
        st.markdown("""
        ### 📊 Dashboard Tienda Aurelion
        
        **Secciones disponibles:**
        - **General:** Vista global de métricas principales y evolución temporal
        - **Clientes:** Análisis de comportamiento, top clientes y distribución geográfica
        - **Productos:** Productos más vendidos, categorías y mapas de calor
        - **Medios de Pago:** Distribución y análisis de métodos de pago
        - **Dashboard Completo:** Embed del reporte completo de Power BI (requiere configuración)
        
        **Características:**
        - ✅ Visualizaciones interactivas con Streamlit
        - ✅ Filtros dinámicos por categoría, ciudad y medio de pago
        - ✅ KPIs actualizados en tiempo real
        - ✅ Exportación de datos disponible
        - ✅ Compatible con Power BI Embedded
        
        **Datos:**
        - Período: Febrero - Junio 2024
        - Ciudades: Córdoba, Carlos Paz, Villa María, Río Cuarto, Alta Gracia, Mendioloza
        - Categorías: Alimentos, Limpieza, Otros
        - Medios de Pago: Efectivo, QR, Tarjeta, Transferencia
        """)
    
    # Botón de descarga del PDF
    if os.path.exists("AurelionVentas_v2.pdf"):
        with open("AurelionVentas_v2.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Descargar Dashboard Completo (PDF)",
                data=pdf_file,
                file_name="TiendaAurelion_Dashboard.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# =====================================================
# 🚪 SALIR
# =====================================================
elif menu == "salir":
    st.title("👋 Gracias por usar la aplicación")
    st.write("Puedes cerrar esta pestaña o volver a seleccionar una opción en el menú lateral.")

# -----------------------------------------------------
# PIE DE PÁGINA
# -----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado para el Proyecto **Aurelion** — IBM 2025")
