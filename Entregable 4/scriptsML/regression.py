"""
Módulo para la Predicción de Ventas por Producto Individual.
Diseñado para integrarse con app.py (Streamlit).

Objetivo: Predecir cuántas unidades se venderán de cada producto el próximo mes.

Funciones públicas principales:
- preparar_datos_productos(df) -> df_productos con features
- visualizar_datos_productos(df_productos) -> Figure
- preparar_entrenamiento(df_productos, test_size) -> X_train, X_test, y_train, y_test
- crear_modelo() -> modelo de regresión
- entrenar_modelo(model, X_train, y_train) -> modelo entrenado
- hacer_predicciones(model, X_test) -> y_pred
- evaluar_modelo(y_test, y_pred) -> dict con métricas
- interpretar_coeficientes(model, feature_names) -> DataFrame
- predecir_productos_proximos_meses(df, model, n_meses) -> DataFrame con predicciones
- interpretar_modelo_texto(model, metrics, coef_df) -> str
- run_pipeline(df, test_size) -> dict completo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================================================
# 1. PREPARAR DATOS DE PRODUCTOS CON FEATURES
# =====================================================
def preparar_datos_productos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el dataframe transaccional en un dataset a nivel producto-mes
    con features para predecir ventas futuras.
    
    Features creadas:
    - Historial de ventas del producto (últimos 3 meses)
    - Categoría del producto (encoded)
    - Precio unitario promedio
    - Tendencia de ventas (creciente/decreciente)
    - Mes del año (1-12 para estacionalidad)
    - Promedio de ventas de la categoría
    - Cantidad de meses con ventas (popularidad)
    
    Returns:
        df_productos: DataFrame con una fila por producto-mes
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes"] = df["fecha"].dt.to_period("M")
    
    # Agrupar por producto y mes
    df_prod_mes = df.groupby(["id_producto", "nombre_producto", "categoria", "mes"]).agg({
        "cantidad": "sum",
        "precio_unitario": "mean",
        "importe": "sum"
    }).reset_index()
    
    # Convertir periodo a timestamp para ordenar
    df_prod_mes["mes_dt"] = df_prod_mes["mes"].dt.to_timestamp()
    df_prod_mes = df_prod_mes.sort_values(["id_producto", "mes_dt"]).reset_index(drop=True)
    
    # Feature 1: Mes del año (estacionalidad)
    df_prod_mes["mes_año"] = df_prod_mes["mes_dt"].dt.month
    
    # Feature 2: Lags de ventas (últimos 3 meses)
    df_prod_mes["lag_1"] = df_prod_mes.groupby("id_producto")["cantidad"].shift(1)
    df_prod_mes["lag_2"] = df_prod_mes.groupby("id_producto")["cantidad"].shift(2)
    df_prod_mes["lag_3"] = df_prod_mes.groupby("id_producto")["cantidad"].shift(3)
    
    # Feature 3: Promedio móvil de 3 meses (CRÍTICO: solo usar datos pasados)
    # Primero hacer shift(1) para desplazar, luego calcular rolling
    df_prod_mes["rolling_mean_3"] = (
        df_prod_mes.groupby("id_producto")["cantidad"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    
    # Feature 4: Tendencia (diferencia entre últimos 2 meses) - SOLO DATOS PASADOS
    df_prod_mes["tendencia"] = df_prod_mes.groupby("id_producto")["cantidad"].transform(
        lambda x: x.shift(1) - x.shift(2)
    )
    df_prod_mes["tendencia"] = df_prod_mes["tendencia"].fillna(0)
    
    # Feature 5: Promedio de ventas de la categoría ese mes (USAR SHIFT PARA EVITAR LEAKAGE)
    # Calcular el promedio de la categoría del mes ANTERIOR
    df_prod_mes["categoria_promedio"] = df_prod_mes.groupby(["categoria", "mes"])["cantidad"].transform("mean")
    df_prod_mes["categoria_promedio"] = df_prod_mes.groupby("id_producto")["categoria_promedio"].shift(1)
    df_prod_mes["categoria_promedio"] = df_prod_mes["categoria_promedio"].fillna(df_prod_mes.groupby("categoria")["cantidad"].transform("mean"))
    
    # Feature 6: Cantidad de meses con ventas (popularidad del producto)
    meses_activos = df_prod_mes.groupby("id_producto")["mes"].transform("count")
    df_prod_mes["meses_activos"] = meses_activos
    
    # Feature 7: Precio unitario (normalizado)
    df_prod_mes["precio_unit"] = df_prod_mes["precio_unitario"]
    
    # Eliminar filas sin lags (primeros 3 meses de cada producto)
    df_prod_mes = df_prod_mes.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)
    
    # Validación
    if len(df_prod_mes) < 20:
        raise ValueError(
            f"⚠️ Datos insuficientes después de crear features.\n"
            f"Se necesitan al menos 20 observaciones (producto-mes), tienes {len(df_prod_mes)}.\n"
            f"Esto puede ocurrir si los productos tienen muy pocos meses de ventas."
        )
    
    return df_prod_mes


# =====================================================
# 2. VISUALIZAR DATOS DE PRODUCTOS
# =====================================================
def visualizar_datos_productos(df_productos: pd.DataFrame) -> plt.Figure:
    """
    Muestra 4 gráficos exploratorios:
    1. Top 10 productos por ventas totales
    2. Ventas promedio por categoría
    3. Distribución de ventas mensuales
    4. Correlación entre precio y cantidad vendida
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Top 10 productos más vendidos
    top_productos = df_productos.groupby("nombre_producto")["cantidad"].sum().nlargest(10).reset_index()
    axes[0, 0].barh(top_productos["nombre_producto"], top_productos["cantidad"], color="steelblue")
    axes[0, 0].set_xlabel("Cantidad Total Vendida", fontsize=11)
    axes[0, 0].set_title("📦 Top 10 Productos Más Vendidos", fontsize=13, fontweight="bold")
    axes[0, 0].invert_yaxis()
    
    # 2. Ventas promedio por categoría
    cat_promedio = df_productos.groupby("categoria")["cantidad"].mean().sort_values(ascending=False).reset_index()
    axes[0, 1].bar(cat_promedio["categoria"], cat_promedio["cantidad"], color="coral", alpha=0.7)
    axes[0, 1].set_xlabel("Categoría", fontsize=11)
    axes[0, 1].set_ylabel("Cantidad Promedio Mensual", fontsize=11)
    axes[0, 1].set_title("📊 Ventas Promedio por Categoría", fontsize=13, fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Distribución de ventas mensuales
    axes[1, 0].hist(df_productos["cantidad"], bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[1, 0].set_xlabel("Cantidad Vendida (unidades)", fontsize=11)
    axes[1, 0].set_ylabel("Frecuencia", fontsize=11)
    axes[1, 0].set_title("📈 Distribución de Ventas Mensuales", fontsize=13, fontweight="bold")
    axes[1, 0].axvline(df_productos["cantidad"].mean(), color='red', linestyle='--', 
                       label=f'Media: {df_productos["cantidad"].mean():.1f}')
    axes[1, 0].legend()
    
    # 4. Precio vs Cantidad (scatter)
    axes[1, 1].scatter(df_productos["precio_unit"], df_productos["cantidad"], alpha=0.5, s=30)
    axes[1, 1].set_xlabel("Precio Unitario ($)", fontsize=11)
    axes[1, 1].set_ylabel("Cantidad Vendida", fontsize=11)
    axes[1, 1].set_title("💰 Relación Precio - Ventas", fontsize=13, fontweight="bold")
    
    # Línea de tendencia
    z = np.polyfit(df_productos["precio_unit"], df_productos["cantidad"], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df_productos["precio_unit"], p(df_productos["precio_unit"]), 
                    "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    return fig


# =====================================================
# 3. PREPARAR ENTRENAMIENTO
# =====================================================
def preparar_entrenamiento(df_productos: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepara los datos para entrenamiento con ESTRATEGIA MEJORADA para datasets pequeños.
    
    En lugar de división temporal estricta (que puede sesgar train/test con datasets chicos),
    usa división por PRODUCTO: algunos productos completos van a test.
    Esto asegura distribuciones similares.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoders
    """
    df = df_productos.copy()
    
    # Codificar categoría
    le_categoria = LabelEncoder()
    df["categoria_encoded"] = le_categoria.fit_transform(df["categoria"])
    
    # Features seleccionadas - MODELO MEJORADO
    feature_cols = [
        "lag_1",                # Ventas mes anterior (MÁS IMPORTANTE)
        "lag_2",                # Ventas hace 2 meses
        "lag_3",                # Ventas hace 3 meses
        "precio_unit",          # Precio del producto
        "categoria_encoded",    # Categoría
    ]
    
    # ESTRATEGIA: Dividir por PRODUCTO, no por tiempo
    # Esto asegura que train y test tengan distribuciones similares
    productos_unicos = df["id_producto"].unique()
    np.random.seed(random_state)
    np.random.shuffle(productos_unicos)
    
    # Calcular cuántos productos van a test
    n_productos_test = max(1, int(len(productos_unicos) * test_size))
    productos_test = productos_unicos[:n_productos_test]
    productos_train = productos_unicos[n_productos_test:]
    
    # Filtrar por producto
    df_train = df[df["id_producto"].isin(productos_train)]
    df_test = df[df["id_producto"].isin(productos_test)]
    
    X_train = df_train[feature_cols]
    y_train = df_train["cantidad"]
    X_test = df_test[feature_cols]
    y_test = df_test["cantidad"]
    
    return X_train, X_test, y_train, y_test, le_categoria


# =====================================================
# 4. CREAR Y ENTRENAR MODELO
# =====================================================
def crear_modelo():
    """Crea un modelo de regresión lineal."""
    return LinearRegression()


def entrenar_modelo(model, X_train, y_train):
    """Entrena el modelo con los datos de entrenamiento."""
    model.fit(X_train, y_train)
    return model


# =====================================================
# 5. PREDICCIONES Y EVALUACIÓN
# =====================================================
def hacer_predicciones(model, X_test):
    """Realiza predicciones sobre el conjunto de test."""
    y_pred = model.predict(X_test)
    # No permitir predicciones negativas
    y_pred = np.maximum(y_pred, 0)
    return y_pred


def evaluar_modelo(y_test, y_pred):
    """
    Calcula métricas: MAE, RMSE, R2 y MAPE.
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    
    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    mse = mean_squared_error(y_test_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_arr, y_pred_arr)
    
    # MAPE (evitar divisiones por cero)
    mask = y_test_arr != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_arr[mask] - y_pred_arr[mask]) / y_test_arr[mask])) * 100
    else:
        mape = 0
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# =====================================================
# 6. INTERPRETACIÓN DE COEFICIENTES
# =====================================================
def interpretar_coeficientes(model, feature_names):
    """
    Devuelve DataFrame con los coeficientes del modelo ordenados por importancia.
    """
    coefs = model.coef_
    
    # Nombres descriptivos
    nombres_descriptivos = {
        "lag_1": "Ventas del mes anterior",
        "lag_2": "Ventas de hace 2 meses",
        "lag_3": "Ventas de hace 3 meses",
        "precio_unit": "Precio unitario",
        "categoria_encoded": "Categoría del producto"
    }
    
    df_coef = pd.DataFrame({
        "Variable": [nombres_descriptivos.get(f, f) for f in feature_names],
        "Coeficiente": coefs,
        "Impacto": ["↑ Aumenta ventas" if c > 0 else "↓ Reduce ventas" for c in coefs]
    })
    df_coef["Importancia_Abs"] = df_coef["Coeficiente"].abs()
    df_coef = df_coef.sort_values("Importancia_Abs", ascending=False).drop(columns=["Importancia_Abs"])
    
    return df_coef.reset_index(drop=True)


# =====================================================
# 7. VISUALIZAR PREDICCIONES
# =====================================================
def plot_predicciones(y_test, y_pred) -> plt.Figure:
    """
    Gráficos de evaluación del modelo:
    1. Real vs Predicho (scatter)
    2. Distribución de errores (residuales)
    3. Productos con mayor error
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    errores = y_test_arr - y_pred_arr
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Real vs Predicho
    axes[0].scatter(y_test_arr, y_pred_arr, alpha=0.5, s=40)
    min_val = min(y_test_arr.min(), y_pred_arr.min())
    max_val = max(y_test_arr.max(), y_pred_arr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
    axes[0].set_xlabel("Cantidad Real Vendida", fontsize=11)
    axes[0].set_ylabel("Cantidad Predicha", fontsize=11)
    axes[0].set_title("📊 Real vs Predicho", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Distribución de errores
    axes[1].hist(errores, bins=30, alpha=0.7, color="purple", edgecolor="black")
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    axes[1].set_xlabel("Error (Real - Predicho)", fontsize=11)
    axes[1].set_ylabel("Frecuencia", fontsize=11)
    axes[1].set_title("📉 Distribución de Errores", fontsize=13, fontweight="bold")
    axes[1].legend()
    
    # 3. Errores absolutos ordenados
    errores_abs = np.abs(errores)
    errores_ordenados = np.sort(errores_abs)[::-1][:30]  # Top 30 errores
    axes[2].plot(errores_ordenados, marker='o', linewidth=2)
    axes[2].set_xlabel("Productos (ordenados por error)", fontsize=11)
    axes[2].set_ylabel("Error Absoluto", fontsize=11)
    axes[2].set_title("🔍 Top 30 Productos con Mayor Error", fontsize=13, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distribucion_real_vs_predicho(y_test, y_pred) -> plt.Figure:
    """
    Histograma comparando distribución de valores reales vs predichos.
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    fig = plt.figure(figsize=(10, 6))

    plt.hist(y_test_arr, bins=30, alpha=0.5, label="Valores Reales", color="steelblue", edgecolor="black")
    plt.hist(y_pred_arr, bins=30, alpha=0.5, label="Valores Predichos", color="coral", edgecolor="black")

    plt.xlabel("Cantidad Vendida", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.title("📊 Distribución de Valores Reales vs Predichos", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()

    return fig


def plot_kde_real_vs_predicho(y_test, y_pred) -> plt.Figure:
    """
    Comparación de densidades de valores reales vs predichos usando KDE.
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    fig = plt.figure(figsize=(10, 6))

    sns.kdeplot(y_test_arr, label="Valores Reales", fill=True, alpha=0.5, color="steelblue")
    sns.kdeplot(y_pred_arr, label="Valores Predichos", fill=True, alpha=0.5, color="coral")

    plt.xlabel("Cantidad Vendida", fontsize=11)
    plt.ylabel("Densidad", fontsize=11)
    plt.title("📈 Distribución de Valores Reales vs Predichos (KDE)", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()

    return fig


# =====================================================
# 8. PREDECIR PRÓXIMOS MESES
# =====================================================
def predecir_productos_proximos_meses(df: pd.DataFrame, model, le_categoria, n_meses: int = 1):
    """
    Predice las ventas de cada producto para los próximos n_meses.
    VERSIÓN MEJORADA: Usa lags + precio + categoría
    
    Returns:
        DataFrame con predicciones por producto y mes futuro
    """
    df_productos = preparar_datos_productos(df)
    
    # Obtener último mes conocido
    ultimo_mes = df_productos["mes_dt"].max()
    
    # Para cada producto, obtener sus últimas features
    productos_unicos = df_productos["id_producto"].unique()
    
    predicciones_futuras = []
    
    for producto_id in productos_unicos:
        df_prod = df_productos[df_productos["id_producto"] == producto_id].sort_values("mes_dt")
        
        if len(df_prod) < 1:
            continue
        
        # Última información del producto
        ultima_fila = df_prod.iloc[-1]
        
        # Features para predicción
        mes_futuro = ultimo_mes + pd.DateOffset(months=1)
        
        X_futuro = pd.DataFrame({
            "lag_1": [ultima_fila["cantidad"]],  # Última venta conocida
            "lag_2": [ultima_fila["lag_1"]],
            "lag_3": [ultima_fila["lag_2"]],
            "precio_unit": [ultima_fila["precio_unit"]],
            "categoria_encoded": [le_categoria.transform([ultima_fila["categoria"]])[0]]
        })
        
        # Predecir
        pred = model.predict(X_futuro)[0]
        pred = max(0, pred)  # No negativos
        
        predicciones_futuras.append({
            "id_producto": producto_id,
            "nombre_producto": ultima_fila["nombre_producto"],
            "categoria": ultima_fila["categoria"],
            "mes_predicho": mes_futuro.strftime("%Y-%m"),
            "cantidad_predicha": round(pred, 2),
            "precio_unitario": ultima_fila["precio_unit"],
            "ingresos_estimados": round(pred * ultima_fila["precio_unit"], 2)
        })
    
    df_predicciones = pd.DataFrame(predicciones_futuras)
    df_predicciones = df_predicciones.sort_values("cantidad_predicha", ascending=False)
    
    return df_predicciones.reset_index(drop=True)


# =====================================================
# 9. INTERPRETACIÓN EN TEXTO
# =====================================================
def interpretar_modelo_texto(model, metrics, coef_df, n_observaciones):
    """
    Genera una interpretación en texto del modelo.
    """
    texto = "### 📊 Interpretación del Modelo de Predicción por Producto\n\n"
    
    texto += f"**Observaciones de entrenamiento:** {n_observaciones} combinaciones producto-mes\n\n"
    texto += f"**Valor base (intercepto):** {model.intercept_:.2f} unidades\n\n"
    
    texto += "#### 🔍 Variables Más Influyentes\n\n"
    
    for i, row in coef_df.head(5).iterrows():
        var = row["Variable"]
        coef = row["Coeficiente"]
        impacto = row["Impacto"]
        
        texto += f"{i+1}. **{var}** ({impacto})\n"
        texto += f"   - Coeficiente: {coef:.4f}\n"
        
        if "lag_1" in var.lower() or "anterior" in var.lower():
            texto += f"   - Por cada unidad vendida el mes pasado, se esperan **{abs(coef):.2f} unidades** este mes\n"
        elif "precio" in var.lower():
            texto += f"   - Por cada $1 de aumento en precio, las ventas cambian en **{coef:.2f} unidades**\n"
        elif "categoria_promedio" in var.lower():
            texto += f"   - Productos de categorías populares venden **{abs(coef):.2f} unidades más** por cada unidad promedio de la categoría\n"
        
        texto += "\n"
    
    texto += "#### 📈 Métricas de Evaluación\n\n"
    texto += f"- **MAE (Error Absoluto Medio):** {metrics['MAE']:.2f} unidades\n"
    texto += f"  → En promedio, el modelo se equivoca por ±{metrics['MAE']:.0f} unidades\n\n"
    
    texto += f"- **RMSE:** {metrics['RMSE']:.2f} unidades\n\n"
    
    texto += f"- **R² (Varianza Explicada):** {metrics['R2']:.3f}\n"
    r2_porcentaje = metrics['R2'] * 100
    texto += f"  → El modelo explica el **{r2_porcentaje:.1f}%** de la variabilidad en ventas\n\n"
    
    texto += f"- **MAPE (Error Porcentual):** {metrics['MAPE']:.2f}%\n\n"
    
    # Calidad del modelo
    texto += "#### 🎯 Calidad del Modelo\n\n"
    
    # Advertencia para datasets pequeños
    if n_observaciones < 50:
        texto += f"⚠️ **Advertencia:** El modelo fue entrenado con solo **{n_observaciones} observaciones**, lo cual limita su capacidad predictiva.\n\n"
        texto += "Para mejorar el modelo se recomienda:\n"
        texto += "- Recopilar más meses de datos históricos (ideal: 12+ meses)\n"
        texto += "- Incluir más productos en el análisis\n"
        texto += "- Usar técnicas de aumento de datos si es necesario\n\n"
    
    if metrics["R2"] > 0.7:
        texto += "✅ **Muy bueno:** El modelo explica más del 70% de las variaciones en ventas.\n"
    elif metrics["R2"] > 0.5:
        texto += "🟢 **Bueno:** El modelo tiene un poder predictivo aceptable (>50%).\n"
    elif metrics["R2"] > 0.3:
        texto += "🟡 **Moderado:** El modelo explica parcialmente las ventas (>30%).\n"
    elif metrics["R2"] > 0:
        texto += "🟠 **Limitado:** El modelo tiene bajo poder explicativo pero es mejor que el promedio simple.\n"
    else:
        texto += "🔴 **Muy bajo:** El modelo no es útil. Considera recopilar más datos o cambiar el enfoque.\n"
    
    texto += "\n"
    if metrics["MAPE"] < 20:
        texto += "✅ **Error aceptable:** Las predicciones tienen un error porcentual menor al 20%.\n"
    elif metrics["MAPE"] < 40:
        texto += "🟡 **Error moderado:** Hay margen de mejora en la precisión.\n"
    else:
        texto += "🔴 **Error alto:** Las predicciones tienen alta variabilidad.\n"
    
    texto += "\n#### ℹ️ Aplicaciones Prácticas\n\n"
    texto += "Con este modelo puedes:\n"
    texto += "- 📦 **Gestionar inventario:** Saber cuántas unidades pedir de cada producto\n"
    texto += "- 💰 **Proyectar ingresos:** Estimar ventas futuras por producto\n"
    texto += "- 🎯 **Identificar tendencias:** Detectar productos en crecimiento o declive\n"
    texto += "- 📊 **Optimizar stock:** Evitar quiebres o exceso de inventario\n"
    
    return texto


# =====================================================
# 10. PIPELINE COMPLETO
# =====================================================
def run_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Ejecuta el pipeline completo de predicción de ventas por producto.
    
    Returns:
        dict con todos los resultados
    """
    # 1. Preparar datos
    df_productos = preparar_datos_productos(df)
    
    # 2. Visualizar
    fig_exploracion = visualizar_datos_productos(df_productos)
    
    # 3. Preparar entrenamiento
    X_train, X_test, y_train, y_test, le_categoria = preparar_entrenamiento(
        df_productos, test_size=test_size, random_state=random_state
    )
    
    # 4. Crear y entrenar
    model = crear_modelo()
    model = entrenar_modelo(model, X_train, y_train)
    
    # 5. Predecir y evaluar
    y_pred = hacer_predicciones(model, X_test)
    metrics = evaluar_modelo(y_test, y_pred)
    
    # 6. Interpretar coeficientes
    coef_df = interpretar_coeficientes(model, X_train.columns.tolist())
    
    # 7. Visualizar resultados
    fig_predicciones = plot_predicciones(y_test, y_pred)
    fig_hist = plot_distribucion_real_vs_predicho(y_test, y_pred)
    fig_kde = plot_kde_real_vs_predicho(y_test, y_pred)
    
    # 8. Predecir próximo mes para todos los productos
    df_futuro = predecir_productos_proximos_meses(df, model, le_categoria, n_meses=1)
    
    # 9. Interpretación en texto
    texto_interpretacion = interpretar_modelo_texto(model, metrics, coef_df, len(X_train))
    
    return {
        "df_productos": df_productos,
        "model": model,
        "le_categoria": le_categoria,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "coef_df": coef_df,
        "df_futuro": df_futuro,
        "fig_exploracion": fig_exploracion,
        "fig_predicciones": fig_predicciones,
        "fig_hist": fig_hist,
        "fig_kde": fig_kde,
        "texto_interpretacion": texto_interpretacion
    }