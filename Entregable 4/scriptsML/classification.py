"""
Módulo para la Clasificación de Medio de Pago.
Diseñado para integrarse con app.py (Streamlit).

Objetivo: Predecir qué medio de pago usará un cliente según características de la compra.

Funciones públicas principales:
- preparar_datos_clasificacion(df) -> df_clasificacion con features
- visualizar_datos_clasificacion(df_clasificacion) -> Figure
- preparar_entrenamiento(df_clasificacion, test_size) -> X_train, X_test, y_train, y_test, encoders
- crear_modelo() -> clasificador
- entrenar_modelo(model, X_train, y_train) -> modelo entrenado
- hacer_predicciones(model, X_test) -> y_pred
- evaluar_modelo(y_test, y_pred, classes) -> dict con métricas
- plot_matriz_confusion(y_test, y_pred, classes) -> Figure
- plot_importancia_features(model, feature_names) -> Figure
- interpretar_modelo_texto(metrics, importancias) -> str
- run_pipeline(df, test_size) -> dict completo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# 1. PREPARAR DATOS PARA CLASIFICACIÓN
# =====================================================
def preparar_datos_clasificacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset para clasificar medio de pago.
    
    Features creadas:
    - Monto total de la compra (importe)
    - Categoría del producto
    - Cantidad de productos
    - Día de la semana
    - Mes del año
    - Ciudad del cliente
    - Historial del cliente (compras previas, gasto promedio)
    
    Target: medio_pago
    
    Returns:
        df_clasificacion: DataFrame con features y target
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    # Feature 1: Día de la semana (0=lunes, 6=domingo)
    df["dia_semana"] = df["fecha"].dt.dayofweek
    
    # Feature 2: Mes del año
    df["mes"] = df["fecha"].dt.month
    
    # Feature 3: Es fin de semana (binario)
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    
    # Feature 4: Rango de precio (bajo, medio, alto)
    df["rango_precio"] = pd.cut(df["importe"], bins=3, labels=["bajo", "medio", "alto"])
    
    # Feature 5: Historial del cliente (gasto promedio previo)
    # Ordenar por cliente y fecha
    df = df.sort_values(["id_cliente", "fecha"]).reset_index(drop=True)
    
    # Calcular gasto acumulado previo por cliente
    df["gasto_acumulado_cliente"] = df.groupby("id_cliente")["importe"].cumsum().shift(1).fillna(0)
    df["compras_previas_cliente"] = df.groupby("id_cliente").cumcount()
    
    # Feature 6: Gasto promedio del cliente (excluyendo compra actual)
    df["gasto_promedio_cliente"] = df["gasto_acumulado_cliente"] / (df["compras_previas_cliente"] + 1)
    df["gasto_promedio_cliente"] = df["gasto_promedio_cliente"].fillna(0)
    
    # Seleccionar features relevantes
    df_clasificacion = df[[
        "importe",
        "cantidad",
        "categoria",
        "ciudad",
        "mes",
        "dia_semana",
        "es_fin_semana",
        "rango_precio",
        "compras_previas_cliente",
        "gasto_promedio_cliente",
        "medio_pago"  # Target
    ]].copy()
    
    # Validación
    if len(df_clasificacion) < 30:
        raise ValueError(
            f"⚠️ Datos insuficientes para clasificación.\n"
            f"Se necesitan al menos 30 transacciones, tienes {len(df_clasificacion)}."
        )
    
    return df_clasificacion


# =====================================================
# 2. VISUALIZAR DATOS DE CLASIFICACIÓN
# =====================================================
def visualizar_datos_clasificacion(df_clasificacion: pd.DataFrame) -> plt.Figure:
    """
    Muestra 4 gráficos exploratorios:
    1. Distribución de medios de pago
    2. Medio de pago por categoría
    3. Importe promedio por medio de pago
    4. Uso de medio de pago por día de la semana
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distribución de medios de pago
    medio_pago_counts = df_clasificacion["medio_pago"].value_counts()
    axes[0, 0].bar(medio_pago_counts.index, medio_pago_counts.values, color="steelblue", alpha=0.7)
    axes[0, 0].set_xlabel("Medio de Pago", fontsize=11)
    axes[0, 0].set_ylabel("Cantidad de Transacciones", fontsize=11)
    axes[0, 0].set_title("💳 Distribución de Medios de Pago", fontsize=13, fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Medio de pago por categoría (heatmap)
    crosstab = pd.crosstab(df_clasificacion["categoria"], df_clasificacion["medio_pago"])
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1])
    axes[0, 1].set_xlabel("Medio de Pago", fontsize=11)
    axes[0, 1].set_ylabel("Categoría", fontsize=11)
    axes[0, 1].set_title("📊 Medio de Pago por Categoría", fontsize=13, fontweight="bold")
    
    # 3. Importe promedio por medio de pago
    importe_promedio = df_clasificacion.groupby("medio_pago")["importe"].mean().sort_values(ascending=False)
    axes[1, 0].barh(importe_promedio.index, importe_promedio.values, color="coral", alpha=0.7)
    axes[1, 0].set_xlabel("Importe Promedio ($)", fontsize=11)
    axes[1, 0].set_title("💰 Importe Promedio por Medio de Pago", fontsize=13, fontweight="bold")
    
    # 4. Uso por día de la semana
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    crosstab_dia = pd.crosstab(df_clasificacion["dia_semana"], df_clasificacion["medio_pago"])
    crosstab_dia.index = [dias[i] for i in crosstab_dia.index]
    crosstab_dia.plot(kind="bar", stacked=True, ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_xlabel("Día de la Semana", fontsize=11)
    axes[1, 1].set_ylabel("Cantidad de Transacciones", fontsize=11)
    axes[1, 1].set_title("📅 Uso de Medio de Pago por Día", fontsize=13, fontweight="bold")
    axes[1, 1].legend(title="Medio de Pago", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


# =====================================================
# 3. PREPARAR ENTRENAMIENTO
# =====================================================
def preparar_entrenamiento(df_clasificacion: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepara los datos para entrenamiento, codificando variables categóricas.
    
    Returns:
        X_train, X_test, y_train, y_test, encoders (dict con label encoders)
    """
    df = df_clasificacion.copy()
    
    # Encoders para variables categóricas
    encoders = {}
    
    # Codificar categoría
    le_categoria = LabelEncoder()
    df["categoria_encoded"] = le_categoria.fit_transform(df["categoria"])
    encoders["categoria"] = le_categoria
    
    # Codificar ciudad
    le_ciudad = LabelEncoder()
    df["ciudad_encoded"] = le_ciudad.fit_transform(df["ciudad"])
    encoders["ciudad"] = le_ciudad
    
    # Codificar rango_precio
    le_rango = LabelEncoder()
    df["rango_precio_encoded"] = le_rango.fit_transform(df["rango_precio"])
    encoders["rango_precio"] = le_rango
    
    # Codificar target (medio_pago)
    le_medio_pago = LabelEncoder()
    y = le_medio_pago.fit_transform(df["medio_pago"])
    encoders["medio_pago"] = le_medio_pago
    
    # Features seleccionadas
    feature_cols = [
        "importe",
        "cantidad",
        "categoria_encoded",
        "ciudad_encoded",
        "mes",
        "dia_semana",
        "es_fin_semana",
        "rango_precio_encoded",
        "compras_previas_cliente",
        "gasto_promedio_cliente"
    ]
    
    X = df[feature_cols]
    
    # División train/test con estratificación (mantiene proporción de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, encoders


# =====================================================
# 4. CREAR Y ENTRENAR MODELO
# =====================================================
def crear_modelo():
    """Crea un clasificador Random Forest."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )


def entrenar_modelo(model, X_train, y_train):
    """Entrena el modelo con los datos de entrenamiento."""
    model.fit(X_train, y_train)
    return model


# =====================================================
# 5. PREDICCIONES Y EVALUACIÓN
# =====================================================
def hacer_predicciones(model, X_test):
    """Realiza predicciones sobre el conjunto de test."""
    return model.predict(X_test)


def evaluar_modelo(y_test, y_pred, classes):
    """
    Calcula métricas de clasificación.
    
    Args:
        y_test: valores reales
        y_pred: valores predichos
        classes: nombres de las clases (lista de strings)
    
    Returns:
        dict con métricas
    """
    accuracy = accuracy_score(y_test, y_pred)
    
    # Métricas ponderadas (weighted) para clases desbalanceadas
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Reporte detallado por clase
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report
    }


# =====================================================
# 6. VISUALIZAR RESULTADOS
# =====================================================
def plot_matriz_confusion(y_test, y_pred, classes) -> plt.Figure:
    """
    Matriz de confusión normalizada.
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel("Predicho", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("🎯 Matriz de Confusión (Normalizada)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    return fig


def plot_importancia_features(model, feature_names) -> plt.Figure:
    """
    Gráfico de importancia de features del Random Forest.
    """
    # Las importancias se obtienen de acuerdo a los siguientes pasos:
    # 1. Para cada árbol en el bosque, se calcula la importancia de cada feature
    #    como la suma de las reducciones de impureza (Gini) que aporta esa feature.
    # 2. Luego, se promedian estas importancias a través de todos los árboles.
    # 3. Finalmente, se normalizan para que sumen 1.
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Nombres descriptivos
    nombres_descriptivos = {
        "importe": "Monto de la compra",
        "cantidad": "Cantidad de productos",
        "categoria_encoded": "Categoría del producto",
        "ciudad_encoded": "Ciudad del cliente",
        "mes": "Mes del año",
        "dia_semana": "Día de la semana",
        "es_fin_semana": "Es fin de semana",
        "rango_precio_encoded": "Rango de precio",
        "compras_previas_cliente": "Compras previas del cliente",
        "gasto_promedio_cliente": "Gasto promedio del cliente"
    }
    
    nombres = [nombres_descriptivos.get(feature_names[i], feature_names[i]) for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(importances)), importances[indices], color="steelblue", alpha=0.7)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(nombres)
    ax.set_xlabel("Importancia", fontsize=11)
    ax.set_title("📊 Importancia de Variables en el Modelo", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_distribucion_predicciones(y_test, y_pred, classes) -> plt.Figure:
    """
    Compara distribución de valores reales vs predichos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Valores reales
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    axes[0].bar([classes[i] for i in unique_test], counts_test, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Medio de Pago", fontsize=11)
    axes[0].set_ylabel("Frecuencia", fontsize=11)
    axes[0].set_title("📊 Distribución Real (Test Set)", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Valores predichos
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    axes[1].bar([classes[i] for i in unique_pred], counts_pred, color="coral", alpha=0.7)
    axes[1].set_xlabel("Medio de Pago", fontsize=11)
    axes[1].set_ylabel("Frecuencia", fontsize=11)
    axes[1].set_title("🔮 Distribución Predicha", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


# =====================================================
# 7. INTERPRETACIÓN EN TEXTO
# =====================================================
def predecir_nueva_transaccion(model, encoders, transaccion_dict):
    """
    Predice el medio de pago para una nueva transacción.
    
    Args:
        model: modelo entrenado
        encoders: diccionario con label encoders
        transaccion_dict: dict con keys:
            - importe
            - cantidad
            - categoria
            - ciudad
            - mes
            - dia_semana
            - es_fin_semana
            - rango_precio
            - compras_previas_cliente
            - gasto_promedio_cliente
    
    Returns:
        dict con predicción y probabilidades
    """
    # Codificar variables categóricas
    categoria_encoded = encoders["categoria"].transform([transaccion_dict["categoria"]])[0]
    ciudad_encoded = encoders["ciudad"].transform([transaccion_dict["ciudad"]])[0]
    rango_precio_encoded = encoders["rango_precio"].transform([transaccion_dict["rango_precio"]])[0]
    
    # Crear array con features
    X_nuevo = np.array([[
        transaccion_dict["importe"],
        transaccion_dict["cantidad"],
        categoria_encoded,
        ciudad_encoded,
        transaccion_dict["mes"],
        transaccion_dict["dia_semana"],
        transaccion_dict["es_fin_semana"],
        rango_precio_encoded,
        transaccion_dict["compras_previas_cliente"],
        transaccion_dict["gasto_promedio_cliente"]
    ]])
    
    # Predecir
    prediccion = model.predict(X_nuevo)[0]
    probabilidades = model.predict_proba(X_nuevo)[0]
    
    # Decodificar predicción
    medio_pago_predicho = encoders["medio_pago"].inverse_transform([prediccion])[0]
    
    # Crear dict con probabilidades por clase
    probs_dict = {}
    for i, clase in enumerate(encoders["medio_pago"].classes_):
        probs_dict[clase] = probabilidades[i]
    
    return {
        "medio_pago_predicho": medio_pago_predicho,
        "probabilidades": probs_dict
    }


def analizar_errores(y_test, y_pred, X_test, encoders):
    """
    Analiza los casos mal clasificados del modelo.
    
    Returns:
        DataFrame con errores y sus características
    """
    # Identificar errores
    errores = y_test != y_pred
    indices_errores = np.where(errores)[0]
    
    if len(indices_errores) == 0:
        return None
    
    # Crear DataFrame con errores
    errores_data = []
    
    for idx in indices_errores:
        real = encoders["medio_pago"].inverse_transform([y_test[idx]])[0]
        pred = encoders["medio_pago"].inverse_transform([y_pred[idx]])[0]
        
        errores_data.append({
            "Índice": idx,
            "Real": real,
            "Predicho": pred,
            "Importe": X_test.iloc[idx]["importe"],
            "Cantidad": X_test.iloc[idx]["cantidad"],
            "Mes": X_test.iloc[idx]["mes"],
            "Día Semana": X_test.iloc[idx]["dia_semana"],
            "Compras Previas": X_test.iloc[idx]["compras_previas_cliente"]
        })
    
    return pd.DataFrame(errores_data)
    
def interpretar_modelo_texto(metrics: dict, classes: list, n_observaciones: int) -> str:    
    """
    Genera una interpretación en texto del modelo.
    """
    texto = "### 📊 Interpretación del Modelo de Clasificación\n\n"
    
    texto += f"**Observaciones de entrenamiento:** {n_observaciones} transacciones\n\n"
    texto += f"**Clases a predecir:** {', '.join(classes)}\n\n"
    
    texto += "#### 📈 Métricas Generales\n\n"
    texto += f"- **Accuracy (Precisión Global):** {metrics['accuracy']:.2%}\n"
    texto += f"  → El modelo acierta correctamente el {metrics['accuracy']*100:.1f}% de las predicciones\n\n"
    
    texto += f"- **Precision (Promedio Ponderado):** {metrics['precision']:.2%}\n"
    texto += f"  → Cuando predice un medio de pago, acierta el {metrics['precision']*100:.1f}% de las veces\n\n"
    
    texto += f"- **Recall (Promedio Ponderado):** {metrics['recall']:.2%}\n"
    texto += f"  → Detecta correctamente el {metrics['recall']*100:.1f}% de los casos de cada medio de pago\n\n"
    
    texto += f"- **F1-Score:** {metrics['f1_score']:.2%}\n"
    texto += f"  → Balance entre precisión y recall\n\n"
    
    # Rendimiento por clase
    texto += "#### 🎯 Rendimiento por Medio de Pago\n\n"
    
    for clase in classes:
        if clase in metrics['report']:
            clase_metrics = metrics['report'][clase]
            texto += f"**{clase}:**\n"
            texto += f"- Precision: {clase_metrics['precision']:.2%}\n"
            texto += f"- Recall: {clase_metrics['recall']:.2%}\n"
            texto += f"- Casos en test: {int(clase_metrics['support'])}\n\n"
    
    # Calidad del modelo
    texto += "#### 🏆 Calidad del Modelo\n\n"
    
    if n_observaciones < 100:
        texto += f"⚠️ **Advertencia:** Dataset pequeño ({n_observaciones} transacciones). Para mejorar:\n"
        texto += "- Recopilar más transacciones históricas\n"
        texto += "- Asegurar que todas las clases tengan ejemplos suficientes\n\n"
    
    if metrics["accuracy"] > 0.8:
        texto += "✅ **Excelente:** El modelo tiene una precisión superior al 80%\n"
    elif metrics["accuracy"] > 0.6:
        texto += "🟢 **Bueno:** El modelo tiene una precisión aceptable (>60%)\n"
    elif metrics["accuracy"] > 0.4:
        texto += "🟡 **Moderado:** El modelo funciona mejor que azar, pero hay margen de mejora\n"
    else:
        texto += "🔴 **Bajo:** El modelo tiene baja precisión. Considera más datos o features\n"
    
    texto += "\n#### ℹ️ Aplicaciones Prácticas\n\n"
    texto += "Con este modelo puedes:\n"
    texto += "- 🎯 **Personalizar la experiencia de pago:** Mostrar primero el método que el cliente probablemente use\n"
    texto += "- 💳 **Optimizar promociones:** Ofrecer descuentos según método de pago predicho\n"
    texto += "- 📊 **Analizar comportamiento:** Entender qué factores influyen en la elección de pago\n"
    texto += "- 🔍 **Detectar anomalías:** Identificar transacciones con métodos de pago inusuales\n"
    
    return texto


# =====================================================
# 8. PIPELINE COMPLETO
# =====================================================
def run_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Ejecuta el pipeline completo de clasificación de medio de pago.
    
    Returns:
        dict con todos los resultados
    """
    # 1. Preparar datos
    df_clasificacion = preparar_datos_clasificacion(df)
    
    # 2. Visualizar
    fig_exploracion = visualizar_datos_clasificacion(df_clasificacion)
    
    # 3. Preparar entrenamiento
    X_train, X_test, y_train, y_test, encoders = preparar_entrenamiento(
        df_clasificacion, test_size=test_size, random_state=random_state
    )
    
    # Obtener nombres de clases
    classes = encoders["medio_pago"].classes_
    
    # 4. Crear y entrenar
    model = crear_modelo()
    model = entrenar_modelo(model, X_train, y_train)
    
    # 5. Predecir y evaluar
    y_pred = hacer_predicciones(model, X_test)
    metrics = evaluar_modelo(y_test, y_pred, classes)
    
    # 6. Visualizar resultados
    fig_matriz = plot_matriz_confusion(y_test, y_pred, classes)
    fig_importancia = plot_importancia_features(model, X_train.columns.tolist())
    fig_distribucion = plot_distribucion_predicciones(y_test, y_pred, classes)
    
    # 7. Analizar errores
    df_errores = analizar_errores(y_test, y_pred, X_test, encoders)
    
    # 8. Interpretación en texto
    texto_interpretacion = interpretar_modelo_texto(metrics, classes, len(X_train))
    
    return {
        "df_clasificacion": df_clasificacion,
        "model": model,
        "encoders": encoders,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "classes": classes,
        "metrics": metrics,
        "df_errores": df_errores,
        "fig_exploracion": fig_exploracion,
        "fig_matriz": fig_matriz,
        "fig_importancia": fig_importancia,
        "fig_distribucion": fig_distribucion,
        "texto_interpretacion": texto_interpretacion
    }