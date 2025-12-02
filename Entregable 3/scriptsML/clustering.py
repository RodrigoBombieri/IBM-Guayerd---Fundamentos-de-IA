"""
Módulo para la Segmentación de Clientes usando RFM y K-Means Clustering.
Diseñado para integrarse con app.py (Streamlit).

Objetivo: Agrupar clientes según su comportamiento de compra (Recency, Frequency, Monetary).

Funciones públicas principales:
- preparar_datos_rfm(df) -> df_rfm con métricas RFM
- calcular_rfm(df) -> DataFrame con RFM por cliente
- determinar_numero_clusters(df_rfm_scaled) -> dict con métricas elbow/silhouette
- crear_modelo_clustering(n_clusters) -> KMeans
- entrenar_clustering(model, df_rfm_scaled) -> modelo entrenado + labels
- asignar_nombres_clusters(df_rfm, labels) -> DataFrame con nombres descriptivos
- visualizar_clusters_2d(df_rfm) -> Figure
- visualizar_clusters_3d(df_rfm) -> Figure (opcional)
- plot_elbow_silhouette(metricas) -> Figure
- interpretar_clusters_texto(df_rfm) -> str
- run_pipeline(df, n_clusters) -> dict completo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =====================================================
# 1. CALCULAR MÉTRICAS RFM
# =====================================================
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las métricas RFM para cada cliente.
    
    RFM:
    - Recency: Días desde la última compra
    - Frequency: Cantidad de compras realizadas
    - Monetary: Total gastado
    
    Returns:
        DataFrame con columnas: id_cliente, nombre_cliente, R, F, M
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    # Fecha de referencia (la más reciente en el dataset)
    fecha_referencia = df["fecha"].max()
    
    # Agrupar por cliente y calcular RFM
    rfm = df.groupby(["id_cliente", "nombre_cliente", "email", "ciudad"]).agg({
        "fecha": lambda x: (fecha_referencia - x.max()).days,  # Recency
        "id_venta": "nunique",  # Frequency (ventas únicas)
        "importe": "sum"  # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        "fecha": "Recency",
        "id_venta": "Frequency",
        "importe": "Monetary"
    }, inplace=True)
    
    # Validación
    if len(rfm) < 3:
        raise ValueError(
            f"⚠️ Datos insuficientes para clustering.\n"
            f"Se necesitan al menos 3 clientes, tienes {len(rfm)}."
        )
    
    return rfm


def preparar_datos_rfm(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos RFM y los normaliza para clustering.
    
    Returns:
        df_rfm: DataFrame con RFM original
        df_rfm_scaled: DataFrame con RFM normalizado
        scaler: StandardScaler ajustado
    """
    # Calcular RFM
    df_rfm = calcular_rfm(df)
    
    # Normalizar RFM (importante para K-Means)
    scaler = StandardScaler()
    rfm_features = df_rfm[["Recency", "Frequency", "Monetary"]].copy()
    
    # Escalar
    rfm_scaled = scaler.fit_transform(rfm_features)
    df_rfm_scaled = pd.DataFrame(
        rfm_scaled,
        columns=["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
    )
    
    return df_rfm, df_rfm_scaled, scaler


# =====================================================
# 2. DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS
# =====================================================
def determinar_numero_clusters(df_rfm_scaled: pd.DataFrame, max_clusters: int = 8) -> dict:
    """
    Calcula métricas Elbow y Silhouette para diferentes valores de K.
    
    Returns:
        dict con listas: k_values, inertias, silhouette_scores
    """
    k_values = range(2, min(max_clusters + 1, len(df_rfm_scaled)))
    inertias = []
    silhouette_scores_list = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_rfm_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(df_rfm_scaled, labels))
    
    return {
        "k_values": list(k_values),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list
    }


# =====================================================
# 3. CREAR Y ENTRENAR MODELO
# =====================================================
def crear_modelo_clustering(n_clusters: int = 4):
    """Crea un modelo K-Means."""
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)


def entrenar_clustering(model, df_rfm_scaled: pd.DataFrame):
    """
    Entrena el modelo y devuelve las etiquetas de cluster.
    """
    labels = model.fit_predict(df_rfm_scaled)
    return model, labels


# =====================================================
# 4. ASIGNAR NOMBRES A CLUSTERS
# =====================================================
def asignar_nombres_clusters(df_rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Asigna nombres descriptivos a cada cluster según características RFM.
    
    Lógica:
    - Recency bajo, Frequency alto, Monetary alto → Champions/VIP
    - Recency bajo, Frequency medio, Monetary medio → Leales
    - Recency alto, Frequency bajo, Monetary bajo → En Riesgo/Dormidos
    - Recency bajo, Frequency bajo → Nuevos
    """
    df = df_rfm.copy()
    df["Cluster"] = labels
    
    # Calcular promedios por cluster
    cluster_stats = df.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).reset_index()
    
    # Normalizar para comparación (0-1)
    for col in ["Recency", "Frequency", "Monetary"]:
        cluster_stats[f"{col}_norm"] = (
            (cluster_stats[col] - cluster_stats[col].min()) / 
            (cluster_stats[col].max() - cluster_stats[col].min() + 1e-10)
        )
    
    # Asignar nombres según características
    nombres_clusters = {}
    
    for _, row in cluster_stats.iterrows():
        cluster_id = row["Cluster"]
        r_norm = row["Recency_norm"]  # Menor es mejor (más reciente)
        f_norm = row["Frequency_norm"]  # Mayor es mejor
        m_norm = row["Monetary_norm"]  # Mayor es mejor
        
        # Lógica de clasificación
        if r_norm < 0.3 and f_norm > 0.7 and m_norm > 0.7:
            nombre = "🌟 Champions (VIP)"
            descripcion = "Compran frecuente, reciente y gastan mucho"
        elif r_norm < 0.4 and f_norm > 0.5:
            nombre = "💚 Clientes Leales"
            descripcion = "Compran regularmente con buen ticket"
        elif r_norm > 0.6 and f_norm < 0.4:
            nombre = "🔴 Clientes Dormidos"
            descripcion = "No compran hace tiempo, bajo gasto"
        elif r_norm > 0.4 and f_norm > 0.4:
            nombre = "🟡 En Riesgo"
            descripcion = "Compraban antes, ahora menos frecuente"
        elif f_norm < 0.3:
            nombre = "🆕 Clientes Nuevos"
            descripcion = "Pocas compras, potencial de crecimiento"
        else:
            nombre = f"📊 Cluster {cluster_id}"
            descripcion = "Comportamiento mixto"
        
        nombres_clusters[cluster_id] = {
            "nombre": nombre,
            "descripcion": descripcion
        }
    
    # Agregar nombres al DataFrame
    df["Cluster_Nombre"] = df["Cluster"].map(lambda x: nombres_clusters[x]["nombre"])
    df["Cluster_Descripcion"] = df["Cluster"].map(lambda x: nombres_clusters[x]["descripcion"])
    
    return df


# =====================================================
# 5. VISUALIZACIONES
# =====================================================
def plot_elbow_silhouette(metricas: dict) -> plt.Figure:
    """
    Gráficos de Elbow y Silhouette para determinar K óptimo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Method
    axes[0].plot(metricas["k_values"], metricas["inertias"], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel("Número de Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Inercia (WCSS)", fontsize=11)
    axes[0].set_title("📉 Método del Codo (Elbow)", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(metricas["k_values"], metricas["silhouette_scores"], marker='o', 
                 linewidth=2, markersize=8, color="coral")
    axes[1].set_xlabel("Número de Clusters (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].set_title("📊 Silhouette Score", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Umbral aceptable (0.5)')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def visualizar_clusters_2d(df_rfm: pd.DataFrame) -> plt.Figure:
    """
    Visualización 2D de clusters usando PCA.
    """
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(df_rfm[["Recency", "Frequency", "Monetary"]])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter por cluster
    clusters_unicos = df_rfm["Cluster"].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters_unicos)))
    
    for cluster, color in zip(sorted(clusters_unicos), colors):
        mask = df_rfm["Cluster"] == cluster
        nombre = df_rfm[mask]["Cluster_Nombre"].iloc[0]
        
        ax.scatter(
            rfm_pca[mask, 0],
            rfm_pca[mask, 1],
            c=[color],
            label=nombre,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} varianza)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} varianza)", fontsize=11)
    ax.set_title("🎯 Segmentación de Clientes (Vista 2D)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualizar_caracteristicas_clusters(df_rfm: pd.DataFrame) -> plt.Figure:
    """
    Gráfico de barras con características promedio por cluster.
    """
    # Calcular promedios por cluster
    cluster_stats = df_rfm.groupby("Cluster_Nombre").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).reset_index()
    
    # Normalizar para visualización
    for col in ["Recency", "Frequency", "Monetary"]:
        cluster_stats[f"{col}_norm"] = (
            cluster_stats[col] / cluster_stats[col].max()
        )
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Recency (invertido: menor es mejor)
    axes[0].barh(cluster_stats["Cluster_Nombre"], cluster_stats["Recency"], color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Días desde última compra", fontsize=11)
    axes[0].set_title("📅 Recency (menor = mejor)", fontsize=12, fontweight="bold")
    axes[0].invert_yaxis()
    
    # Frequency
    axes[1].barh(cluster_stats["Cluster_Nombre"], cluster_stats["Frequency"], color="coral", alpha=0.7)
    axes[1].set_xlabel("Número de compras", fontsize=11)
    axes[1].set_title("🔄 Frequency (mayor = mejor)", fontsize=12, fontweight="bold")
    axes[1].invert_yaxis()
    
    # Monetary
    axes[2].barh(cluster_stats["Cluster_Nombre"], cluster_stats["Monetary"], color="gold", alpha=0.7)
    axes[2].set_xlabel("Total gastado ($)", fontsize=11)
    axes[2].set_title("💰 Monetary (mayor = mejor)", fontsize=12, fontweight="bold")
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_distribucion_clusters(df_rfm: pd.DataFrame) -> plt.Figure:
    """
    Pie chart con distribución de clientes por cluster.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cluster_counts = df_rfm["Cluster_Nombre"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
    
    wedges, texts, autotexts = ax.pie(
        cluster_counts.values,
        labels=cluster_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    
    # Mejorar formato
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("📊 Distribución de Clientes por Segmento", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    return fig


# =====================================================
# 6. INTERPRETACIÓN EN TEXTO
# =====================================================
def interpretar_clusters_texto(df_rfm: pd.DataFrame) -> str:
    """
    Genera una interpretación en texto de los clusters.
    """
    texto = "### 📊 Interpretación de la Segmentación RFM\n\n"
    
    texto += f"**Total de clientes analizados:** {len(df_rfm)}\n\n"
    texto += f"**Número de segmentos identificados:** {df_rfm['Cluster'].nunique()}\n\n"
    
    texto += "#### 🎯 Características de cada Segmento\n\n"
    
    # Estadísticas por cluster
    for cluster_nombre in sorted(df_rfm["Cluster_Nombre"].unique()):
        df_cluster = df_rfm[df_rfm["Cluster_Nombre"] == cluster_nombre]
        
        texto += f"**{cluster_nombre}**\n\n"
        texto += f"*{df_cluster['Cluster_Descripcion'].iloc[0]}*\n\n"
        
        texto += f"- **Clientes:** {len(df_cluster)} ({len(df_cluster)/len(df_rfm)*100:.1f}% del total)\n"
        texto += f"- **Recency promedio:** {df_cluster['Recency'].mean():.0f} días\n"
        texto += f"- **Frequency promedio:** {df_cluster['Frequency'].mean():.1f} compras\n"
        texto += f"- **Monetary promedio:** ${df_cluster['Monetary'].mean():,.2f}\n"
        texto += f"- **Gasto total del segmento:** ${df_cluster['Monetary'].sum():,.2f}\n\n"
    
    texto += "#### 💡 Estrategias Recomendadas por Segmento\n\n"
    
    estrategias = {
        "Champions": "🌟 Mantener satisfechos con programa VIP, early access a productos, atención premium",
        "Leales": "💚 Programa de fidelización, rewards por compras, cross-selling inteligente",
        "Dormidos": "🔴 Campaña de reactivación, descuentos especiales, recordar beneficios",
        "En Riesgo": "🟡 Encuesta de satisfacción, ofertas personalizadas, mejorar experiencia",
        "Nuevos": "🆕 Welcome campaign, educación sobre productos, incentivar segunda compra"
    }
    
    for palabra_clave, estrategia in estrategias.items():
        clusters_con_palabra = [c for c in df_rfm["Cluster_Nombre"].unique() if palabra_clave in c]
        if clusters_con_palabra:
            texto += f"**{clusters_con_palabra[0]}**\n"
            texto += f"- {estrategia}\n\n"
    
    texto += "#### 📈 Insights Clave\n\n"
    
    # Cluster más valioso
    cluster_mas_valioso = df_rfm.groupby("Cluster_Nombre")["Monetary"].sum().idxmax()
    valor_cluster = df_rfm[df_rfm["Cluster_Nombre"] == cluster_mas_valioso]["Monetary"].sum()
    
    texto += f"- **Segmento más valioso:** {cluster_mas_valioso} (${valor_cluster:,.2f} en ventas)\n"
    
    # Cluster más grande
    cluster_mas_grande = df_rfm["Cluster_Nombre"].value_counts().idxmax()
    cant_cluster = df_rfm["Cluster_Nombre"].value_counts().max()
    
    texto += f"- **Segmento más grande:** {cluster_mas_grande} ({cant_cluster} clientes)\n"
    
    # Ticket promedio más alto
    ticket_promedio_por_cluster = df_rfm.groupby("Cluster_Nombre")["Monetary"].mean()
    cluster_mayor_ticket = ticket_promedio_por_cluster.idxmax()
    
    texto += f"- **Mayor ticket promedio:** {cluster_mayor_ticket} (${ticket_promedio_por_cluster.max():,.2f})\n"
    
    return texto


# =====================================================
# 7. PIPELINE COMPLETO
# =====================================================
def run_pipeline(df: pd.DataFrame, n_clusters: int = 4):
    """
    Ejecuta el pipeline completo de clustering RFM.
    
    Returns:
        dict con todos los resultados
    """
    # 1. Preparar datos RFM
    df_rfm, df_rfm_scaled, scaler = preparar_datos_rfm(df)
    
    # 2. Determinar número óptimo de clusters
    metricas_k = determinar_numero_clusters(df_rfm_scaled)
    
    # 3. Crear y entrenar modelo
    model = crear_modelo_clustering(n_clusters=n_clusters)
    model, labels = entrenar_clustering(model, df_rfm_scaled)
    
    # 4. Asignar nombres a clusters
    df_rfm = asignar_nombres_clusters(df_rfm, labels)
    
    # 5. Calcular Silhouette Score
    silhouette = silhouette_score(df_rfm_scaled, labels)
    
    # 6. Visualizaciones
    fig_elbow = plot_elbow_silhouette(metricas_k)
    fig_2d = visualizar_clusters_2d(df_rfm)
    fig_caracteristicas = visualizar_caracteristicas_clusters(df_rfm)
    fig_distribucion = plot_distribucion_clusters(df_rfm)
    
    # 7. Interpretación
    texto_interpretacion = interpretar_clusters_texto(df_rfm)
    
    return {
        "df_rfm": df_rfm,
        "df_rfm_scaled": df_rfm_scaled,
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "silhouette_score": silhouette,
        "metricas_k": metricas_k,
        "fig_elbow": fig_elbow,
        "fig_2d": fig_2d,
        "fig_caracteristicas": fig_caracteristicas,
        "fig_distribucion": fig_distribucion,
        "texto_interpretacion": texto_interpretacion
    }