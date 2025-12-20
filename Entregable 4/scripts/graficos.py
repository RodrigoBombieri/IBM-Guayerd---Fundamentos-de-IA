# =============================================================
# GRAFICOS.PY - Proyecto Aurelion
# =============================================================
# Autor: Rodrigo Sebastián Bombieri
# Descripción: Generación de gráficos y análisis visuales
# para el Proyecto Aurelion - IBM SkillsBuild 2025
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import chi2_contingency
import folium

# -------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# -------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# -------------------------------------------------------------
# FUNCIÓN DE CARGA DE DATOS
# -------------------------------------------------------------
def load_data(path="./Base de datos/ventas_completas.csv"):
    """Carga el dataset principal limpio."""
    try:
        df = pd.read_csv(path)
        st.success("✅ Datos cargados correctamente.")
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        return None

# ------------------------------------------------------------
# ESTADÍSTICAS (EDA)
# ------------------------------------------------------------

def analisis_estadistico(df):
    """Análisis estadístico detallado del dataset."""

    st.subheader("📊 Análisis Estadístico General")
    # --- Estadísticas descriptivas básicas ---
    st.markdown("#### 📋 Estadísticas Descriptivas Generales")
    st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"))

    # --- Medidas adicionales ---
    st.markdown("#### 📈 Medidas de Tendencia Central y Dispersión")
    columnas_numericas = ["importe", "cantidad"]

    for col in columnas_numericas:
        st.markdown(f"**📦 Variable:** `{col}`")
        st.write(f"- Media: {df[col].mean():.2f}")
        st.write(f"- Mediana: {df[col].median():.2f}")
        st.write(f"- Moda: {df[col].mode()[0]:.2f}")
        st.write(f"- Mínimo: {df[col].min():.2f}")
        st.write(f"- Máximo: {df[col].max():.2f}")
        st.write(f"- Rango: {(df[col].max() - df[col].min()):.2f}")
        st.write(f"- Desviación estándar: {df[col].std():.2f}")
        st.write(f"- Varianza: {df[col].var():.2f}")
        st.write(f"- Asimetría: {df[col].skew():.2f}")
        st.write(f"- Curtosis: {df[col].kurtosis():.2f}")
        st.markdown("---")

    # --- Percentiles ---
    st.markdown("#### 📊 Percentiles (Importe)")
    percentiles = df["importe"].quantile([0.25, 0.5, 0.75]).to_dict()
    st.json({f"{int(k*100)}%": round(v, 2) for k, v in percentiles.items()})

    # --- Detección de valores atípicos ---
    st.markdown("#### ⚠️ Detección de Valores Atípicos (Importe)")
    q1 = df["importe"].quantile(0.25)
    q3 = df["importe"].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    outliers = df[(df["importe"] < limite_inf) | (df["importe"] > limite_sup)]
    st.write(f"- Límite inferior: {limite_inf:.2f}")
    st.write(f"- Límite superior: {limite_sup:.2f}")
    st.write(f"- Total de valores atípicos detectados: {len(outliers)}")

    if len(outliers) > 0:
        st.dataframe(outliers[["id_venta", "importe", "categoria", "ciudad"]].head(10))
    else:
        st.success("✅ No se detectaron valores atípicos significativos.")

    # --- Correlaciones ---
    st.markdown("#### 🔗 Correlación entre variables numéricas")
    corr = df[["importe", "cantidad"]].corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None))

    # --- Visual complementario ---
    st.markdown("#### 📉 Distribución del Importe con Regla de 68-95-99.7")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["importe"], kde=True, bins=30, ax=ax)
    mean = df["importe"].mean()
    std = df["importe"].std()
    for i, pct in enumerate([1, 2, 3], 1):
        ax.axvline(mean + i*std, color="red", linestyle="--", alpha=0.5)
        ax.axvline(mean - i*std, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Distribución del Importe con Desviaciones Estándar")
    st.pyplot(fig)

# ============================================================
# 1️⃣ Distribución de Importe (Histograma + KDE)
# ============================================================
def plot_importe_distribution(df):
    data = pd.to_numeric(df['importe'], errors='coerce').dropna()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(data, bins=30, kde=True, color='#4C78A8', ax=ax)
    ax.set_title('💵 Distribución de Importes por Ticket', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importe ($)')
    ax.set_ylabel('Frecuencia')
    plt.grid(alpha=0.2)
    fig.tight_layout()
    return fig

# ============================================================
# 2️⃣ Boxplot de Importe por Categoría
# ============================================================
def plot_box_importe_by_categoria(df):
    tmp = df[['categoria','importe']].dropna()
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=tmp, x='categoria', y='importe', palette='YlOrRd', ax=ax)
    ax.set_title('📦 Importe por Categoría', fontsize=13, fontweight='bold')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Importe ($)')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    return fig

# ============================================================
# 3️⃣ Boxplot de Precio Unitario por Categoría
# ============================================================
def plot_box_precio_by_categoria(df):
    tmp = df[['categoria','precio_unitario']].dropna()
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(data=tmp, x='categoria', y='precio_unitario', palette='YlGnBu', ax=ax)
    ax.set_title('🏷️ Precio Unitario por Categoría', fontsize=13, fontweight='bold')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Precio Unitario ($)')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    return fig

# ============================================================
# 4️⃣ Dispersión Cantidad vs Importe (por Categoría)
# ============================================================
def plot_scatter_cantidad_importe(df):
    tmp = df[['cantidad','importe','categoria']].dropna()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=tmp, x='cantidad', y='importe', hue='categoria', alpha=0.7, ax=ax)
    ax.set_title('📊 Relación entre Cantidad e Importe', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cantidad')
    ax.set_ylabel('Importe ($)')
    ax.legend(title='Categoría', loc='best')
    plt.grid(alpha=0.2)
    fig.tight_layout()
    return fig

# ============================================================
# 5️⃣ Serie temporal de ventas
# ============================================================
def plot_time_series_importe(df, freq='D'):
    tmp = df.copy()
    tmp['fecha'] = pd.to_datetime(tmp['fecha'], errors='coerce')
    tmp = tmp.dropna(subset=['fecha','importe'])
    ts = tmp.set_index('fecha').sort_index().resample(freq)['importe'].sum()

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(ts.index, ts.values, color='#72B7B2', linewidth=2)
    ax.set_title(f'📈 Evolución de Ventas ({freq})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Importe Total ($)')
    plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# ============================================================
# 6️⃣ Heatmap de Ventas por Año y Mes
# ============================================================
def viz_heatmap_ventas_calendario(df):
    d = _prep(df)
    d = d.dropna(subset=['fecha'])
    d['anio'] = d['fecha'].dt.year
    d['mes'] = d['fecha'].dt.month
    piv = d.pivot_table(index='anio', columns='mes', values='importe', aggfunc='sum').fillna(0)

    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(piv, cmap='YlOrRd', annot=True, fmt=".0f", linewidths=0.5,
                cbar_kws={'label': 'Importe total ($)'}, ax=ax)
    ax.set_title('🗓️ Ventas por Año y Mes', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Año')
    plt.xticks(ticks=np.arange(6), labels=['Ene','Feb','Mar','Abr','May','Jun'])
    plt.tight_layout()
    return fig

# Carga de datos y definición de 10 funciones: 5 de visualización y 5 de negocio
# - Usamos el dataframe local ventas_completas.csv ya presente en el entorno.
# - Creamos funciones modulares que dibujan con matplotlib/seaborn y devuelven datos clave para integraciones.

# Utilidad: preparar fechas y numéricos

def _prep(df):
    out = df.copy()
    out['fecha'] = pd.to_datetime(out['fecha'], errors='coerce')
    for c in ['importe','precio_unitario','cantidad']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

# ------------------------------------------------------------
# VISUALIZACIONES
# ------------------------------------------------------------
def viz_importe_hist_kde(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["importe"], kde=True, ax=ax)
    ax.set_title("Distribución del Importe")
    ax.set_xlabel("Importe ($)")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig

def viz_top_productos_barras(df, n=10):
    tmp = df.groupby('nombre_producto', as_index=False)['importe'].sum().sort_values('importe', ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10,5))
    g = sns.barplot(data=tmp, x='importe', y='nombre_producto', palette='Blues_r', ax=ax)
    ax.set_title(f'🏆 Top {n} Productos por Ingresos', fontsize=13, fontweight='bold')
    ax.set_xlabel('Ingresos ($)')
    ax.set_ylabel('Producto')
    ax.bar_label(g.containers[0], fmt='%.0f', padding=3)
    plt.tight_layout()
    return fig

def viz_categorias_pastel(df):
    tmp = df.groupby('categoria', as_index=False)['importe'].sum().sort_values('importe', ascending=False)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(tmp['importe'], labels=tmp['categoria'], autopct='%1.0f%%', startangle=140, colors=sns.color_palette('Set2'))
    ax.set_title('🧁 Participación de Ingresos por Categoría', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

def grafico_categoria_mas_vendida_por_mes(df):
    categoria_mes = df.groupby(["mes", "categoria"])["importe"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(9,5))
    g = sns.barplot(x="mes", y="importe", hue="categoria", data=categoria_mes, palette="YlOrRd", ax=ax)
    ax.set_title("📅 Categoría más Vendida por Mes", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mes")
    ax.set_ylabel("Importe Total ($)")
    plt.legend(title='Categoría', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def viz_tiempo_linea(df, freq='D'):
    tmp = df.copy()
    tmp['fecha'] = pd.to_datetime(tmp['fecha'], errors='coerce')
    tmp = tmp.dropna(subset=['fecha','importe'])
    ts = tmp.set_index('fecha').sort_index().resample(freq)['importe'].sum()

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(ts.index, ts.values, color='#72B7B2')
    ax.set_title(f'Ventas en el Tiempo ({freq})')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Importe Total')
    fig.tight_layout()

    return fig  # ✅ Devolver la figura, no la serie

def grafico_importe_promedio_por_dia(df):
    # Convertir la columna de fecha y extraer el día del mes
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_del_mes'] = df['fecha'].dt.day
    # 1. Calcular el importe total por cada venta ÚNICA (SUMA de todas las líneas de la misma id_venta)
    importe_por_venta_unica = df.groupby(['id_venta', 'dia_del_mes'])['importe'].sum().reset_index()
    # 2. Calcular el IMPORTE PROMEDIO por DÍA del mes
    importe_promedio_dia = importe_por_venta_unica.groupby('dia_del_mes')['importe'].mean().reset_index(name='importe_promedio')
    # =================================================================
    # 2. GRÁFICO DE DISPERSIÓN Y ANÁLISIS
    # =================================================================
    sns.set_style("whitegrid")
    # Renombrar columnas para el gráfico
    data_plot = importe_promedio_dia.rename(columns={'dia_del_mes': 'Día del Mes', 'importe_promedio': 'Importe Promedio de Venta'})
    # Calcular la correlación de Pearson
    correlacion = data_plot['Día del Mes'].corr(data_plot['Importe Promedio de Venta'])
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    # Gráfico de dispersión
    sns.scatterplot(
        x='Día del Mes',
        y='Importe Promedio de Venta',
        data=data_plot,
        ax=ax,
        color='darkgreen',
        alpha=0.8,
        edgecolor='black',
        s=100
    )

    # Añadir línea de regresión lineal
    z = np.polyfit(data_plot['Día del Mes'], data_plot['Importe Promedio de Venta'], 1)
    p = np.poly1d(z)
    ax.plot(data_plot['Día del Mes'], p(data_plot['Día del Mes']),
            color='red', linestyle='--', linewidth=3,
            label=f'Línea de Tendencia (Pendiente: {z[0]:.2f})')

    # Línea horizontal con el promedio
    promedio_general = importe_promedio_dia['importe_promedio'].mean()
    ax.axhline(promedio_general, color='blue', linestyle=':', linewidth=1.5, label=f'Promedio General: {promedio_general:.0f}')
    ax.legend(fontsize=10, loc='upper left')

    ax.set_title('Correlación: Importe Promedio de Venta Única vs. Día del Mes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Día del Mes', fontsize=12)
    ax.set_ylabel('Importe Promedio de Venta Única', fontsize=12)
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(axis='both', alpha=0.3)

    plt.tight_layout()
    return fig

def grafico_mapa_ventas(df):
    """Mapa interactivo de ventas por ciudad."""
    # Paso 1: Calcular ventas únicas por ciudad
    ventas_por_ciudad = df.groupby('ciudad')['id_venta'].nunique().reset_index()
    ventas_por_ciudad.rename(columns={'id_venta': 'numero_ventas'}, inplace=True)

    # Paso 2: Asignar coordenadas geográficas simuladas a cada ciudad
    # NOTA: Estas son coordenadas simuladas. Para un análisis real, deberías usar
    # una fuente de datos geográfica (ej. Geopy, Nominatim, un archivo shapefile, etc.)
    # Aquí asumimos algunas ubicaciones representativas para las ciudades de ejemplo.
    coordenadas_ciudades = {
        'Rio Cuarto': [-33.1227, -64.3248],
        'Alta Gracia': [-31.6521, -64.4273],
        'Cordoba': [-31.4201, -64.1888],
        'Carlos Paz': [-31.4234, -64.5043],
        'Villa Maria': [-32.4073, -63.2433],
        'Mendiolaza': [-31.2667, -64.3167],
        'Desconocido': [-34.6037, -58.3816] # Coordenada genérica para desconocidos
    }

    ventas_por_ciudad['latitud'] = ventas_por_ciudad['ciudad'].map(lambda x: coordenadas_ciudades.get(x, [0, 0])[0])
    ventas_por_ciudad['longitud'] = ventas_por_ciudad['ciudad'].map(lambda x: coordenadas_ciudades.get(x, [0, 0])[1])

    # Eliminar ciudades sin coordenadas simuladas si las hay (aunque con .get(x, [0,0]) no debería haber)
    ventas_por_ciudad = ventas_por_ciudad[(ventas_por_ciudad['latitud'] != 0) | (ventas_por_ciudad['longitud'] != 0)].copy()

    # Paso 3: Crear el mapa base
    # El centro del mapa puede ser una ciudad central o el promedio de las coordenadas
    centro_mapa = [-31.4201, -64.1888] # Centro aproximado de la provincia de Córdoba
    mapa_ventas = folium.Map(location=centro_mapa, zoom_start=9)

    # Paso 4: Añadir marcadores o círculos al mapa
    for index, row in ventas_por_ciudad.iterrows():
        ciudad = row['ciudad']
        ventas = row['numero_ventas']
        lat = row['latitud']
        lon = row['longitud']
    # Usar círculos con radio proporcional al número de ventas
        folium.CircleMarker(
            location=[lat, lon],
            radius=ventas * 0.5,  # Ajusta este factor para el tamaño del círculo
            popup=f"Ciudad: {ciudad}<br>Ventas: {ventas}",
            color='blue',
            fill=True,
            fill_color='steelblue',
            fill_opacity=0.6
        ).add_to(mapa_ventas)
    components.html(mapa_ventas._repr_html_(), height=500)

def grafico_medio_pago(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    if "medio_pago" in df.columns:
        medio_pago = df["medio_pago"].value_counts().reset_index()
        medio_pago.columns = ["medio_pago", "cantidad"]
        sns.barplot(x="medio_pago", y="cantidad", data=medio_pago, palette="viridis", ax=ax)
        ax.set_title("Medios de pago más usados")
        ax.set_xlabel("Medio de pago")
        ax.set_ylabel("Cantidad de ventas")
    else:
        ax.text(0.5, 0.5, "No hay columna 'medio_pago'", ha="center", va="center")
        ax.axis("off")
    return fig

def grafico_ventas_por_ciudad(df):
    """Ventas totales por ciudad."""
    ventas_ciudad = df.groupby("ciudad")["importe"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="importe", y="ciudad", data=ventas_ciudad, ax=ax)
    ax.set_title("Ventas Totales por Ciudad")
    ax.set_xlabel("Importe Total ($)")
    ax.set_ylabel("Ciudad")
    return fig


# ------------------------------------------------------------
# MÉTRICAS DE NEGOCIO
# ------------------------------------------------------------
# =========================================================
# 1️⃣ Ticket promedio por categoría
# =========================================================
def biz_ticket_promedio_vs_categoria(df):
    d = _prep(df)
    ticket_total = d.groupby(['id_venta', 'categoria'], as_index=False)['importe'].sum()
    kpi = ticket_total.groupby('categoria', as_index=False)['importe'].mean().rename(columns={'importe': 'ticket_promedio'})

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=kpi, x='categoria', y='ticket_promedio', palette='Set2', ax=ax)

    # Etiquetas numéricas arriba de cada barra
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), f"${p.get_height():,.0f}",
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    ax.set_title('💳 Ticket Promedio por Categoría', fontsize=14, fontweight='bold')
    ax.set_xlabel('Categoría', fontsize=12)
    ax.set_ylabel('Ticket Promedio ($)', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================
# 2️⃣ Ticket promedio por medio de pago
# =========================================================
def grafico_ticket_promedio_por_medio(df):
    ticket_medio = df.groupby("medio_pago")["importe"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x="medio_pago", y="importe", data=ticket_medio, ax=ax, palette="Set3")

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), f"${p.get_height():,.0f}",
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    ax.set_title("💰 Ticket Promedio por Medio de Pago", fontsize=14, fontweight='bold')
    ax.set_xlabel("Medio de Pago", fontsize=12)
    ax.set_ylabel("Importe Promedio ($)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================
# 3️⃣ Top clientes por ingresos
# =========================================================
def biz_top_clientes_ingresos(df, n=10):
    topc = df.groupby(['id_cliente', 'nombre_cliente'], as_index=False)['importe'].sum().sort_values('importe', ascending=False).head(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=topc, x='importe', y='nombre_cliente', palette='magma', ax=ax)

    for p in ax.patches:
        ax.text(p.get_width() + (p.get_width()*0.01), p.get_y() + p.get_height()/2,
                f"${p.get_width():,.0f}", va='center', fontsize=9, color='black')

    ax.set_title(f'👥 Top {n} Clientes por Ingresos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ingresos Totales ($)', fontsize=12)
    ax.set_ylabel('Cliente', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================
# 4️⃣ Frecuencia de compra por cliente
# =========================================================
def biz_frecuencia_compra_por_cliente(df):
    d = _prep(df)
    freq = d.groupby('id_cliente')['id_venta'].nunique().reset_index(name='compras')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(freq['compras'], bins=20, color='#F58518', ax=ax, kde=True)

    ax.set_title('🛍️ Frecuencia de Compra por Cliente', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Compras por Cliente', fontsize=12)
    ax.set_ylabel('Cantidad de Clientes', fontsize=12)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================
# 5️⃣ Margen aproximado por categoría
# =========================================================
def biz_margen_aprox_por_categoria(df, margen_unitario_pct=0.25):
    d = _prep(df)
    d['margen_aprox'] = d['precio_unitario'] * d['cantidad'] * float(margen_unitario_pct)
    cat = d.groupby('categoria', as_index=False)['margen_aprox'].sum().sort_values('margen_aprox', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=cat, x='margen_aprox', y='categoria', palette='Greens', ax=ax)

    for p in ax.patches:
        ax.text(p.get_width() + (p.get_width()*0.01), p.get_y() + p.get_height()/2,
                f"${p.get_width():,.0f}", va='center', fontsize=9, color='black')

    ax.set_title(f"📊 Margen Aproximado por Categoría (supuesto {int(margen_unitario_pct*100)}%)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Margen Aproximado ($)', fontsize=12)
    ax.set_ylabel('Categoría', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================
# 6️⃣ Cohorte mensual de retención
# =========================================================
def biz_cohorte_mes_alta_retencion(df):
    d = _prep(df)
    d['fecha_alta'] = pd.to_datetime(d['fecha_alta'], errors='coerce')
    ventas = d.dropna(subset=['fecha', 'id_cliente', 'fecha_alta'])
    ventas['cohorte'] = ventas['fecha_alta'].dt.to_period('M').astype(str)
    ventas['periodo'] = ventas['fecha'].dt.to_period('M').astype(str)

    base = ventas.groupby(['cohorte'])['id_cliente'].nunique().rename('clientes_cohorte')
    tabla = ventas.groupby(['cohorte', 'periodo'])['id_cliente'].nunique().reset_index()
    tabla = tabla.merge(base, on='cohorte', how='left')
    tabla['retencion'] = tabla['id_cliente'] / tabla['clientes_cohorte']
    piv = tabla.pivot(index='cohorte', columns='periodo', values='retencion').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(piv, cmap='YlOrRd', vmin=0, vmax=1, annot=True, fmt=".0%", cbar_kws={'label': 'Tasa de Retención'}, ax=ax)

    ax.set_title('📅 Retención de Clientes por Cohorte de Alta', fontsize=14, fontweight='bold')
    ax.set_xlabel('Periodo de Compra', fontsize=12)
    ax.set_ylabel('Cohorte (Mes de Alta)', fontsize=12)
    plt.tight_layout()
    return fig

