# 📘 Documentación del Proyecto  
## 🪐 **Aurelion - Sistema de Análisis de Ventas con Machine Learning**

---

## 🧩 Contexto General  

La tienda **Aurelion** busca mejorar su gestión comercial mediante una plataforma interactiva que le permita **analizar, visualizar, predecir y optimizar sus datos de ventas** en tiempo real.  
El proyecto nace de la necesidad de reemplazar las planillas manuales por una herramienta dinámica y automatizada que facilite la toma de decisiones estratégicas basadas en datos.

---

## ⚠️ Problema Detectado  

El manejo manual de los registros genera diversas dificultades:  
- Dificultad para detectar **productos más vendidos o con baja rotación**.  
- Falta de claridad en **tendencias de ventas** y comportamiento del cliente.  
- Imposibilidad de **visualizar métricas comerciales** de forma inmediata.  
- **Ausencia de capacidad predictiva** para planificar inventario y anticipar demanda.
- Falta de **segmentación de clientes** para estrategias de marketing personalizadas.

---

## 💡 Propuesta de Solución  

Se desarrolló una **aplicación interactiva** en **Python** utilizando **Streamlit**, que permite visualizar, analizar y **predecir** patrones en los datos de ventas de manera intuitiva, integrando componentes gráficos, métricas de negocio y **modelos de Machine Learning**.  

La solución evoluciona en **tres Sprints**:

### 🌀 SPRINT 1 – Base del Sistema  
- Limpieza y normalización del dataset (`datos_limpios.csv`).  
- Documentación del flujo de trabajo y objetivos del proyecto.  
- Implementación inicial del entorno Streamlit con estructura modular (`app.py`, `utils.py`, `graficos.py`).  
- Carga dinámica del dataset y visualización básica de los datos.  

### 🚀 SPRINT 2 – Análisis y Visualización Avanzada  
- Incorporación del **Análisis Exploratorio de Datos (EDA)**.  
- Creación de **visualizaciones generales interactivas**.  
- Integración de **métricas de negocio** para la toma de decisiones estratégicas.  
- Diseño visual mejorado con un estilo **profesional y uniforme** en toda la app.  

### 🤖 SPRINT 3 – Machine Learning e Inteligencia Predictiva
- Implementación de **3 modelos de Machine Learning** para predicción y clasificación.
- Desarrollo de módulos especializados (`regression.py`, `classification.py`, `clustering.py`).
- Integración de **análisis predictivo** para anticipar ventas futuras.
- Sistema de **segmentación automática de clientes** usando clustering.
- **Clasificación inteligente** del medio de pago preferido por cliente.

---

## 🎯 Objetivo General  

Brindar a **Aurelion** una herramienta integral que permita **comprender, predecir y optimizar sus ventas**, mediante una interfaz visual moderna, métricas automatizadas y **modelos de Machine Learning** que impulsen la **toma de decisiones basadas en datos e inteligencia artificial**.

---

## 🎯 Objetivos Específicos  

- Analizar patrones de venta por **categoría**, **cliente**, **ciudad** y **medio de pago**.  
- Identificar **tendencias temporales** en las ventas (diarias, mensuales y anuales).  
- Detectar **productos y clientes más rentables**.  
- **Predecir ventas futuras** por producto para optimizar inventario.
- **Clasificar y anticipar** el medio de pago que usará cada cliente.
- **Segmentar clientes** automáticamente según su valor y comportamiento (RFM).
- Evaluar la **efectividad comercial** mediante indicadores clave.  
- Mejorar la experiencia de uso mediante una interfaz clara y atractiva.  

---

## 🧠 Estructura de la Aplicación  

### 📄 **1. Documentación actualizada**  
Información general del proyecto, su evolución y objetivos.  

### 🧾 **2. Datos limpios (CSV)**  
Muestra el dataset depurado, con valores corregidos, tipos de datos consistentes y estructura apta para análisis.  

### 🔍 **3. Análisis Exploratorio (EDA)**  
Incluye funciones que analizan y describen el comportamiento de los datos:  
- Distribución de importes y precios por categoría.  
- Relación entre cantidad e importe.  
- Series temporales y evolución de ventas.  
- **Heatmap** de ventas por año y mes.  

Cada gráfico cuenta con una **leyenda interpretativa**, explicando su propósito y los hallazgos visuales.  

### 📈 **4. Visualizaciones Generales**  
Ofrece una vista panorámica de la actividad comercial a través de gráficos interactivos:  
- Productos y categorías más vendidos.  
- Ventas por ciudad, medio de pago y periodo.  
- Mapas y diagramas comparativos.  
- Promedio de importes por día y evolución temporal.  

### 💼 **5. Métricas de Negocio**  
Incorpora indicadores claves de desempeño (**KPI**) que aportan una visión estratégica:  
- Ticket promedio global y por categoría.  
- Margen estimado por tipo de producto.  
- Ranking de clientes por ingresos.  
- Frecuencia de compra y retención mensual.  
- Análisis de cohortes de clientes según fecha de alta.  

### 🤖 **6. SPRINT 3 - Machine Learning**

#### 📦 **6.1. Predicción de Ventas por Producto** (`regression.py`)

**Objetivo:** Predecir cuántas unidades se venderán de cada producto el próximo mes para optimizar la gestión de inventario.

**Algoritmo elegido:** Regresión Lineal con división por producto  
**Justificación:** 
- Simple e interpretable para stakeholders
- Funciona bien con datasets pequeños (35 observaciones producto-mes)
- Coeficientes claros que muestran qué factores impulsan las ventas

**Entradas (X):**
- `lag_1`: Ventas del mes anterior (feature más importante)
- `lag_2`: Ventas de hace 2 meses
- `lag_3`: Ventas de hace 3 meses  
- `precio_unit`: Precio unitario del producto
- `categoria_encoded`: Categoría del producto codificada

**Salida (y):** Cantidad de unidades vendidas del producto

**Métricas de evaluación:**
- **R² Score:** 0.338 (33.8% de varianza explicada)
- **MAE:** 1.1 unidades (error promedio)
- **RMSE:** 1.4 unidades
- **MAPE:** 70.9% (alto debido a valores pequeños)

**Modelo ML implementado:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**División train/test:**
- División por **producto completo** (no temporal)
- 80% entrenamiento, 20% test
- Estrategia para evitar leakage temporal en features de lag

**Predicciones y métricas calculadas:**
- Predicción de ventas para el próximo mes de cada producto
- Top N productos con mayores ventas proyectadas
- Ingresos estimados por producto
- Coeficientes del modelo (importancia de cada variable)

**Resultados en gráficos:**
- Gráfico de análisis exploratorio (top productos, ventas por categoría)
- Real vs Predicho (scatter plot)
- Distribución de errores
- Top 30 productos con mayor error
- Distribución Real vs Predicha (histograma + KDE)

**Interpretación:**
- El modelo logra predecir razonablemente bien con datos limitados
- `lag_1` (ventas del mes anterior) es el predictor más importante
- Útil para planificación de compras y gestión de stock

---

#### 💳 **6.2. Clasificación de Medio de Pago** (`classification.py`)

**Objetivo:** Predecir qué medio de pago usará un cliente según características de la transacción para personalizar la experiencia de pago.

**Algoritmo elegido:** Random Forest Classifier  
**Justificación:**
- Robusto ante datos ruidosos y desbalanceados
- No requiere normalización de features
- Proporciona importancia de variables
- Maneja bien interacciones no lineales entre features

**Entradas (X):**
- `importe`: Monto total de la compra
- `cantidad`: Cantidad de productos
- `categoria_encoded`: Categoría del producto
- `ciudad_encoded`: Ciudad del cliente
- `mes`: Mes del año (1-12)
- `dia_semana`: Día de la semana (0-6)
- `es_fin_semana`: Binario (0/1)
- `rango_precio_encoded`: Rango de precio (bajo/medio/alto)
- `compras_previas_cliente`: Historial de compras
- `gasto_promedio_cliente`: Gasto promedio histórico

**Salida (y):** Medio de pago (efectivo, tarjeta, qr, transferencia)

**Métricas de evaluación:**
- **Accuracy:** 44.9% (mejor que azar del 25%)
- **Precision:** 44.4% (ponderada)
- **Recall:** 44.9% (ponderada)
- **F1-Score:** 44.3% (ponderada)

**Modelo ML implementado:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
```

**División train/test:**
- 80% entrenamiento, 20% test
- **Estratificación** para mantener proporción de clases
- Shuffle activado para evitar sesgos temporales

**Predicciones y métricas calculadas:**
- Predicción de medio de pago para nuevas transacciones
- Probabilidades por cada clase
- Matriz de confusión normalizada
- Reporte de clasificación por clase (precision, recall, F1)
- Análisis de errores (confusiones más frecuentes)

**Resultados en gráficos:**
- Distribución de medios de pago
- Medio de pago por categoría (heatmap)
- Importe promedio por medio de pago
- Uso por día de la semana
- Matriz de confusión
- Importancia de features
- Distribución Real vs Predicha

**Funcionalidades interactivas:**
- **Prueba con nuevas transacciones:** Permite simular una compra y ver predicción en tiempo real
- **Análisis de errores:** Muestra casos mal clasificados y patrones de confusión

**Interpretación:**
- Accuracy moderado debido a dataset pequeño (69 transacciones en test)
- Las 4 clases tienen comportamientos similares (dificulta clasificación)
- Útil para personalizar opciones de pago mostradas al cliente

---

#### 👥 **6.3. Segmentación de Clientes (RFM Clustering)** (`clustering.py`)

**Objetivo:** Agrupar clientes según su comportamiento de compra para estrategias de marketing personalizadas.

**Algoritmo elegido:** K-Means Clustering  
**Justificación:**
- Algoritmo clásico y probado para segmentación RFM
- Rápido y eficiente
- Fácil de interpretar (centroides = perfil promedio del segmento)
- Escalable a más clientes

**Entradas (X):**
- **R (Recency):** Días desde la última compra (menor = mejor)
- **F (Frequency):** Cantidad de compras realizadas (mayor = mejor)
- **M (Monetary):** Total gastado (mayor = mejor)

**Salida (y):** No supervisado - el modelo descubre grupos automáticamente

**Métricas de evaluación:**
- **Silhouette Score:** Mide qué tan bien definidos están los clusters
  - > 0.7: Excelente separación
  - 0.5 - 0.7: Buena separación
  - < 0.5: Clusters se solapan
- **Método del Codo (Elbow):** Determina K óptimo buscando punto de inflexión
- **Inercia (WCSS):** Suma de distancias al cuadrado dentro de clusters

**Modelo ML implementado:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Normalizar RFM (crítico para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Entrenar K-Means
model = KMeans(n_clusters=4, random_state=42)
labels = model.fit_predict(X_scaled)
```

**División train/test:**
- No aplica (aprendizaje no supervisado)
- Se usa todo el dataset para descubrir patrones
- Validación mediante Silhouette Score

**Segmentos identificados automáticamente:**
- 🌟 **Champions (VIP):** Compran frecuente, reciente y gastan mucho
- 💚 **Clientes Leales:** Compran regularmente con buen ticket
- 🟡 **En Riesgo:** Compraban antes, ahora menos frecuente
- 🔴 **Clientes Dormidos:** No compran hace tiempo, bajo gasto
- 🆕 **Clientes Nuevos:** Pocas compras, potencial de crecimiento

**Predicciones y métricas calculadas:**
- Asignación de cada cliente a un segmento
- Características promedio por segmento (R, F, M)
- Distribución porcentual de clientes por segmento
- Valor total generado por segmento
- Ranking de segmentos por rentabilidad

**Resultados en gráficos:**
- Método del Codo + Silhouette Score (determinar K óptimo)
- Visualización 2D con PCA (reduce 3D → 2D)
- Características promedio por segmento (barras horizontales)
- Distribución de clientes por segmento (pie chart)

**Estrategias recomendadas por segmento:**
- **Champions:** Programa VIP, early access, atención premium
- **Leales:** Programa de fidelización, rewards, cross-selling
- **Dormidos:** Campaña de reactivación, descuentos especiales
- **En Riesgo:** Encuesta de satisfacción, ofertas personalizadas
- **Nuevos:** Welcome campaign, incentivar segunda compra

**Interpretación:**
- Segmentación automática basada en comportamiento real
- Útil para campañas de marketing dirigidas
- Permite asignar recursos según valor del cliente

---

### 🧠 **7. Resultados y Conclusiones**  
Síntesis de los principales hallazgos del análisis y posibles líneas de acción para optimizar las ventas, incluyendo insights de los modelos de Machine Learning.

---

## 🛠️ Tecnologías Utilizadas  

| Tipo | Herramienta |
|------|--------------|
| **Lenguaje principal** | Python 🐍 |
| **Framework web** | Streamlit 📊 |
| **Análisis y manipulación** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn |
| **Modelos implementados** | Linear Regression, Random Forest, K-Means |
| **Formato de datos** | CSV |
| **Diseño visual** | Streamlit Theme + Estilo profesional personalizado |

---

## 📊 Resultados Obtenidos  

### Sprint 1 y 2:
- Se logró una **visualización integral y automatizada** de los datos de ventas.  
- Se identificaron **productos clave y categorías con mayor rentabilidad**.  
- Se detectaron **picos de demanda por mes y medio de pago preferido**.  
- Se generó una interfaz amigable que facilita la **toma de decisiones comerciales**.  

### Sprint 3 (Machine Learning):
- **Modelo de Regresión:** Predice ventas futuras con R² = 0.338, útil para planificación de inventario a pesar del dataset limitado.
- **Modelo de Clasificación:** Accuracy del 44.9%, mejor que azar (25%), permite personalizar experiencia de pago.
- **Modelo de Clustering:** Segmenta clientes automáticamente en 4-5 grupos con características diferenciadas, habilitando marketing personalizado.
- Se integró capacidad **predictiva e inteligente** a la plataforma.
- Se habilitó **segmentación automática** para estrategias de retención.

---

## 🚀 Próximos Pasos Sugeridos

1. **Recopilación de más datos:** Ampliar el dataset a 12+ meses para mejorar precisión de modelos.
2. **Features adicionales:** Incorporar hora del día, dispositivo usado, promociones activas.
3. **Modelos más avanzados:** Probar XGBoost, LightGBM o redes neuronales con más datos.
4. **Automatización:** Reentrenar modelos periódicamente con datos nuevos.
5. **Integración con sistemas:** Conectar predicciones con sistema de inventario y CRM.

---

## 📖 Estructura del Código

```
proyecto_aurelion/
│
├── app.py                      # Aplicación principal Streamlit
├── documentacion.md            # Este archivo
│
├── Base de datos/
│   └── ventas_completas.csv    # Dataset limpio
│
├── scripts/
│   ├── utils.py                # Funciones auxiliares
│   ├── graficos.py             # Generación de gráficos
│   └── metricas.py             # Cálculo de KPIs
│
└── scriptsML/                  # SPRINT 3
    ├── regression.py           # Modelo de predicción de ventas
    ├── classification.py       # Modelo de clasificación de pago
    └── clustering.py           # Modelo de segmentación RFM
```

---

## 👨‍💻 Autores  
**Equipo de Desarrollo - Proyecto Aurelion**  
📅 Versión: Sprint 3 (Machine Learning)  
💻 Desarrollado con Python, Streamlit y Scikit-learn  
🤖 Modelos: Regresión Lineal, Random Forest, K-Means Clustering

---

## 📄 Licencia
Proyecto educativo desarrollado como parte del curso de Machine Learning - IBM.