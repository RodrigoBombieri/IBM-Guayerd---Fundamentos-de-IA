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
- **Necesidad de dashboards ejecutivos** para presentaciones y seguimiento de KPIs.
---

## 💡 Propuesta de Solución  

Se desarrolló una **aplicación interactiva** en **Python** utilizando **Streamlit**, que permite visualizar, analizar y **predecir** patrones en los datos de ventas de manera intuitiva, integrando componentes gráficos, métricas de negocio y **modelos de Machine Learning**.  

La solución evoluciona en **cuatro Sprints**:

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

### 📊 SPRINT 4 – Power BI Dashboard & Business Intelligence
- Integración de **dashboards profesionales** con Power BI.
- Creación de **4 vistas especializadas** (General, Clientes, Productos, Medios de Pago).
- Desarrollo de **visualizaciones ejecutivas** para presentaciones.
- Implementación de **Power BI Embedded** para integración web.
- **Dashboards interactivos** replicados en Streamlit como alternativa.
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
- **Presentar insights ejecutivos** mediante dashboards profesionales de Power BI.
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
### 📊 **7. SPRINT 4 - Power BI Dashboard**

**Objetivo:** Proporcionar dashboards profesionales interactivos para análisis ejecutivo y presentaciones de negocio.

#### 🎯 **7.1. Visión General del Dashboard**

El módulo de Power BI integra **visualizaciones avanzadas** diseñadas para diferentes audiencias dentro de la organización:
- **Gerencia:** Vista general con KPIs principales
- **Ventas:** Análisis detallado de clientes y conversión
- **Operaciones:** Gestión de productos e inventario
- **Finanzas:** Análisis de medios de pago y flujo de caja

---

#### 📈 **7.2. Dashboard General**

**Propósito:** Vista ejecutiva de alto nivel con métricas clave del negocio.

**KPIs Principales:**
- 💰 **Total Ventas:** $2.65 millones (período febrero-junio 2024)
- 📦 **Cantidad Vendidos:** 1,016 mil unidades
- 🎯 **Ticket Promedio:** $22.10 mil por transacción

**Visualizaciones incluidas:**
1. **Evolución temporal de las ventas** (gráfico de líneas)
   - Muestra tendencia mensual de febrero a junio 2024
   - Identifica picos y valles en el período
   - Útil para detectar estacionalidad

2. **Cantidad total por categoría** (gráfico de barras)
   - **Alimentos:** 846 unidades (83%)
   - **Limpieza:** 170 unidades (17%)
   - Evidencia clara dominancia de alimentos

3. **Selector de medio de pago** (slicer interactivo)
   - Filtros: Efectivo, QR, Tarjeta, Transferencia
   - Permite análisis específico por método de pago
   - Actualiza todos los gráficos dinámicamente

**Aplicación práctica:**
- Reportes mensuales para directorio
- Seguimiento de metas trimestrales
- Identificación rápida de anomalías

---

#### 👥 **7.3. Dashboard de Clientes**

**Propósito:** Análisis profundo del comportamiento y valor de los clientes.

**Métricas destacadas:**
- 👥 **Clientes Totales:** 67 clientes únicos
- 💰 **Ticket Promedio por Cliente:** $39.57 mil
- 🏆 **Top Cliente:** Agustina Flores ($132.16 mil)

**Visualizaciones incluidas:**
1. **Top 10 clientes según importe** (gráfico de barras horizontal)
   - Ranking de clientes más valiosos
   - Valores en miles de pesos
   - Identificación de clientes VIP:
     - Agustina Flores: $132.16 mil
     - Bruno Castro: $118.79 mil
     - Bruno Diaz: $90.7 mil

2. **Importe total por ciudad** (gráfico de barras)
   - **Río Cuarto:** Líder con mayor facturación
   - **Alta Gracia:** Segunda posición
   - **Córdoba:** Tercera posición
   - Permite identificar mercados clave

3. **Frecuencia de compras vs. importe por cliente** (gráfico de dispersión)
   - Eje X: Frecuencia de compras (1-5)
   - Eje Y: Ventas por cliente
   - Identifica clientes de alto valor vs alta frecuencia
   - Detecta oportunidades de upselling

4. **Filtros dinámicos:**
   - Categoría: Alimentos, Limpieza, Otros
   - Mes: January - June
   - Interacción cruzada entre visualizaciones

**Estrategias derivadas:**
- Programas de lealtad para top 10
- Expansión en ciudades con bajo rendimiento
- Campañas para aumentar frecuencia de compra

---

#### 📦 **7.4. Dashboard de Productos**

**Propósito:** Gestión de inventario y análisis de rendimiento por producto y categoría.

**Insights principales:**
- 🏆 **Producto Top:** Desodorante aerosol ($94 mil)
- 📊 **Categoría Dominante:** Alimentos ($2.21 millones)
- 🌍 **Ciudad Top:** Río Cuarto ($792 mil)

**Visualizaciones incluidas:**
1. **Top 5 productos más vendidos por importe** (gráfico de barras)
   - Desodorante aerosol: $94 mil
   - Queso rallado 150g: $90 mil
   - Pizza congelada muzzarella: $86 mil
   - Ron 700ml: $81 mil
   - Yerba mate suave 1kg: $78 mil

2. **Evolución de ventas por categoría** (gráfico de líneas)
   - Comparación temporal: Alimentos vs Limpieza
   - Identifica tendencias por categoría
   - Útil para planificación de compras

3. **Mapa de calor: Ventas por ciudad y categoría** (tabla matriz)
   - Cruce ciudad × categoría con totales
   - Formato: $XXX,XXX por celda
   - Fila "Total" con suma por categoría
   - Columna "Total" con suma por ciudad
   - **Ejemplo:** Río Cuarto-Alimentos: $655,578

4. **Filtro por medio de pago** (slicer)
   - Análisis de qué productos se compran con cada método
   - Detecta patrones de compra según medio de pago

**Aplicación práctica:**
- Reabastecimiento basado en ventas históricas
- Negociación con proveedores de productos top
- Estrategia de surtido por ciudad
- Promociones cruzadas entre categorías

---

#### 💳 **7.5. Dashboard de Medios de Pago**

**Propósito:** Análisis de preferencias de pago y optimización de métodos de cobro.

**Métricas clave:**
- 💵 **Efectivo:** $934.82 mil (35.3%)
- 📱 **QR:** $714.28 mil (26.9%)
- 💳 **Tarjeta:** $542.22 mil (20.4%)
- 🏦 **Transferencia:** $460.10 mil (17.4%)

**Visualizaciones incluidas:**
1. **Participación de medios de pago en las ventas** (gráfico de torta/barras)
   - Muestra distribución porcentual
   - Efectivo domina con 35.3%
   - Identifica dependencia de métodos físicos vs digitales

2. **Evolución de ventas por medio de pago** (gráfico de líneas)
   - Tendencia mensual (Mes 1-6)
   - 4 líneas (una por medio de pago)
   - Detecta cambios en preferencias a lo largo del tiempo

3. **Cantidad por medio de pago y categoría** (gráfico de barras apiladas)
   - **Efectivo:** 296 alimentos + 49 limpieza
   - **QR:** 211 alimentos + 55 limpieza
   - **Transferencia:** 179 alimentos + 39 limpieza
   - **Tarjeta:** 160 alimentos + 27 limpieza
   - Permite identificar correlación producto-pago

4. **Cantidad vendida por mes y medio de pago** (gráfico de barras horizontal)
   - Vista de 6 meses con desglose por método
   - Identifica picos estacionales por medio
   - Útil para prever flujo de caja

**Estrategias derivadas:**
- Promociones con descuentos en medios menos usados
- Inversión en infraestructura digital (QR, POS)
- Negociación de comisiones con procesadores
- Campaña de educación sobre medios electrónicos

#### 📥 **7.6. Funcionalidades Adicionales**

**Exportación de reportes:**
- 📄 Descarga de dashboard completo en PDF
- 📊 Exportación de tablas filtradas a CSV
- 📈 Capturas de gráficos individuales

**Filtros globales:**
- 📅 Rango de fechas personalizado
- 🏙️ Filtro por ciudad
- 🏷️ Filtro por categoría
- 💳 Filtro por medio de pago

**Interactividad:**
- 🔄 Botón de refresco de datos
- 🔗 Link directo a Power BI Service
- 📱 Diseño responsive para móviles
- 🎨 Tema profesional coherente con la app

---

#### 💼 **7.7. Casos de Uso por Rol**

| Rol | Vista Recomendada | Objetivo | Frecuencia |
|-----|------------------|----------|------------|
| **Gerente General** | Dashboard General | Monitoreo de KPIs principales | Diaria |
| **Gerente de Ventas** | Dashboard de Clientes | Identificar oportunidades de venta | Semanal |
| **Gerente de Operaciones** | Dashboard de Productos | Gestión de inventario | Diaria |
| **Gerente Financiero** | Dashboard de Medios de Pago | Optimización de flujo de caja | Mensual |
| **Analista de Datos** | Todas las vistas + ML | Insights profundos y predicciones | Continua |
---



### 🧠 **8. Resultados y Conclusiones**  
Síntesis de los principales hallazgos del análisis y posibles líneas de acción para optimizar las ventas, incluyendo insights de los modelos de Machine Learning.

---

## 🛠️ Tecnologías Utilizadas  

| Tipo | Herramienta |
|------|--------------|
| **Lenguaje principal** | Python 🐍 |
| **Framework web** | Streamlit 📊 |
| **Análisis y manipulación** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Business Intelligence** | Power BI Desktop, Power BI Service |
| **Dashboards** | Power BI Embedded, Streamlit Components |
| **Machine Learning** | Scikit-learn |
| **Modelos implementados** | Linear Regression, Random Forest, K-Means |
| **Formato de datos** | CSV, PBIX |
| **Diseño visual** | Streamlit Theme + Estilo profesional personalizado |

---

## 📊 Resultados Obtenidos  

### Sprint 1 y 2:
- Se logró una **visualización integral y automatizada** de los datos de ventas.  
- Se identificaron **productos clave y categorías con mayor rentabilidad**.  
- Se detectaron **picos de demanda por mes y medio de pago preferido**.  
- Se generó una interfaz amigable que facilita la **toma de decisiones comerciales**.  

### Sprint 3 Machine Learning:
- **Modelo de Regresión:** Predice ventas futuras con R² = 0.338, útil para planificación de inventario a pesar del dataset limitado.
- **Modelo de Clasificación:** Accuracy del 44.9%, mejor que azar (25%), permite personalizar experiencia de pago.
- **Modelo de Clustering:** Segmenta clientes automáticamente en 4-5 grupos con características diferenciadas, habilitando marketing personalizado.
- Se integró capacidad **predictiva e inteligente** a la plataforma.
- Se habilitó **segmentación automática** para estrategias de retención.

### Sprint 4 (Power BI):
- **Dashboards profesionales:** 4 vistas especializadas (General, Clientes, Productos, Medios de Pago).
- **KPIs ejecutivos:** Total Ventas ($2.65M), Ticket Promedio ($22.1K), 67 clientes únicos.
- **Integración dual:** Power BI Embedded + Réplica en Streamlit para máxima flexibilidad.
- **Insights accionables:** Top productos, ciudades clave, patrones de pago identificados.
- **Exportación:** Reportes en PDF para presentaciones ejecutivas.
- Se habilitó **visualización profesional** para stakeholders y toma de decisiones estratégicas.
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
|
|__ AurelionVentas.pbix/        # Archivo Power Bi

---

## 👨‍💻 Autores  
**Equipo de Desarrollo - Proyecto Aurelion**  
📅 Versión: Sprint 4 (Power BI Dashboard)  
💻 Desarrollado con Python, Streamlit, Scikit-learn y Power BI  
🤖 Modelos: Regresión Lineal, Random Forest, K-Means Clustering  
📊 Dashboards: Power BI Desktop & Embedded Integration
---

## 📄 Licencia
Proyecto educativo desarrollado como parte del curso de Machine Learning - IBM.