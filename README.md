# 🪐 Aurelion - Sistema Inteligente de Análisis de Ventas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)
**Plataforma de análisis, visualización y predicción de ventas con Machine Learning y Power BI**

[Demo](#-demo) • [Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Documentación](#-documentación)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Demo](#-demo)
- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelos de Machine Learning](#-modelos-de-machine-learning)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Roadmap](#-roadmap)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)
- [Autores](#-autores)

---

## 🎯 Descripción

**Aurelion** es una aplicación web interactiva desarrollada en Python que permite a negocios de retail **analizar, visualizar y predecir** sus patrones de ventas mediante técnicas de análisis de datos y Machine Learning.

### Problema que resuelve

Las tiendas tradicionales enfrentan dificultades para:
- ❌ Identificar productos con baja rotación
- ❌ Predecir demanda futura para gestión de inventario
- ❌ Segmentar clientes para estrategias personalizadas
- ❌ Anticipar métodos de pago preferidos
- ❌ Presentar insights ejecutivos de forma profesional
  
### Solución

✅ Dashboards interactivos con métricas en tiempo real  
✅ Visualizaciones avanzadas (EDA completo)  
✅ **3 modelos de ML** para predicción y clasificación  
✅ Segmentación automática de clientes (RFM)  
✅ Interfaz intuitiva sin código
✅ **4 dashboards profesionales con Power BI**
---

## 🎬 Demo

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/aurelion.git

# Navegar al directorio
cd aurelion

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

<!-- Agrega aquí un GIF o capturas de pantalla -->
<!-- ![Demo](assets/demo.gif) -->

---

## ✨ Características

### 📊 Análisis Exploratorio (EDA)
- Distribución de ventas por categoría, ciudad y medio de pago
- Series temporales y tendencias
- Heatmaps de actividad comercial
- Análisis de correlaciones

### 📈 Visualizaciones Interactivas
- Top productos y categorías
- Evolución temporal de ventas
- Mapas geográficos de clientes
- Gráficos comparativos personalizables

### 💼 Métricas de Negocio (KPIs)
- Ticket promedio
- Margen por categoría
- Ranking de clientes
- Retención mensual
- Análisis de cohortes

### 🤖 Machine Learning (Sprint 3)

#### 1️⃣ Predicción de Ventas por Producto
- **Modelo:** Regresión Lineal
- **Objetivo:** Predecir unidades vendidas el próximo mes
- **R² Score:** 0.338
- **Uso:** Planificación de inventario

#### 2️⃣ Clasificación de Medio de Pago
- **Modelo:** Random Forest Classifier
- **Objetivo:** Predecir método de pago del cliente
- **Accuracy:** 44.9%
- **Uso:** Personalización de experiencia de pago

#### 3️⃣ Segmentación de Clientes (RFM)
- **Modelo:** K-Means Clustering
- **Objetivo:** Agrupar clientes por comportamiento
- **Segmentos:** Champions, Leales, En Riesgo, Dormidos, Nuevos
- **Uso:** Marketing personalizado y retención

### 📊 Power BI Dashboards (Sprint 4)

#### Dashboard General
- **KPIs Principales:** Total Ventas ($2.65M), Ticket Promedio ($22.1K), Cantidad Vendidos (1,016K)
- **Visualizaciones:** Evolución temporal, Cantidad por categoría, Filtro de medio de pago
- **Uso:** Reportes ejecutivos y seguimiento de metas

#### Dashboard de Clientes
- **Métricas:** 67 clientes únicos, Top 10 clientes por importe, Ticket promedio por cliente
- **Visualizaciones:** Importe por ciudad, Frecuencia vs Importe, Filtros dinámicos
- **Uso:** Identificación de clientes VIP y oportunidades de venta

#### Dashboard de Productos
- **Insights:** Top 5 productos, Evolución por categoría, Mapa de calor ciudad×categoría
- **Visualizaciones:** Barras, líneas temporales, tabla matriz interactiva
- **Uso:** Gestión de inventario y planificación de compras

#### Dashboard de Medios de Pago
- **Distribución:** Efectivo (35.3%), QR (26.9%), Tarjeta (20.4%), Transferencia (17.4%)
- **Visualizaciones:** Evolución mensual, Cantidad por categoría, Filtros cruzados
- **Uso:** Optimización de métodos de cobro y flujo de caja
---



## 🛠️ Tecnologías

<div align="center">

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **Framework Web** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **Data Science** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **Visualización** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| **Business Intelligence** | ![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black) |
</div>

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Power BI Desktop (opcional, para editar dashboards)
  
### Paso a Paso

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/aurelion.git
cd aurelion
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Verificar estructura de datos**
Asegúrate de tener los archivos necesarios:
```
Base de datos/ventas_completas.csv
Base de datos/ventas_completas.csv
AurelionVentas_v2.pbix  # Archivo Power BI
AurelionVentas_v2.pdf   # PDF del dashboard (opcional)
```

5. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

### Configuración de Power BI (Opcional)

Para integrar Power BI Embedded:

1. Publica tu reporte en Power BI Service
2. Obtén la URL de incrustación
3. Configura en `app.py`:
```python
powerbi_embed_url = "https://app.powerbi.com/view?r=TU_URL_AQUI"
```
---

## 📖 Uso

### Navegación de la Aplicación

#### 1. **Inicio**
Carga y explora el dataset con estadísticas básicas

#### 2. **Análisis Exploratorio (EDA)**
- Visualiza distribuciones y correlaciones
- Analiza tendencias temporales
- Explora heatmaps de actividad

#### 3. **Visualizaciones Generales**
- Top productos y categorías
- Ventas por región y medio de pago
- Gráficos comparativos interactivos

#### 4. **Métricas de Negocio**
- KPIs principales
- Ranking de clientes
- Análisis de cohortes

#### 5. **Machine Learning** 🤖

##### Predicción de Ventas
1. Selecciona el porcentaje de test
2. Visualiza métricas del modelo (R², MAE, RMSE)
3. Revisa predicciones para el próximo mes
4. Descarga resultados en CSV

##### Clasificación de Medio de Pago
1. Ajusta parámetros de entrenamiento
2. Analiza matriz de confusión
3. **Prueba con nuevas transacciones** (simulador interactivo)
4. Revisa análisis de errores

##### Segmentación de Clientes
1. Define número de clusters (K)
2. Visualiza método del codo y Silhouette Score
3. Explora características de cada segmento
4. Descarga lista de clientes segmentados

---

## 🤖 Modelos de Machine Learning

### 📦 1. Predicción de Ventas por Producto

```python
from sklearn.linear_model import LinearRegression

# Features: lag_1, lag_2, lag_3, precio_unit, categoria
model = LinearRegression()
model.fit(X_train, y_train)
```

**Métricas:**
- R² Score: 0.338
- MAE: 1.1 unidades
- RMSE: 1.4 unidades

**Aplicación:** Optimizar compras y gestión de inventario

---

### 💳 2. Clasificación de Medio de Pago

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

**Métricas:**
- Accuracy: 44.9%
- Precision: 44.4%
- Recall: 44.9%

**Aplicación:** Personalizar opciones de pago mostradas

---

### 👥 3. Segmentación RFM

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm_data)

model = KMeans(n_clusters=4, random_state=42)
labels = model.fit_predict(X_scaled)
```

**Segmentos identificados:**
- 🌟 Champions (VIP)
- 💚 Clientes Leales
- 🟡 En Riesgo
- 🔴 Dormidos
- 🆕 Nuevos

**Aplicación:** Campañas de marketing dirigidas

#### 4. **SPRINT 4 - Power BI** 📊

##### 📊 Dashboard General
- **Visualiza:** KPIs principales, evolución temporal, cantidad por categoría
- **Filtra por:** Medio de pago (Efectivo, QR, Tarjeta, Transferencia)
- **Usa para:** Reportes mensuales, seguimiento de metas, detección de anomalías

##### 👥 Dashboard de Clientes
- **Visualiza:** Top 10 clientes, importe por ciudad, frecuencia vs importe
- **Filtra por:** Categoría, mes, ciudad
- **Usa para:** Identificar clientes VIP, expansión geográfica, campañas de frecuencia

##### 📦 Dashboard de Productos
- **Visualiza:** Top 5 productos, evolución por categoría, mapa de calor ciudad×categoría
- **Filtra por:** Medio de pago
- **Usa para:** Reabastecimiento, negociación con proveedores, estrategia de surtido

##### 💳 Dashboard de Medios de Pago
- **Visualiza:** Participación en ventas, evolución mensual, cantidad por categoría
- **Análisis detallado:** Selecciona un medio de pago para insights profundos
- **Usa para:** Optimización de métodos de cobro, inversión en infraestructura, negociación de comisiones

##### 🔗 Dashboard Completo (Power BI Embed)
- **Opción A:** Embeber dashboard publicado en Power BI Service
- **Opción B:** Visualizar PDF del dashboard
- **Funciones:** Refrescar, abrir en Power BI, descargar PDF
---

## 📁 Estructura del Proyecto

```
aurelion/
│
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
├── documentacion.md                # Documentación técnica completa
│
├── Base de datos/
│   └── ventas_completas.csv        # Dataset de ventas
│
├── scripts/
│   ├── utils.py                    # Funciones auxiliares
│   ├── graficos.py                 # Generación de visualizaciones
│   └── metricas.py                 # Cálculo de KPIs
│
└── scriptsML/                      # Machine Learning (Sprint 3)
    ├── regression.py               # Predicción de ventas
    ├── classification.py           # Clasificación de pago
    └── clustering.py               # Segmentación RFM
│
└── AurelionVentas_v2.pbix          # Dashboard Power BI
```

---

## 📊 Resultados

### Insights Obtenidos

✅ **Productos más rentables** identificados automáticamente  
✅ **Picos de demanda** detectados por temporada  
✅ **Clientes VIP** segmentados (top 20% genera 60% ingresos)  
✅ **Predicción de inventario** con 67% de precisión  
✅ **Segmentación automática** en 4 grupos diferenciados  
✅ **Dashboards ejecutivos** con 4 vistas especializadas

### Impacto en el Negocio

- 📈 Mejora en planificación de compras
- 🎯 Marketing personalizado por segmento
- 💰 Reducción de quiebres de stock
- 🔄 Mayor retención de clientes en riesgo
- 📊 Presentaciones ejecutivas profesionales
- 👔 Mejora en toma de decisiones estratégicas
---

## 🗺️ Roadmap

### Versión Actual (v4.0)
- [x] EDA completo
- [x] Visualizaciones interactivas
- [x] Métricas de negocio
- [x] 3 modelos de Machine Learning
- [x] Segmentación RFM
- [x] 4 dashboards profesionales con Power BI
- [x] Integración Power BI Embedded

#### v4.1 - Mejoras de ML
- [ ] Ampliar dataset a 12+ meses
- [ ] Implementar XGBoost para mejor accuracy
- [ ] Agregar features de hora del día
- [ ] Cross-validation para modelos

#### v4.2 - Automatización
- [ ] Reentrenamiento automático mensual
- [ ] Alertas por email (stock bajo, clientes en riesgo)
- [ ] API REST para integraciones
- [ ] Dashboard para mobile
- [ ] Automatización de reportes Power BI

#### v5.0 - Avanzado
- [ ] Predicción de churn (abandono de clientes)
- [ ] Sistema de recomendación de productos
- [ ] Análisis de sentimiento de reviews
- [ ] Integración con sistemas ERP
- [ ] Power BI con datos en tiempo real
---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Si quieres mejorar este proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de contribución

- 🐛 Reportar bugs
- 💡 Sugerir nuevas features
- 📝 Mejorar documentación
- 🧪 Agregar tests
- 🎨 Mejorar UI/UX

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👨‍💻 Autores

**Rodrigo Bombieri**

- 📧 Email: rodrigosbombieri@gmail.com
- 💼 LinkedIn: [Rodrigo Bombieri](https://www.linkedin.com/in/rodrigobombieri-dev/)
- 🐙 GitHub: [@RodrigoBombieri](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- Inspirado en casos de uso reales de retail
- Desarrollado como proyecto educativo de Machine Learning
- Construido con ❤️ usando Python y Streamlit

---

## 📞 Soporte

Si tienes preguntas o necesitas ayuda:

- 📫 Abre un [Issue](https://github.com/RodrigoBombieri/aurelion/issues)
- 💬 Consulta la [Documentación](documentacion.md)
- 📧 Contacta al equipo

---

<div align="center">

**⭐ Si este proyecto te fue útil, dale una estrella en GitHub ⭐**

Hecho con 🪐 por el equipo de Aurelion

# Certificado de finalización del curso
<img width="2000" height="1414" alt="Rodrigo Sebastián Bombieri" src="https://github.com/user-attachments/assets/b1ef435a-b019-4719-acbd-34b3a9da29c8" />

[⬆ Volver arriba](#-aurelion---sistema-inteligente-de-análisis-de-ventas)

</div>
