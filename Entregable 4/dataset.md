# dataset.md

# 📊DATASET

## 📈 Fuente, definición y estructura

**Fuente:**
Base de datos simulada de ventas en formato **CSV**.

### Definición y estructura de datasets:

#### 'productos.csv'
- 'id_producto' (int) - Identificador único
- 'nombre_producto' (str) - Nombre descriptivo
- 'categoria' (str) - Categoría del producto
- 'precio_unitario' (float) - Precio por unidad

#### 'clientes.csv'
- 'id_cliente' (int) - Identificador único
- 'nombre_cliente' (str) - Nombre completo
- 'email' (str) - Email de contacto
- 'ciudad' (str) - Ubicación
- 'fecha_alta' (date) - Fecha de alta en sistema

#### 'detalles_ventas.csv'
- 'id_ventas' (int) - Identificador único
- 'id_producto' (int) - Clave foránea de relación con Productos
- 'categoria' (str) - Categoría del producto
- 'nombre_producto' (str) - Nombre descriptivo (VER)
- 'cantidad' (int) - Unidades vendidas
- 'precio_unitario' (float) - Precio por unidad
- 'importe' (float) - precio_unitario * cantidad

### 'ventas.csv'
- 'id_venta' (int) - Identificador único
- 'fecha' (date) - Fecha de la venta
- 'id_cliente' (int) - Clave foránea de relación con Clientes
- 'nombre_cliente' (str) - Nombre del cliente (VER)
- 'email' - (str) Email de contacto (VER)
- 'medio_pago' (str) - Medio de pago elegido


### Tipos y escala:
- Datos estructurados, escala pequeña (archivos CSV locales).
- Tipos de datos: **enteros, flotantes, cadenas de texto, fechas**.