# =============================================================
# data_utils.py  |  Proyecto Aurelion - Limpieza y unificación
# =============================================================

import pandas as pd
import os

BASE_PATH = "./Base de datos/"

# --- FUNCIÓN 1: CARGA DE DATOS BRUTOS ---
def cargar_datos():
    print("📥 Cargando archivos CSV originales...")
    try:
        ventas = pd.read_csv(os.path.join(BASE_PATH, "ventas.csv"))
        detalle = pd.read_csv(os.path.join(BASE_PATH, "detalle_ventas.csv"))
        productos = pd.read_csv(os.path.join(BASE_PATH, "productos.csv"))
        clientes = pd.read_csv(os.path.join(BASE_PATH, "clientes.csv"))
        print("✅ Archivos cargados correctamente.")
        return ventas, detalle, productos, clientes
    except Exception as e:
        print(f"❌ Error al cargar los archivos: {e}")
        return None, None, None, None


# --- FUNCIÓN 2: LIMPIEZA Y NORMALIZACIÓN ---
def limpiar_datos(ventas, detalle, productos, clientes):
    print("\n🧹 Iniciando proceso de limpieza y normalización...")

    # Eliminar duplicados
    for nombre, df in {"Ventas": ventas, "Detalle": detalle, "Productos": productos, "Clientes": clientes}.items():
        antes = len(df)
        df.drop_duplicates(inplace=True)
        print(f" - {nombre}: {antes - len(df)} duplicados eliminados.")

    # Completar nulos básicos
    ventas.fillna({"medio_pago": "Desconocido"}, inplace=True)
    detalle.fillna({"cantidad": 0, "importe": 0}, inplace=True)
    productos.fillna({"categoria": "Desconocido", "precio_unitario": 0}, inplace=True)
    clientes.fillna({"ciudad": "Desconocido", "email": "sin_email@aurelion.com"}, inplace=True)

    # Normalización de texto y fechas
    ventas["fecha"] = pd.to_datetime(ventas["fecha"], errors="coerce")
    ventas["medio_pago"] = ventas["medio_pago"].str.strip().str.lower()
    productos["categoria"] = productos["categoria"].str.strip().str.title()
    clientes["ciudad"] = clientes["ciudad"].str.strip().str.title()

    print("✅ Limpieza y normalización completadas.")
    return ventas, detalle, productos, clientes


# --- FUNCIÓN 3: UNIÓN DE LOS DATASETS ---
def unir_datos(ventas, detalle, productos, clientes):
    print("\n🔗 Uniendo datasets...")

    ventas_detalle = pd.merge(detalle, ventas, on="id_venta", how="left")
    ventas_productos = pd.merge(ventas_detalle, productos, on="id_producto", how="left")
    ventas_completas = pd.merge(
        ventas_productos, clientes, on="id_cliente", how="left", suffixes=('_venta', '_cliente')
    )

    # Calcular columnas derivadas
    if "importe" not in ventas_completas.columns:
        ventas_completas["importe"] = ventas_completas["cantidad"] * ventas_completas["precio_unitario"]

    ventas_completas["mes"] = pd.to_datetime(ventas_completas["fecha"]).dt.month

    print(f"📊 Dataset combinado: {ventas_completas.shape[0]} filas, {ventas_completas.shape[1]} columnas.")
    return ventas_completas


# --- FUNCIÓN 4: EXPORTAR RESULTADOS ---
def exportar_archivos(ventas, detalle, productos, clientes, ventas_completas):
    print("\n💾 Guardando archivos limpios y dataset maestro...")

    ventas.to_csv(os.path.join(BASE_PATH, "ventas_limpio.csv"), index=False)
    detalle.to_csv(os.path.join(BASE_PATH, "detalle_ventas_limpio.csv"), index=False)
    productos.to_csv(os.path.join(BASE_PATH, "productos_limpio.csv"), index=False)
    clientes.to_csv(os.path.join(BASE_PATH, "clientes_limpio.csv"), index=False)
    ventas_completas.to_csv(os.path.join(BASE_PATH, "ventas_completas.csv"), index=False)

    print("✅ Archivos guardados correctamente en:", BASE_PATH)

# =====================================================
# NUEVAS FUNCIONES DE CLASIFICACIÓN Y LIMPIEZA FINAL
# =====================================================

# --- 5️⃣ FUNCIÓN: RECLASIFICAR CATEGORÍAS ---
def reclasificar_categorias(productos):
    print("\n🍽️ Reclasificando categorías de productos...")

    # Normalizar nombres
    productos['nombre_producto'] = productos['nombre_producto'].str.lower().str.strip()

    alimentos = [
        'aceite de girasol 1l', 'aceitunas verdes 200g', 'aceitunas negras 200g', 'agua mineral 500ml',
        'alfajor simple', 'alfajor triple', 'arroz largo fino 1kg', 'avena instantánea 250g',
        'azúcar 1kg', 'barrita de cereal 30g', 'bizcochos salados', 'café molido 250g',
        'caldo concentrado verdura', 'caldo concentrado carne', 'caramelos masticables', 'cerveza rubia 1l',
        'cerveza negra 1l', 'chicle menta', 'chocolate con leche 100g', 'chocolate amargo 100g',
        'coca cola 1.5l', 'dulce de leche 400g', 'energética nitro 500ml', 'empanadas congeladas',
        'fanta naranja 1.5l', 'fernet 750ml', 'fideos spaghetti 500g', 'galletitas vainilla',
        'galletitas chocolate', 'garbanzos 500g', 'gin 700ml', 'granola 250g',
        'hamburguesas congeladas x4', 'harina de trigo 1kg', 'helado vainilla 1l', 'helado chocolate 1l',
        'helado de frutilla 1l', 'jugo de manzana 1l', 'jugo de naranja 1l', 'jugo en polvo naranja',
        'jugo en polvo limón', 'leche descremada 1l', 'leche entera 1l', 'lentejas secas 500g',
        'manteca 200g', 'maní salado 200g', 'medialunas de manteca', 'mermelada de durazno 400g',
        'mermelada de frutilla 400g', 'miel pura 250g', 'mix de frutos secos 200g', 'pan lactal integral',
        'pan lactal blanco', 'papas fritas onduladas 100g', 'papas fritas clásicas 100g', 'pepsi 1.5l',
        'pizza congelada muzzarella', 'porotos negros 500g', 'queso cremoso 500g', 'queso untable 190g',
        'queso azul 150g', 'queso rallado 150g', 'ron 700ml', 'sal fina 500g', 'salsa de tomate 500g',
        'sopa instantánea pollo', 'sprite 1.5l', 'stevia 100 sobres', 'té verde 20 saquitos',
        'té negro 20 saquitos', 'turrón 50g', 'verduras congeladas mix', 'vinagre de alcohol 500ml',
        'vino tinto malbec 750ml', 'vino blanco 750ml', 'vodka 700ml', 'whisky 750ml', 'yerba mate suave 1kg',
        'yerba mate intensa 1kg', 'yogur natural 200g'
    ]

    limpieza = [
        'lavandina 1l', 'limpiavidrios 500ml', 'detergente líquido 750ml', 'desengrasante 500ml',
        'papel higiénico x4', 'toallas húmedas x50', 'trapo de piso', 'cepillo de dientes',
        'crema dental 90g', 'hilo dental', 'jabón de tocador', 'shampoo 400ml',
        'mascarilla capilar', 'desodorante aerosol', 'servilletas x100'
    ]

    def clasificar_producto(nombre):
        if nombre in alimentos:
            return 'alimentos'
        elif nombre in limpieza:
            return 'limpieza'
        else:
            return 'otros'

    productos['categoria'] = productos['nombre_producto'].apply(clasificar_producto)
    print("✅ Reclasificación completada.")
    return productos


# --- 6️⃣ FUNCIÓN: ELIMINAR COLUMNAS DUPLICADAS ---
def eliminar_columnas_duplicadas(df):
    print("\n🧩 Eliminando y consolidando columnas duplicadas...")

    reemplazos = {
        "nombre_producto_x": "nombre_producto",
        "nombre_producto_y": "nombre_producto",
        "precio_unitario_x": "precio_unitario",
        "precio_unitario_y": "precio_unitario",
        "nombre_cliente_venta": "nombre_cliente",
        "nombre_cliente_cliente": "nombre_cliente",
        "email_venta": "email",
        "email_cliente": "email"
    }

    # --- 1️⃣ Renombrar columnas duplicadas a su versión limpia
    df = df.rename(columns=reemplazos)

    # --- 2️⃣ Si existen columnas duplicadas después del rename, eliminarlas conservando la primera
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # --- 3️⃣ Reordenar las columnas principales para dejar el dataset más claro
    columnas_ordenadas = [
        "id_venta", "fecha", "mes", "id_cliente", "nombre_cliente", "email", "ciudad",
        "fecha_alta", "id_producto", "nombre_producto", "categoria",
        "cantidad", "precio_unitario", "importe", "medio_pago"
    ]
    columnas_finales = [col for col in columnas_ordenadas if col in df.columns]
    df = df[columnas_finales]

    print("✅ Columnas duplicadas consolidadas y dataset ordenado correctamente.")
    return df




# --- FUNCIÓN PRINCIPAL ---
def ejecutar_limpieza_total():
    print("\n🚀 Iniciando proceso completo de limpieza y combinación...\n")
    ventas, detalle, productos, clientes = cargar_datos()
    if any(x is None for x in (ventas, detalle, productos, clientes)):
        print("❌ Error en la carga, proceso detenido.")
        return

    ventas, detalle, productos, clientes = limpiar_datos(ventas, detalle, productos, clientes)
    productos = reclasificar_categorias(productos)
    ventas_completas = unir_datos(ventas, detalle, productos, clientes)
    ventas_completas = eliminar_columnas_duplicadas(ventas_completas)
    exportar_archivos(ventas, detalle, productos, clientes, ventas_completas)
    print("\n🎉 Proceso finalizado exitosamente. Dataset listo para análisis.")
    return ventas_completas


ejecutar_limpieza_total()