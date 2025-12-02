# pseudocodigo.md

# 💡 Pseudocódigo

INICIO PROGRAMA

    IMPORTAR librería Streamlit COMO st

    EJECUTAR en terminal con el comando "streamlit run app.py"

    CONFIGURAR página con título "Documentación Aurelion"

    MOSTRAR título principal: "📚 Proyecto Aurelion"
    MOSTRAR subtítulo: "Consulta interactiva de documentación"

    # --- MENÚ LATERAL ---
    CREAR menú lateral con opciones:
        - "documentacion.md"
        - "dataset.md"
        - "pseudocodigo.md"
        - "copilot.md"
        - "diagrama.drawio.png"
        - "salir"
    GUARDAR opción elegida en variable 'menu'

    # --- FUNCIÓN PARA MOSTRAR ARCHIVOS MARKDOWN ---
    DEFINIR función mostrar_markdown(nombre_archivo):
        INTENTAR:
            ABRIR archivo con nombre_archivo en modo lectura UTF-8
            LEER su contenido y guardarlo en variable 'contenido'
            MOSTRAR contenido usando st.markdown()
        SI archivo no existe:
            MOSTRAR mensaje de error con st.error()
        SI ocurre otro error:
            MOSTRAR mensaje con detalle del error

    # --- LÓGICA PRINCIPAL SEGÚN OPCIÓN ---
    SI 'menu' == "documentacion.md":
        MOSTRAR encabezado "📄 Documentación General"
        LLAMAR mostrar_markdown("documentacion.md")

    SINO SI 'menu' == "dataset.md":
        MOSTRAR encabezado "🧾 Dataset"
        LLAMAR mostrar_markdown("dataset.md")

    SINO SI 'menu' == "pseudocodigo.md":
        MOSTRAR encabezado "💡 Pseudocódigo"
        LLAMAR mostrar_markdown("pseudocodigo.md")

    SINO SI 'menu' == "copilot.md":
        MOSTRAR encabezado "🤖 Instrucciones para Copilot"
        LLAMAR mostrar_markdown("copilot.md")

    SINO SI 'menu' == "diagrama.drawio.png":
        MOSTRAR diagrama.drawio.png

    SINO SI 'menu' == "salir":
        MOSTAR Mensajes de despedida e indicaciones para cerrar la ventana

    # --- PIE DE PÁGINA ---
    MOSTRAR línea separadora en el menú lateral
    MOSTRAR información de crédito: "Desarrollado para el Proyecto Aurelion — IBM 2025"

FIN PROGRAMA
