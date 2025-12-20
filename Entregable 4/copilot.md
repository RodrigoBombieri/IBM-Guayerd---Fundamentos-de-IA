# copilot.md 

# 🤖 Registro de interacción y asistencia con IA

## 📘 Proyecto: Aplicación de consulta de documentación técnica (Streamlit)

**Fecha:** Octubre 2025  
**Autor:** Rodrigo Sebastián Bombieri  

---

## 🧩 Descripción general

Durante el desarrollo de una aplicación en **Python** utilizando **Streamlit**, se empleó asistencia de **IA (Copilot/ChatGPT)** para estructurar un sistema que permite **consultar documentación técnica desde un menú interactivo**.  

El programa muestra el contenido de distintos archivos Markdown relacionados con el proyecto:  
- `documentacion.md`  
- `dataset.md`  
- `pseudocodigo.md`  
- `copilot.md`  

El objetivo principal fue **facilitar la visualización ordenada** del material del proyecto dentro de una interfaz simple, clara y navegable.  

---

## 💬 Prompts utilizados (reformulados)

1. **Estructura del programa base**  
   > “¿Podrías ayudarme a crear una aplicación en Python con Streamlit que muestre el contenido de distintos archivos markdown según una opción de menú?”

2. **Carga dinámica del contenido**  
   > “¿Cómo puedo hacer para que Streamlit lea automáticamente los archivos markdown (`.md`) y los muestre según la selección del usuario?”

3. **Mejoras visuales**  
   > “¿Qué elementos de Streamlit puedo usar para que la interfaz sea más clara y profesional al mostrar la documentación?”

4. **Integración del pseudocódigo**  
   > “¿Podrías incluir también una sección en el menú para mostrar el pseudocódigo del programa dentro de la aplicación?”

5. **Validación y control**  
   > “¿Cómo puedo asegurarme de que el menú no arroje errores si algún archivo markdown no se encuentra disponible?”

---

## 🧠 Sugerencias de la IA (aceptadas o modificadas)

| Tipo de sugerencia | Descripción | Acción tomada |
|--------------------|--------------|----------------|
| 💡 **Menú interactivo** | Uso de `st.sidebar.radio()` en lugar de `st.selectbox()` para un acceso más claro a las secciones. | ✅ Aceptada |
| 📄 **Visualización de texto** | Reemplazo de `st.write()` por `st.markdown()` para mantener formato original de los archivos `.md`. | ✅ Aceptada |
| 🧱 **Estructura modular** | Crear una función `mostrar_contenido(ruta)` que lea y renderice cada archivo Markdown. | ✅ Aceptada |
| ⚙️ **Gestión de errores** | Agregar manejo de excepciones en la lectura de archivos para evitar interrupciones. | ✏️ Modificada y aplicada parcialmente |
| 🎨 **Aspecto visual** | Incorporar títulos con emojis y secciones claramente separadas para cada documento. | ✅ Aceptada |
| 🔁 **Actualización dinámica** | Permitir que el usuario cambie de sección sin recargar manualmente la aplicación. | ✅ Aceptada |

---

## ⚙️ Mejoras sugeridas (pendientes o implementadas posteriormente)

- Incluir un **modo oscuro/claro** opcional desde la interfaz de usuario.  
- Implementar una **búsqueda por palabras clave** dentro de los markdowns.  
- Agregar una sección adicional con **visualización de diagramas** o flujos generados automáticamente desde pseudocódigo.  
- Centralizar rutas y nombres de archivos en un archivo de configuración (`config.json`).  
- Incorporar un **contador de consultas** para registrar las interacciones del usuario con la documentación.

---

## 📈 Conclusión

La asistencia de IA permitió:  
- Diseñar un menú funcional y atractivo dentro de Streamlit.  
- Automatizar la lectura y presentación de la documentación.  
- Mantener coherencia visual entre las distintas secciones.  
- Reducir el tiempo de desarrollo, priorizando claridad y mantenibilidad.  

El archivo `copilot.md` cumple la función de **registro de colaboración con IA**, garantizando trazabilidad y transparencia en la generación del código y de las decisiones de diseño adoptadas.

---

**Versión:** 1.0  
**Última actualización:** 7 de octubre de 2025  
