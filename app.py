import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Configuración de la página con Layout Ancho para aprovechar el espacio bilateral
st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎", layout="wide")

# ==========================================
# Paleta de Colores Orgánica y Estilos Modernos
# ==========================================
st.markdown("""
<style>
    /* Fondo general de la aplicación (Tono crema/hueso muy suave) */
    .stApp {
        background-color: #F7F9F6;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* Personalización del contenedor de carga de archivos */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #81C784;
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Encabezados y títulos */
    h1 {
        color: #1B5E20;
        font-weight: 800;
        margin-bottom: 5px;
    }
    h3 {
        color: #2E7D32;
        font-weight: 600;
    }
    
    /* Caja de resultados de métricas */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 5px solid #4CAF50;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricValue"] {
        color: #2E7D32;
    }
    
    /* Ajuste fino para textos informativos */
    .stMarkdown p {
        color: #4E5D4E;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado principal centrado e integrado al concepto orgánico
st.markdown("<div style='text-align: center; padding: 10px 0;'>", unsafe_allow_html=True)
st.title("🍎 Clasificador Inteligente de Frutas y Verduras")
st.caption("Módulo de Inteligencia Artificial | Allan Manuel Orellana - 20211920128")
st.markdown("Reconocimiento automatizado de productos agrícolas mediante Redes Neuronales Convolucionales.")
st.markdown("</div>---", unsafe_allow_html=True)

# ----------------------------
# CARGAR EL MODELO ENTRENADO
# ----------------------------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_frutas.keras")

modelo = cargar_modelo()

# Lista en orden alfabético estricto
clases_lista = [
    "Almonds 1", "Apple Red Yellow 1", "Avocado Green 1", "Banana 1", 
    "Beetroot 1", "Blackberry 3", "Cabbage white 1", "Cactus fruit green 1", 
    "Cantaloupe 3", "Carambola 1", "Tomato 7"
]

# Diccionario de traducción al español
traduccion_frutas = {
    "Almonds 1": "Almendras",
    "Apple Red Yellow 1": "Manzana Roja Amarilla",
    "Avocado Green 1": "Aguacate Verde",
    "Banana 1": "Banano",
    "Beetroot 1": "Remolacha",
    "Blackberry 3": "Mora",
    "Cabbage white 1": "Repollo Blanco",
    "Cactus fruit green 1": "Tuna Verde (Fruto del Cactus)",
    "Cantaloupe 3": "Melón",
    "Carambola 1": "Carambola (Fruta Estrella)",
    "Tomato 7": "Tomate"
}

# ----------------------------
# INTERFAZ INTERACTIVA BI-COLUMNA
# ----------------------------
# Si no hay archivo, se muestra en pantalla completa. Al subirlo, se distribuye en 2 columnas limpias.
if "archivo_subido" not in st.session_state:
    col_izq, col_der = st.columns([1, 1])
else:
    col_izq, col_der = st.columns([1.1, 0.9], gap="large")

with col_izq:
    st.markdown("### 📸 Captura / Carga de Imagen")
    archivo_subido = st.file_uploader("Arrastra o selecciona la fotografía de la fruta aquí...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Renderizado de la imagen en la columna izquierda con marco estético
    with col_izq:
        imagen_pil = Image.open(archivo_subido)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(imagen_pil, caption="Muestra bajo análisis fotométrico", use_container_width=True)
        
    # Procesamiento y visualización de resultados en la columna derecha
    with col_der:
        st.markdown("### 🧪 Diagnóstico del Modelo")
        
        with st.spinner("Analizando patrones morfológicos y cromáticos..."):
            # ----------------------------
            # PREPROCESAMIENTO DE IMAGEN (Sin alteraciones funcionales)
            # ----------------------------
            imagen_rgb = imagen_pil.convert("RGB")
            img_array = np.array(imagen_rgb)
            img_redimensionada = cv2.resize(img_array, (100, 100))
            img_normalizada = img_redimensionada.astype(float) / 255.0
            input_tensor = img_normalizada.reshape(-1, 100, 100, 3)
            
            # ----------------------------
            # PREDICCIÓN Y TRADUCCIÓN
            # ----------------------------
            prediccion = modelo.predict(input_tensor)
            indice_maximo = np.argmax(prediccion[0], axis=-1)
            porcentaje_confianza = prediccion[0][indice_maximo] * 100
            
            fruta_ingles = clases_lista[indice_maximo]
            fruta_espanol = traduccion_frutas.get(fruta_ingles, fruta_ingles)
        
        # Bloques de resultados rediseñados
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"🍏 **Categoría Detectada:** `{fruta_espanol}`")
        
        # Métrica limpia estilo cuadro de control
        st.metric(
            label="Índice de Certeza Algorítmica", 
            value=f"{porcentaje_confianza:.2f}%",
            delta="Predicción Establecida" if porcentaje_confianza > 75 else "Revisar Iluminación"
        )
        
        # Pequeño panel informativo dinámico según la confianza
        st.markdown("---")
        if porcentaje_confianza > 80:
            st.caption("💡 **Nota:** El modelo presenta un nivel alto de seguridad sobre las características geométricas observadas en esta muestra.")
        else:
            st.caption("⚠️ **Consejo:** Si la fruta no corresponde, intenta tomar la foto desde otro ángulo o mejorando las condiciones de luz ambiental.")
else:
    with col_der:
        st.markdown("### ℹ️ Instrucciones de uso")
        st.info("Para comenzar el reconocimiento, por favor selecciona una imagen del catálogo local en el panel izquierdo.")
        st.markdown("""
        **Clases soportadas actualmente por el sistema:**
        * 🪵 Frutos secos (Almendras)
        * 🍎 Pomáceas (Manzana Roja Amarilla)
        * 🥑 Drupas (Aguacate Verde)
        * 🍌 Tropicales (Banano, Carambola)
        * 🥕 Tubérculos y Crucíferas (Remolacha, Repollo Blanco)
        * 🍇 Bayas y Cucurbitáceas (Mora, Melón)
        * 🌵 Cactáceas (Tuna Verde)
        * 🍅 Solanáceas (Tomate)
        """)
