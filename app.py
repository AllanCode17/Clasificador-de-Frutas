import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎", layout="centered")

# ==========================================
# CAMBIO DE APARIENCIA POR CSS (NUEVA PALETA DE COLORES)
# ==========================================
st.markdown("""
<style>
    /* Fondo general oscuro satinado */
    .stApp {
        background-color: #0D1117;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Título principal y textos */
    h1 {
        color: #FFB300 !important;
        font-weight: 800 !important;
        text-shadow: 0px 2px 10px rgba(255, 179, 0, 0.2);
    }
    
    .stMarkdown p, div {
        color: #C9D1D9;
    }

    /* Caja de subida de archivos personalizada */
    div[data-testid="stFileUploader"] {
        background-color: #161B22 !important;
        border: 2px dashed #30363D !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    /* Contenedores de éxito e información refinados */
    .stAlert {
        background-color: #161B22 !important;
        border-radius: 10px !important;
        border: 1px solid #30363D !important;
    }
    
    /* Asegurar que las alertas de predicción resalten con el tono dorado */
    div[data-testid="stNotification"] {
        border-left: 5px solid #FFB300 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Clasificador de Frutas - Inteligencia_Artificial - Allan Manuel Orelana Orellana 20211920128")
st.write("Sube una imagen para que el modelo MobileNetV2 identifique a qué categoría pertenece.")

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
# INTERFAZ DE CARGA DE ARCHIVOS
# ----------------------------
archivo_subido = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido)
    st.image(imagen_pil, caption="Imagen subida", use_container_width=True)
    
    st.write("Clasificando...")
    
    # ----------------------------
    # PREPROCESAMIENTO DE IMAGEN
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
    
    # Mostrar resultados estilizados en español
    st.success(f"**Predicción:** {fruta_espanol}")
    st.info(f"**Confianza del modelo:** {porcentaje_confianza:.2f}%")
