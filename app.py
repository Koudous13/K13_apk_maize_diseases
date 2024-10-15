import os
import gdown
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


# Charger le modèle de prédiction
url = 'https://drive.google.com/uc?export=download&id=1-2clgdew6-_EtJLIO4pqmOacVol2uNfZ'
output = 'K13_best_model_maize_diseases.keras'

@st.cache_resource
def download_and_load_model():
    # Télécharger le fichier si nécessaire
   try:
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        model = load_model(output, custom_objects=None, compile=True)
        return model
   except Exception as e:
        st.error("Erreur lors du chargement du modèle: " + str(e))
        return None
       
# Appliquer un thème CSS pour l'application
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #42a5f5 0%, #478ed1 100%);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .reportview-container {
        background: #1c1c1c;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: scale(1.1);
    }
    .image-container {
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    .prediction-box {
        background-color: #242424;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    .prediction-box h2 {
        font-size: 32px;
        color: #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🌽 Détection des Maladies des Feuilles de Maïs")

# Charger le modèle en gérant les erreurs possibles
try:
    with st.spinner('Loading model...'):
        model = download_and_load_model()
        st.success('Modèle chargé avec succès !')
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {str(e)}")

st.subheader("Chargez une photo de feuille de maïs ou utilisez la webcam.")

# Widgets pour charger une image ou en prendre une avec la webcam
uploaded_file = st.file_uploader("Téléchargez une image de la feuille", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Ou prenez une photo via la webcam")

# Traiter l'image si une entrée est fournie
if uploaded_file or camera_input:
    with st.spinner("Traitement de l'image..."):
        try:
            image = Image.open(uploaded_file or camera_input)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, caption="Image chargée", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Prétraitement de l'image pour la prédiction
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Prédire la classe de la maladie
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Afficher une barre de progression dynamique
            st.progress(confidence)

            # Résultat de la prédiction
            disease_names = [
                "Cercospora Leaf Spot (Tâche grise)",
                "Common Rust (Rouille commune)",
                "Northern Leaf Blight (Brûlure du nord)",
                "Feuille saine"
            ]
            result = disease_names[predicted_class]

            st.markdown(f"""
                <div class="prediction-box">
                    <h2>⚠️ {result}</h2>
                    <p>Confiance : {confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {str(e)}")


# Footer
st.markdown("""
    <hr>
    <div style="text-align: center; color: #bbb;">
        <p>🌽 Application de Détection des Maladies des Feuilles de Maïs - Propulsée par l'IA</p>
        <p>© 2024. Tous droits réservés.</p>
        <p> Koudous Daouda +22959009829 </p>
    </div>
""", unsafe_allow_html=True)
