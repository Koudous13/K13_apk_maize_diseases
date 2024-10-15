import os
import gdown
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Définir les variables du modèle
url = 'https://drive.google.com/uc?export=download&id=1-2clgdew6-_EtJLIO4pqmOacVol2uNfZ'
output = 'K13_best_model_maize_diseases.keras'

# Fonction pour télécharger et charger le modèle
@st.cache_resource
def download_and_load_model():
    try:
        # Vérifier si le fichier est déjà téléchargé
        if not os.path.exists(output):
            st.info("Téléchargement du modèle en cours...")
            gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Charger le modèle
        model = load_model(output)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

# Appliquer un thème CSS pour l'application
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #42a5f5 0%, #478ed1 100%); color: white; font-family: 'Arial', sans-serif; }
    .reportview-container { background: #1c1c1c; border-radius: 10px; padding: 15px; }
    .stButton>button { background-color: #28a745; color: white; padding: 12px 24px; border-radius: 8px; font-weight: bold; }
    .prediction-box { background-color: #242424; padding: 20px; margin-top: 20px; text-align: center; }
    .prediction-box h2 { font-size: 32px; color: #17a2b8; }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🌽 Détection des Maladies des Feuilles de Maïs")

# Charger le modèle avec gestion d'erreur
with st.spinner('Chargement du modèle...'):
    model = download_and_load_model()

if model:
    st.success('Modèle chargé avec succès !')
else:
    st.error("Le modèle n'a pas pu être chargé.")
    st.stop()  # Arrête l'application si le modèle n'est pas disponible

# Widgets pour charger une image ou en prendre une avec la webcam
uploaded_file = st.file_uploader("Téléchargez une image de la feuille", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Ou prenez une photo via la webcam")

# Traitement de l'image si une entrée est fournie
if uploaded_file or camera_input:
    try:
        image = Image.open(uploaded_file or camera_input)
        st.image(image, caption="Image chargée", use_column_width=True)

        # Prétraitement de l'image pour la prédiction
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prédire la classe de la maladie
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Afficher le résultat de la prédiction
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
