import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Lien et nom de sortie du modèle
url = "https://drive.google.com/uc?export=download&id=1-2clgdew6-_EtJLIO4pqmOacVol2uNfZ"
output = "K13_best_model_maize_diseases.keras"

# Fonction de téléchargement avec gestion des erreurs
def download_model(url, output):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Gère les erreurs HTTP
        with open(output, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        return True  # Téléchargement réussi
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors du téléchargement : {e}")
        return False

# Charger le modèle avec vérification
@st.cache_resource
def load_model_from_file():
    if not os.path.exists(output):
        if download_model(url, output):
            st.success("Modèle téléchargé avec succès.")
        else:
            st.error("Téléchargement du modèle échoué.")
            return None
    try:
        model = load_model(output)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Charger et afficher le modèle
st.title("🌽 Détection des Maladies des Feuilles de Maïs")

with st.spinner('Chargement du modèle...'):
    model = load_model_from_file()
if model:
    st.success("Modèle chargé avec succès !")
else:
    st.error("Impossible de charger le modèle.")

# Widgets pour charger une image
uploaded_file = st.file_uploader("Téléchargez une image de la feuille", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Ou prenez une photo via la webcam")

# Traitement et prédiction
if uploaded_file or camera_input:
    image = Image.open(uploaded_file or camera_input)
    st.image(image, caption="Image chargée", use_column_width=True)

    # Prétraitement
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    try:
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Classes de maladies
        disease_names = [
            "Cercospora Leaf Spot (Tâche grise)",
            "Common Rust (Rouille commune)",
            "Northern Leaf Blight (Brûlure du nord)",
            "Feuille saine"
        ]
        result = disease_names[predicted_class]

        # Afficher le résultat
        st.markdown(f"**Résultat :** {result}")
        st.progress(confidence)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
