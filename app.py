'''import os
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
        model = load_model(output)
        return model
   except Exception as e:
        st.error("Erreur lors du chargement du modèle: " + str(e))
        return None



# Appliquer un thème CSS pour l'application
st.markdown("""
    <style>
    /* Style global de la page */
    body {
        background: linear-gradient(135deg, #42a5f5 0%, #478ed1 100%);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #ffffff;
    }
    .reportview-container {
        background: #1c1c1c;
        border-radius: 10px;
        padding: 15px;
    }
    /* Personnalisation des boutons */
    .stButton>button {
        background-color: #28a745;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: scale(1.1);
    }
    /* Animation des images */
    .image-container {
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    /* Style des résultats de prédiction */
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
    .prediction-box p {
        font-size: 18px;
        color: #ddd;
    }
    /* Barres de progression personnalisées */
    .stProgress .css-1yx4hzk {
        background-color: #1c1c1c;
        border-radius: 8px;
        overflow: hidden;
    }
    .stProgress .css-1yx4hzk .css-18tm38b {
        background-color: #42a5f5;
        transition: width 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🌽 Détection des Maladies des Feuilles de Maïs")

with st.spinner('Loading model...'):
    model = download_and_load_model()

st.success('Model loaded successfully!')

st.subheader("Chargez une photo de feuille de maïs ou utilisez la webcam pour en prendre une.")

# Widgets pour charger ou prendre une photo
uploaded_file = st.file_uploader("Téléchargez une image de la feuille de maïs", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Ou prenez une photo via la webcam")

# Si l'utilisateur charge ou prend une image
if uploaded_file or camera_input:
    with st.spinner('Traitement de l\'image...'):
        if uploaded_file:
            image = Image.open(uploaded_file)
        else:
            image = Image.open(camera_input)

        # Afficher l'image chargée avec un effet animé
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, caption="Image chargée", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Prétraitement de l'image pour le modèle
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction de la maladie
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Afficher une barre de progression dynamique
        st.progress(np.max(predictions))

        # Affichage des résultats de la prédiction avec des boîtes stylées
        if predicted_class == 3:
            st.markdown("""
            <div class="prediction-box">
                <h2>🌿 La feuille est saine !</h2>
                <p>Conseil : Continuez à surveiller vos cultures pour détecter les premiers signes de maladies.</p>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_class == 0:
            st.markdown("""
            <div class="prediction-box">
                <h2>⚠️ Cercospora Leaf Spot (Tâche grise)</h2>
                <p>Traitement recommandé : Appliquez des fongicides appropriés. Veillez à l’élimination des débris.</p>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_class == 1:
            st.markdown("""
            <div class="prediction-box">
                <h2>⚠️ Common Rust (Rouille commune)</h2>
                <p>Traitement recommandé : Utilisez des variétés résistantes et appliquez des fongicides précoces.</p>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_class == 2:
            st.markdown("""
            <div class="prediction-box">
                <h2>⚠️ Northern Leaf Blight (Brûlure du nord)</h2>
                <p>Traitement recommandé : Optez pour des variétés résistantes et traitez avec des fongicides spécifiques.</p>
            </div>
            """, unsafe_allow_html=True)

# Bouton de relance pour une nouvelle image
if st.button("Prendre une nouvelle photo ou charger une autre image"):
    st.experimental_rerun()

# Footer pour ajouter un petit design de fin
st.markdown("""
    <hr>
    <div style="text-align: center; color: #bbb;">
        <p>🌽 Application de Détection des Maladies des Feuilles de Maïs - Propulsée par l'IA</p>
        <p>© 2024. Tous droits réservés.</p>
    </div>
""", unsafe_allow_html=True)

'''







import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model

# URL du fichier modèle sur Google Drive
url = 'https://drive.google.com/uc?export=download&id=1-2clgdew6-_EtJLIO4pqmOacVol2uNfZ'
output = 'K13_best_model_maize_diseases.keras'

# Vérifier le répertoire de travail actuel
def get_working_directory():
    current_dir = os.getcwd()
    st.write(f"Répertoire de travail actuel : {current_dir}")
    return current_dir

# Téléchargement et chargement du modèle avec Streamlit cache
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(output):
        st.write("Téléchargement du modèle...")
        gdown.download(url, output, quiet=False)

    # Vérifier que le fichier est bien présent
    assert os.path.exists(output), "Le fichier modèle n'a pas été téléchargé correctement."
    
    # Charger le modèle
    model = load_model(output)
    return model

get_working_directory()

