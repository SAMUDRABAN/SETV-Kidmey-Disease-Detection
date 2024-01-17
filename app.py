import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Load and display logo
# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the logo image
logo_filename = 'setv_global_cover.jpeg'
logo_path = os.path.join("assets","setv_global_cover.jpeg")

# Load and display the logo image
logo = Image.open(logo_path)
st.image(logo, width=200)

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_kidney_disease_model():
    model = tf.keras.models.load_model(os.path.join("Model", "my_kidney_model2.h5"))
    return model

kidney_disease_model = load_kidney_disease_model()
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Function to prepare image for prediction
def prepare_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit application layout
st.title("SETV Kidney Disease Classification")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg','webp'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # On image upload
    if st.button('Predict'):
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image)
        predicted_class = classes[np.argmax(prediction)]
        
        st.write(f'Prediction: {predicted_class}')
