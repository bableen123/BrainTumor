import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Set title
st.title("🧠 Brain Tumor Classification with MobileNetV2")
st.write("Upload an MRI brain scan image to classify it into one of four categories:")

# Class names (same order as used during training)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Load trained model
@st.cache_resource
def load_mobilenet_model():
    return load_model("mobilenet_model.h5")

model = load_mobilenet_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)


    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🧠 Predicted Class: **{predicted_class}**")
