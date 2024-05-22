import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image
import io

st.title("Deteksi Minuman")

# Load model
model = load_model('model.h5')

# Load labels from JSON
with open('class_indices.json', 'r') as f:
    labels = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((150, 150))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Capture image from camera
picture = st.camera_input("Silakan Ambil Gambar")

if picture:
    # Display the captured image
    st.image(picture)

    # Preprocess the image
    img_bytes = picture.getvalue()
    img = preprocess_image(img_bytes)

    # Perform inference
    classes = model.predict(img)
    predicted_class_index = np.argmax(classes, axis=1)[0]
    predicted_class_label = labels[str(predicted_class_index)]
    
    # Display the prediction
    st.write(f'Predicted: {predicted_class_label}')
