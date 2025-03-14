import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Streamlit Web App UI
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (28x28 pixels) to recognize.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the image to grayscale
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for CNN model

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Predict digit
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Show result
    st.write(f"### 🧠 Predicted Digit: **{predicted_digit}**")
