import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove
import gdown

# Load your trained model
output = "custom_fpi.keras"
gdown.download("https://drive.google.com/file/d/1R4mTVbREr9wL6dzeBOSpu73urCih1pA_/view?usp=sharing", output,)
model = tf.keras.models.load_model(output)  # Update with your model file

# Define class labels
class_labels = ["Bacterial Blight", "Frogeye Leaf Spot","Healthy", "Potassium Deficiency"]

# Streamlit UI
st.title("🌱 Soybean Leaf Disease Classifier")
st.write("Upload an image of a soybean leaf, and the model will predict the disease.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((256, 256))  # Adjust based on model input size
    image = remove(image).convert("RGB")
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"Prediction: {class_labels[predicted_class]}")
    st.info(f"Confidence: {confidence:.2f}%")

