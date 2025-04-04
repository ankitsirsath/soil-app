import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "C:/Users/Ankit/OneDrive/Desktop/demo_soil/plant_disease_model.h5"  # Update the correct path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = {
    0: 'Tomato Spider Mites (Two-Spotted Spider Mite) 🕷️',
    1: 'Healthy Bell Pepper 🌶️✅',
    2: 'Tomato Septoria Leaf Spot 🍃⚠️',
    3: 'Tomato Late Blight 🍅🦠',
    4: 'Tomato Bacterial Spot 🍅❌',
    5: 'Tomato Target Spot 🍅🎯',
    6: 'Tomato Early Blight 🍅⚠️',
    7: 'Potato Healthy 🥔✅',
    8: 'Potato Early Blight 🥔⚠️',
    9: 'Potato Late Blight 🥔❌',
    10: 'Tomato Mosaic Virus 🍅🦠',
    11: 'Tomato Yellow Leaf Curl Virus 🍅🟡',
    12: 'Bell Pepper Bacterial Spot 🌶️❌',
    13: 'Tomato Healthy 🍅✅',
    14: 'Tomato Leaf Mold 🍃🦠'
}  # Ensure this matches your dataset

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Match model input size
    image = np.array(image) / 255.0  # Normalize pixel values (0-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.set_page_config(page_title="Soil Disease Classifier", layout="centered")
st.title("🌱 Plant Disease Classification Model")
st.write("Upload an image of a leaf to classify its disease type.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Confidence percentage

    # Get label
    predicted_label = class_labels.get(predicted_class, "Unknown ❓")

    # Display results     
    st.subheader("🔍 Prediction Result")
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence Level:** {confidence:.2f}%")

    # Optional: Show Raw Prediction Probabilities
    st.write("📊 **Prediction Probabilities:**")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels.get(i, 'Unknown')}: {prob * 100:.2f}%")
