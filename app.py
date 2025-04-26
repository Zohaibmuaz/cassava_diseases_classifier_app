# app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Cassava Disease Classifier", page_icon="🌿")

# Class names
class_names = [
    'cassava-bacterial-blight-cbb',
    'cassava-brown-streak-disease-cbsd',
    'cassava-green-mottle-cgm',
    'cassava-healthy',
    'cassava-mosaic-disease-cmd'
]

# Load model
@st.cache_resource
def load_model():
    model = torch.load('model_trained_crop_diseases.pth', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Streamlit app layout
st.title('Cassava Crop Disease Classifier 🌿🦠')
st.markdown("Upload an image of a cassava leaf and detect the disease!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    with st.spinner('Analyzing image... 🔍'):
        input_tensor = preprocess_image(image)
        time.sleep(1)  # Small delay for UX
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
            predicted_class = class_names[predicted.item()]

    # Show prediction and confidence
    st.success(f"🎯 **Prediction: {predicted_class}**")
    st.info(f"🧠 **Confidence: {confidence.item()*100:.2f}%**")

    # Plot all class probabilities
    st.subheader("Prediction Probabilities 📊")
    prob_data = pd.DataFrame({
        'Disease': class_names,
        'Probability': probs.numpy()
    }).sort_values(by='Probability', ascending=False)

    st.bar_chart(prob_data.set_index('Disease'))

else:
    st.warning('⚠️ Please upload an image to classify.')
