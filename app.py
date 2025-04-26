import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set page config (First Streamlit command)
st.set_page_config(page_title="Cassava Disease Classifier", page_icon="🌿")

# Classes
class_names = [
    'cassava-bacterial-blight-cbb',
    'cassava-brown-streak-disease-cbsd',
    'cassava-green-mottle-cgm',
    'cassava-healthy',
    'cassava-mosaic-disease-cmd'
]

# Load model from Google Drive
@st.cache_resource
def load_model():
    file_id = '1Euawh26Mb8RCH2AtSQvoA3rQwUalO7MB'
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        model = torch.load(BytesIO(response.content), map_location=torch.device('cpu'))
        model.eval()
        return model
    else:
        raise Exception("Failed to download the model. Please check the URL or the network connection.")

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing like pre-trained models
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Adding batch dimension
    return image

# Streamlit App
st.title('Cassava Crop Disease Classifier 🌿🦠')
st.markdown("Upload an image of a cassava leaf and detect the disease!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    with st.spinner('Analyzing image... 🔍'):
        input_tensor = preprocess_image(image)
        
        time.sleep(1)  # Little delay for spinner
        
        # Loading model and making prediction
        model = load_model()  # Ensure model is loaded before inference
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
            predicted_class = class_names[predicted.item()]
        
    st.success(f"🎯 **Prediction: {predicted_class}**")
    st.info(f"🧠 **Confidence: {confidence.item()*100:.2f}%**")

    # Plot all class probabilities
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(class_names, probs.numpy(), color='royalblue')
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities')
    st.pyplot(fig)

    # Progress Bar for visualization
    progress_text = "Loading prediction confidence..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.005)  # Small delay for better progress bar visualization
        my_bar.progress(percent_complete + 1, text=progress_text)

else:
    st.warning('⚠️ Please upload an image to classify.')
