import os
import torch
import gdown
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import time
from io import BytesIO

# Model Path
model_path = "model_trained.pth"

# Check if model exists locally, if not, download it
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1Euawh26Mb8RCH2AtSQvoA3rQwUalO7MB"
    gdown.download(url, model_path, quiet=False)

# Classes
class_names = [
    'cassava-bacterial-blight-cbb',
    'cassava-brown-streak-disease-cbsd',
    'cassava-green-mottle-cgm',
    'cassava-healthy',
    'cassava-mosaic-disease-cmd'
]

# Function to load the model
@st.cache_resource
def load_model():
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Load the model
model = load_model()

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 for the model
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Pretrained model normalization
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
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
        
        time.sleep(1)  # Spinner ke liye halka delay
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
            predicted_class = class_names[predicted.item()]
        
    st.success(f"🎯 **Prediction: {predicted_class}**")
    st.info(f"🧠 **Confidence: {confidence.item()*100:.2f}%**")

    # Progress Bar
    progress_text = "Loading prediction confidence..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)

    st.balloons()  # 🎈

else:
    st.warning('Please upload an image to classify.')
