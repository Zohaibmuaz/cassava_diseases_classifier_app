import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import time
import pandas as pd

# Define your custom ResNet50 model architecture
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet50 model
        resnet50 = models.resnet50(pretrained=True)
        # Replace the fully connected layer (fc) with custom layers
        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])  # Removing original FC layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)
        return x

# Load the model architecture and weights
@st.cache_resource
def load_model():
    model = CustomResNet50()  # Ensure to use the same architecture
    # Load the saved weights (model_state_dict.pth)
    model.load_state_dict(torch.load("model_state_dict.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
model = load_model()

# Define the class names for prediction
class_names = [
    'cassava-bacterial-blight-cbb',
    'cassava-brown-streak-disease-cbsd',
    'cassava-green-mottle-cgm',
    'cassava-healthy',
    'cassava-mosaic-disease-cmd'
]

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Streamlit app layout
st.title('Cassava Crop Disease Classifier 🌿🦠')
st.caption("Model: Your trained PyTorch model")
st.markdown("Upload an image of a cassava leaf and detect the disease!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    with st.spinner('Analyzing image... 🔍'):
        input_tensor = preprocess_image(image)
        time.sleep(1)  # Small delay for better UX
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
            predicted_class = class_names[predicted.item()]

    # Show prediction and confidence
    st.success(f"🎯 **Prediction: {predicted_class}**")
    st.info(f"🧠 **Confidence: {confidence.item()*100:.2f}%**")

    # Display all class probabilities
    st.subheader("Prediction Probabilities 📊")
    prob_data = pd.DataFrame({
        'Disease': class_names,
        'Probability': probs.numpy()
    }).sort_values(by='Probability', ascending=True)

    st.bar_chart(prob_data.set_index('Disease'))

    st.balloons()

else:
    st.warning('⚠️ Please upload an image to classify.')
