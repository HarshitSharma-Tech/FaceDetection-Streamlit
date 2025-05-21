import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def main():
    st.title("Emotion Detection")
    
    model, device = load_model()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Emotion labels
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()
            emotion = labels[pred]
            
        st.write(f"Predicted Emotion: {emotion}")

if __name__ == '__main__':
    main()