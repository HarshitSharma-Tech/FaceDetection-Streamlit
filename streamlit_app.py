import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

z# Page config
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# Initialize session states
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        device = torch.device("cpu")  # Force CPU for deployment
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
        
        # Check if model file exists
        if not os.path.exists('resnet18_model.pth'):
            st.error("Model file not found! Please ensure resnet18_model.pth is in the same directory.")
            return None, device
            
        model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_emotion(image, model, device, transform):
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()
            return pred
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    st.title("😊 Emotion Detection")
    st.write("Upload an image to detect the emotion!")
    
    # Load model only once
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Sad 😢', 'Surprise 😲', 'Neutral 😐']
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and st.session_state.model_loaded:
        try:
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with col2:
                if st.button("Detect Emotion"):
                    with st.spinner("Detecting emotion..."):
                        pred = predict_emotion(image, st.session_state.model, 
                                            st.session_state.device, transform)
                        if pred is not None:
                            emotion = labels[pred]
                            st.success(f"Predicted Emotion: {emotion}")
                            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        with col2:
            st.info("Please upload an image to begin emotion detection")

if __name__ == '__main__':
    main()