import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Page config
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
        # Safely handle device selection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Use updated ResNet50 for improved accuracy
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
        
        # Check if fine-tuned model exists
        if not os.path.exists('resnet50_model.pth'):
            st.warning("Fine-tuned weights for ResNet50 not found! Using ImageNet pretrained weights. "
                       "For higher accuracy, fine-tune the model on your emotion dataset and save as 'resnet50_model.pth'.")
        else:
            model.load_state_dict(torch.load('resnet50_model.pth', map_location=device))
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

# Video processor for live capture using streamlit-webrtc
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, model, device):
        # Initialize image transform and labels
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Sad 😢', 'Surprise 😲', 'Neutral 😐']
        self.model = model
        self.device = device

    def recv(self, frame):
        # Convert the incoming frame to a numpy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB for PIL processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # Predict emotion
        pred = predict_emotion(pil_img, self.model, self.device, self.transform)
        emotion = self.labels[pred] if pred is not None else "N/A"
        # Overlay predicted emotion on the frame
        cv2.putText(img, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("😊 Emotion Detection")
    st.write("Select an input mode:")
    
    # Load model only once
    if not st.session_state.get("model_loaded", False):
        with st.spinner("Loading model..."):
            model, device = load_model()
            if model is None:
                st.error("Could not load model!")
            else:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
    
    mode = st.radio("Input Mode", ("Image Upload", "Live Video Capture"))
    
    if mode == "Image Upload":
        st.write("Upload an image to detect the emotion!")
        # Image transform (same as in live mode)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
                
    else:  # Live Video Capture mode
        st.write("Live video capture mode enabled. Please allow access to your camera.")
        # Ensure the model is loaded before starting the video stream
        if st.session_state.get("model_loaded", False):
            webrtc_streamer(
                key="emotion-detection",
                video_processor_factory=lambda: EmotionVideoProcessor(
                    st.session_state.model,
                    st.session_state.device)
            )
        else:
            st.error("Model not loaded yet. Please wait.")
    
if __name__ == '__main__':
    main()