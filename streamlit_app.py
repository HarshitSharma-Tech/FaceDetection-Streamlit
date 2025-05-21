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

# Initialize session states at the very beginning
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'transform' not in st.session_state:
    st.session_state.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
        
        # Use ResNet50 with pretrained weights
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
        
        # Check if fine-tuned model exists
        if os.path.exists('resnet50_model.pth'):
            model.load_state_dict(torch.load('resnet50_model.pth', map_location=device))
        else:
            st.warning("Fine-tuned weights not found. Using pretrained ImageNet weights.")
            
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Video processor for live capture
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Get model and device from session state
        self.model = st.session_state.model
        self.device = st.session_state.device
        self.transform = st.session_state.transform
        self.labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 
                      'Sad 😢', 'Surprise 😲', 'Neutral 😐']

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        try:
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output, 1).item()
                emotion = self.labels[pred]
                
            cv2.putText(img, f"Emotion: {emotion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("😊 Emotion Detection")
    
    # Load model only once at startup
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model!")
                return

    # Input mode selection
    mode = st.radio("Select Input Mode:", ("Image Upload", "Live Video Capture"))
    
    if mode == "Image Upload":
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", 
                                           type=["jpg", "jpeg", "png"])
            
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                if st.button("Detect Emotion"):
                    with st.spinner("Detecting..."):
                        input_tensor = st.session_state.transform(image).unsqueeze(0)
                        input_tensor = input_tensor.to(st.session_state.device)
                        
                        with torch.no_grad():
                            output = st.session_state.model(input_tensor)
                            pred = torch.argmax(output, 1).item()
                            emotion = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 
                                     'Happy 😊', 'Sad 😢', 'Surprise 😲', 
                                     'Neutral 😐'][pred]
                            
                        with col2:
                            st.success(f"Predicted Emotion: {emotion}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        st.write("Live video capture mode enabled. Please allow camera access.")
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )

if __name__ == '__main__':
    main()