import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMOTIONS = {
    0: 'Angry 😠', 
    1: 'Disgust 🤢', 
    2: 'Fear 😨', 
    3: 'Happy 😊',
    4: 'Sad 😢', 
    5: 'Surprise 😲', 
    6: 'Neutral 😐'
}

# Cache the transform
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Cache the model loading
@st.cache_resource
def load_model() -> tuple[Optional[torch.nn.Module], torch.device]:
    try:
        device = torch.device("cpu")  # Force CPU for consistency
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = torch.nn.Linear(model.fc.in_features, len(EMOTIONS))
        
        if os.path.exists('resnet50_model.pth'):
            model.load_state_dict(
                torch.load('resnet50_model.pth', map_location=device)
            )
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None, torch.device("cpu")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = st.session_state["model"]
        self.transform = st.session_state["transform"]
        self.device = st.session_state["device"]

    @torch.no_grad()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            # Process image
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            emotion = EMOTIONS[torch.argmax(output, 1).item()]
            
            # Draw result
            cv2.putText(
                img, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

def process_image(image: Image.Image) -> Optional[str]:
    try:
        with torch.no_grad():
            input_tensor = st.session_state["transform"](image).unsqueeze(0)
            input_tensor = input_tensor.to(st.session_state["device"])
            output = st.session_state["model"](input_tensor)
            return EMOTIONS[torch.argmax(output, 1).item()]
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

def main():
    st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")
    st.title("😊 Emotion Detection")

    # Initialize session state
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            model, device = load_model()
            if model is None:
                st.error("Failed to load model!")
                return
            
            st.session_state["model"] = model
            st.session_state["device"] = device
            st.session_state["transform"] = get_transform()
    
    # Mode selection
    mode = st.radio("Select Input Mode:", ("Image Upload", "Live Video Capture"))
    
    if mode == "Image Upload":
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    
                    if st.button("Detect Emotion"):
                        with st.spinner("Detecting..."):
                            emotion = process_image(image)
                            if emotion:
                                with col2:
                                    st.success(f"Predicted Emotion: {emotion}")
                            else:
                                st.error("Error processing image")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.write("Live video capture mode enabled. Please allow camera access.")
        try:
            webrtc_streamer(
                key="emotion-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        except Exception as e:
            st.error(f"Error starting video stream: {str(e)}")

if __name__ == '__main__':
    main()