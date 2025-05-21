import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
import logging
import mediapipe as mp
from typing import Optional, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - emotion classes
EMOTIONS = {
    0: 'Angry 😠', 
    1: 'Disgust 🤢', 
    2: 'Fear 😨', 
    3: 'Happy 😊',
    4: 'Sad 😢', 
    5: 'Surprise 😲', 
    6: 'Neutral 😐'
}

# Page config and styling
def setup_page():
    st.set_page_config(
        page_title="Emotion Detection",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .css-18e3th9 {
            padding-top: 1rem;
        }
        .css-1kyxreq {
            margin-top: -60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

# Cache the MediaPipe face detector
@st.cache_resource
def get_face_detector():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for close range, 1 for far range
        min_detection_confidence=0.5
    )
    return face_detection

# Cache the model loading
@st.cache_resource
def load_model() -> Tuple[Optional[torch.nn.Module], torch.device]:
    try:
        # Determine best device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
            
        # Initialize model with modern API
        model = models.resnet18(weights=None)  # No pre-trained weights needed as we'll load our own
        model.fc = torch.nn.Linear(model.fc.in_features, len(EMOTIONS))
        
        # Load trained weights
        model_path = os.path.join(os.path.dirname(__file__), 'resnet18_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            st.error(f"Model file not found at {model_path}")
            logger.error(f"Model file not found at {model_path}")
            return None, device
        
        # Move to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        error_msg = f"Model loading error: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, torch.device("cpu")

class VideoProcessor(VideoProcessorBase):
    """Video processor for real-time emotion detection."""
    
    def __init__(self):
        super().__init__()
        self.model = st.session_state["model"]
        self.transform = st.session_state["transform"]
        self.device = st.session_state["device"]
        self.face_detection = st.session_state["face_detection"]
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.processing_times = []

    @torch.no_grad()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Start processing timer
            proc_start = time.time()
            
            # Get frame and convert BGR to RGB
            img = frame.to_ndarray(format="bgr24")
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_h, frame_w, _ = img.shape
            
            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Run face detection
            results = self.face_detection.process(rgb_frame)
            
            # Add background metrics info
            cv2.putText(
                img, f"FPS: {self.fps:.1f}", (frame_w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
            
            # If faces detected
            faces_found = False
            if results.detections:
                for detection in results.detections:
                    # Extract face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h), \
                                int(bbox.width * frame_w), int(bbox.height * frame_h)
                    
                    # Ensure bbox is within frame boundaries
                    x, y = max(0, x), max(0, y)
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)
                    
                    # Draw face bounding box
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face image
                    face_img = img[y:y+h, x:x+w]
                    if face_img.size > 0:
                        # Convert to PIL for model input
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                        
                        # Predict emotion
                        output = self.model(input_tensor)
                        emotion_idx = torch.argmax(output, 1).item()
                        emotion = EMOTIONS[emotion_idx]
                        confidence = torch.nn.functional.softmax(output, dim=1)[0][emotion_idx].item()
                        
                        # Draw result
                        text = f"{emotion} ({confidence:.2f})"
                        cv2.putText(
                            img, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        faces_found = True
            
            # If no faces found, process whole frame
            if not faces_found:
                # Convert to PIL
                pil_img = Image.fromarray(rgb_frame)
                
                # Process image
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                emotion_idx = torch.argmax(output, 1).item()
                emotion = EMOTIONS[emotion_idx]
                
                # Draw result
                cv2.putText(
                    img, f"Overall: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            
            # Record processing time
            proc_time = time.time() - proc_start
            self.processing_times.append(proc_time)
            if len(self.processing_times) > 100:  # Keep only last 100 measurements
                self.processing_times.pop(0)
            
            # Display avg processing time
            avg_time = sum(self.processing_times) / len(self.processing_times) * 1000  # ms
            cv2.putText(
                img, f"Processing: {avg_time:.1f}ms", (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame

def process_image(image: Image.Image) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Process a still image for emotion detection with face detection."""
    try:
        # Convert image to numpy array for face detection
        img_array = np.array(image)
        
        # Detect faces
        face_detection = st.session_state["face_detection"]
        results = face_detection.process(img_array)
        
        # Create a copy for drawing
        output_image = img_array.copy()
        
        if results.detections:
            # Process first detected face
            detection = results.detections[0]  # Take first face
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img_array.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                int(bbox.width * w), int(bbox.height * h)
            
            # Ensure bbox is within image boundaries
            x, y = max(0, x), max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Draw rectangle on output image
            cv2.rectangle(output_image, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Extract face and process
            face_img = Image.fromarray(img_array[y:y+height, x:x+width])
            
            with torch.no_grad():
                input_tensor = st.session_state["transform"](face_img).unsqueeze(0)
                input_tensor = input_tensor.to(st.session_state["device"])
                output = st.session_state["model"](input_tensor)
                emotion_idx = torch.argmax(output, 1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][emotion_idx].item()
                
            # Draw prediction on output image
            emotion = EMOTIONS[emotion_idx]
            cv2.putText(
                output_image, f"{emotion} ({confidence:.2f})", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            return emotion, confidence, output_image
        else:
            # Process entire image if no face detected
            with torch.no_grad():
                input_tensor = st.session_state["transform"](image).unsqueeze(0)
                input_tensor = input_tensor.to(st.session_state["device"])
                output = st.session_state["model"](input_tensor)
                emotion_idx = torch.argmax(output, 1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][emotion_idx].item()
                
            # Draw prediction on output image
            emotion = EMOTIONS[emotion_idx]
            cv2.putText(
                output_image, f"Overall: {emotion} ({confidence:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            
            return emotion, confidence, output_image
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None, None, None

def about_section():
    """Show information about the project."""
    st.sidebar.markdown("## About")
    st.sidebar.info(
        """
        This app uses a ResNet18 model to detect emotions in real-time from webcam feed or uploaded images.
        
        **Detectable emotions:**
        - 😠 Angry 
        - 🤢 Disgust
        - 😨 Fear
        - 😊 Happy
        - 😢 Sad
        - 😲 Surprise
        - 😐 Neutral
        
        Made with ❤️ using PyTorch and Streamlit.
        """
    )
    
    st.sidebar.markdown("## Parameters")
    detection_confidence = st.sidebar.slider(
        "Face Detection Confidence", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5,
        step=0.1
    )
    
    # Update face detector if confidence changed
    if "detection_confidence" not in st.session_state or st.session_state["detection_confidence"] != detection_confidence:
        st.session_state["detection_confidence"] = detection_confidence
        mp_face_detection = mp.solutions.face_detection
        st.session_state["face_detection"] = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=detection_confidence
        )

def main():
    # Setup page
    setup_page()
    
    # App title
    st.title("😊 Emotion Detection")
    
    # About section in sidebar
    about_section()
    
    # Initialize session state
    if "model" not in st.session_state:
        with st.spinner("Loading model and resources..."):
            # Load model
            model, device = load_model()
            if model is None:
                st.error("Failed to load emotion detection model!")
                return
            
            # Initialize session state
            st.session_state["model"] = model
            st.session_state["device"] = device
            st.session_state["transform"] = get_transform()
            st.session_state["face_detection"] = get_face_detector()
            st.session_state["detection_confidence"] = 0.5
    
    # Mode selection tabs
    tab1, tab2 = st.tabs(["📷 Live Video", "🖼️ Image Upload"])
    
    with tab1:
        st.markdown("### Real-time Emotion Detection")
        st.info("This uses your webcam to detect emotions in real-time. Please allow camera access.")
        
        # WebRTC configuration with STUN servers
        rtc_config = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Create WebRTC streamer
        try:
            webrtc_streamer(
                key="emotion-detection",
                video_processor_factory=VideoProcessor,
                client_settings=rtc_config,
                async_processing=True,
                mode=WebRtcMode.SENDRECV
            )
        except Exception as e:
            st.error(f"Error starting video stream: {str(e)}")
    
    with tab2:
        st.markdown("### Analyze Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Create columns for display
            col1, col2 = st.columns(2)
            
            try:
                # Load and display original image
                image = Image.open(uploaded_file)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Analyze button
                if st.button("Analyze Emotion"):
                    with st.spinner("Detecting emotion..."):
                        # Process image
                        start_time = time.time()
                        emotion, confidence, result_image = process_image(image)
                        processing_time = (time.time() - start_time) * 1000  # ms
                        
                        # Show results
                        if emotion:
                            with col2:
                                st.subheader("Detected Emotion")
                                st.image(result_image, caption=f"Processed Image", use_column_width=True)
                                
                                # Results in a nice card
                                st.markdown(f"""
                                <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6;">
                                    <h3 style="margin-top: 0;">Results</h3>
                                    <p><b>Emotion:</b> {emotion}</p>
                                    <p><b>Confidence:</b> {confidence:.2f}</p>
                                    <p><b>Processing Time:</b> {processing_time:.2f} ms</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("Error processing image. Please try another image.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()