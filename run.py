from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
import logging
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model - use context manager for better resource handling
def load_emotion_model():
    try:
        # Determine device
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
        model.fc = torch.nn.Linear(model.fc.in_features, 7)  # 7 emotion classes
        
        # Load trained weights with better error handling
        model_path = os.path.join(os.path.dirname(__file__), 'resnet18_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load models at startup
model, device = load_emotion_model()

# Initialize MediaPipe Face Detection - more robust than Haar Cascades
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for close range, 1 for far range
    min_detection_confidence=0.5
)

# Image transform - modern PyTorch practice
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Emotion labels - consistent with other files
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def gen_frames():
    # Camera setup with better error handling
    camera_indices = [0, 1]  # Try these camera indices
    cap = None
    
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            logger.info(f"Successfully opened camera with index {idx}")
            break
        logger.warning(f"Failed to open camera with index {idx}")
    
    if cap is None or not cap.isOpened():
        logger.error("Could not open any camera")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               cv2.imencode('.jpg', 
                          np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + 
               b'\r\n')
        return
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to grab frame")
                break

            try:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection with MediaPipe
                results = face_detection.process(rgb_frame)
                
                # If faces detected
                if results.detections:
                    for detection in results.detections:
                        # Extract face bounding box
                        bbox = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                    int(bbox.width * iw), int(bbox.height * ih)
                        
                        # Ensure bbox is within frame boundaries
                        x, y = max(0, x), max(0, y)
                        w = min(w, iw - x)
                        h = min(h, ih - y)
                        
                        # Draw face bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Process face and predict emotion
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size > 0:  # Check if face crop is valid
                            # Preprocess
                            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
                            
                            # Predict with error handling
                            with torch.no_grad():
                                output = model(input_tensor)
                                pred = torch.argmax(output, 1).item()
                                confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item()
                                emotion = labels[pred]
                            
                            # Add text with prediction and confidence
                            text = f"{emotion} ({confidence:.2f})"
                            cv2.putText(frame, text, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # If no faces detected, still process the whole frame
                if not results.detections:
                    # Process the entire image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = torch.argmax(output, 1).item()
                        emotion = labels[pred]
                    
                    # Display overall prediction
                    cv2.putText(frame, f'Overall: {emotion}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert frame to bytes for streaming
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                      
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                # Return the original frame if processing fails
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                      
    finally:
        # Ensure camera is released
        if cap is not None:
            cap.release()
        logger.info("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    # Use environment variable for production port
    port = int(os.environ.get('PORT', 5000))
    
    # More secure settings for production
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
