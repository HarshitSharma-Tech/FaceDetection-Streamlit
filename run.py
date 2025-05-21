from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
import logging

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

# Initialize OpenCV Face Detection
def get_face_detector():
    # Check if DNN model exists, otherwise use Haar cascades
    model_file = os.path.join(os.path.dirname(__file__), "opencv_face_detector_uint8.pb")
    config_file = os.path.join(os.path.dirname(__file__), "opencv_face_detector.pbtxt")
    
    if os.path.exists(model_file) and os.path.exists(config_file):
        logger.info("Using OpenCV DNN face detector")
        face_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        detector_type = "dnn"
    else:
        logger.info("Using Haar Cascade face detector")
        model_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_detector = cv2.CascadeClassifier(model_file)
        detector_type = "haar"
        
    return face_detector, detector_type

# Get the face detector
face_detector, detector_type = get_face_detector()

# Detect faces in image
def detect_faces(image, min_confidence=0.5):
    """
    Detect faces in an image using either Haar Cascades or DNN detector
    Returns list of face rectangles as (x, y, w, h)
    """
    if detector_type == "haar":
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    else:
        # Use DNN detector
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        face_detector.setInput(blob)
        detections = face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Convert to x, y, w, h format
                x = max(0, x1)
                y = max(0, y1)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                
                faces.append((x, y, w, h))
        
        return faces

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
                # Detect faces
                faces = detect_faces(frame)
                
                # If faces detected
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Ensure bbox is within frame boundaries
                        ih, iw, _ = frame.shape
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
                if len(faces) == 0:
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
