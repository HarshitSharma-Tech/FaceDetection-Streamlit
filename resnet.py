import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
import argparse
import logging
import mediapipe as mp
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define the class labels in the same order as the model was trained
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self, model_path=None, device=None):
        """Initialize the emotion detector with model and face detection."""
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS device")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA device")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model(model_path or 'resnet18_model.pth')
        
        # Set up MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close range, 1 for far range
            min_detection_confidence=0.5
        )
        
        # Prepare image transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path):
        """Load the ResNet model with proper error handling."""
        try:
            # Initialize model with modern API
            model = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_labels))
            
            # Load fine-tuned weights
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Fine-tuned weights not found at {model_path}. Using ImageNet pretrained weights.")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect_emotion(self, frame):
        """Detect faces and emotions in a frame."""
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return frame
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
            
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        # Process detected faces
        if results.detections:
            for detection in results.detections:
                # Extract face bounding box
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height), \
                            int(bbox.width * frame_width), int(bbox.height * frame_height)
                
                # Ensure bbox is within frame boundaries
                x, y = max(0, x), max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                # Process face for emotion detection
                face_img = frame[y:y+h, x:x+w]
                if face_img.size > 0:
                    try:
                        # Convert to PIL and apply transform
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                        
                        # Predict emotion
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            pred_idx = torch.argmax(output, 1).item()
                            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred_idx].item()
                            emotion = class_labels[pred_idx]
                        
                        # Draw rectangle and emotion label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{emotion} ({confidence:.2f})"
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        logger.error(f"Error processing face: {str(e)}")
        else:
            # If no faces detected, analyze entire frame
            try:
                # Convert to PIL and apply transform
                img_pil = Image.fromarray(rgb_frame)
                input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                
                # Predict emotion
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred_idx = torch.argmax(output, 1).item()
                    emotion = class_labels[pred_idx]
                
                # Display overall emotion
                cv2.putText(frame, f"Overall mood: {emotion}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                
        return frame
    
    def run_webcam(self):
        """Run emotion detection on webcam feed."""
        # Try multiple camera indices
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
            return
        
        fps_start_time = time.time()
        frame_count = 0
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Process frame
                result_frame = self.detect_emotion(frame)
                
                # Display FPS
                cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                           (result_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display result
                cv2.imshow('Emotion Detection', result_frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logger.error(f"Error in webcam loop: {str(e)}")
        finally:
            # Cleanup
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera resources released")

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection')
    parser.add_argument('--model', type=str, default='resnet18_model.pth',
                        help='Path to the model weights file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    device = torch.device("cpu") if args.cpu else None
    
    # Create detector and run
    detector = EmotionDetector(model_path=args.model, device=device)
    logger.info("Starting emotion detection...")
    detector.run_webcam()
