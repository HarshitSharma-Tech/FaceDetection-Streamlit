from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os

app = Flask(__name__)

# Load model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 7)  # 7 classes
model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
model = model.to(device)  # Move model to the right device
model.eval()  # Set to evaluation mode

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Emotion labels (adjust to your dataset)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def gen_frames():
    # Try different camera indices if 0 doesn't work
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}, trying index 1...")
        cap = cv2.VideoCapture(1)  # Try camera index 1
        if not cap.isOpened():
            print("Could not open any camera")
            return
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        try:
            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, 1).item()
                emotion = labels[pred]

            # Display
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Use environment variable for production
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
