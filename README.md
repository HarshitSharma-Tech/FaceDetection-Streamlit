# Emotion Detection

A real-time emotion detection application built with PyTorch, MediaPipe, and Streamlit/Flask.

![Emotion Detection Demo](https://github.com/username/FaceDetection/raw/main/static/demo.gif)

## Features

- **Real-time emotion detection** via webcam
- **Image upload** for static analysis
- **Face detection** using MediaPipe (more accurate than Haar Cascades)
- **Confidence scores** for each emotion prediction
- **Performance metrics** (FPS, processing time)
- **Responsive UI** with both Streamlit and Flask interfaces
- **Multiple deployment options** (Streamlit Cloud or Flask server)
- **Detects 7 emotions**: 
  - 😠 Angry 
  - 🤢 Disgust 
  - 😨 Fear 
  - 😊 Happy 
  - 😢 Sad 
  - 😲 Surprise 
  - 😐 Neutral

## Setup

### Prerequisites

- Python 3.9+
- Pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/FaceDetection.git
   cd FaceDetection
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `resnet18_model.pth` is in the root directory (pre-trained model for emotion detection)

## Running the Application

### Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

This will launch the application on `http://localhost:8501`

### Flask Web Server

```bash
python run.py
```

The Flask server will start on `http://localhost:5000`

### Standalone Script

To run emotion detection directly from your webcam without a web interface:

```bash
python resnet.py
```

Press 'q' to quit the webcam feed.

## Project Structure

- `streamlit_app.py` - Streamlit web application
- `run.py` - Flask web server
- `resnet.py` - Standalone script and core emotion detection class
- `resnet18_model.pth` - Pre-trained ResNet18 model weights
- `requirements.txt` - Python dependencies
- `static/` - Static assets for Flask app
- `templates/` - HTML templates for Flask app

## Model Architecture

The emotion detection uses a fine-tuned ResNet18 model pre-trained on ImageNet and then trained on facial emotion datasets. The model architecture:

- Base: ResNet18
- Final layer: 7 output classes (emotions)
- Input size: 224x224 RGB images

## Performance Optimization

- Facial detection using MediaPipe
- Model inference optimization with PyTorch
- Processing only detected face regions

## License

MIT

## Acknowledgements

- PyTorch team for the deep learning framework
- Streamlit for the web app framework
- MediaPipe team for the face detection model
