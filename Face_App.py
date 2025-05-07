import cv2
import numpy as np
import pickle
import streamlit as st
import time
import gdown
import os
from collections import deque
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Google Drive file IDs (replace with your actual file IDs)
MODEL_FILE_ID = "16vjqOa4HxbPS3LcqjzXyQgZHZ7GWcfGe"
LABEL_ENCODER_FILE_ID = "1EgoI9SCkKBfbUzqmq44FtB1x7kAkP2uF"
LABEL_DICT_FILE_ID = "1ZRxuwvSQf8ErNcGURGUaC-uwN__HOETv"

# Set page config
st.set_page_config(page_title="Face Matching System", layout="wide")

# Constants
DEFAULT_FACE_SIZE = (128, 128)
DEFAULT_HOG_PARAMS = {
    'win_size': (64, 64),
    'block_size': (16, 16),
    'block_stride': (8, 8),
    'cell_size': (8, 8),
    'nbins': 9
}

@st.cache_resource
def download_from_drive(file_id, output_path):
    """Download files from Google Drive with caching"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_resource
def load_models():
    """Load models from Google Drive with caching"""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp_models", exist_ok=True)
        
        # Download files
        model_path = download_from_drive(MODEL_FILE_ID, "temp_models/face_model.pkl")
        le_path = download_from_drive(LABEL_ENCODER_FILE_ID, "temp_models/label_encoder.pkl")
        label_path = download_from_drive(LABEL_DICT_FILE_ID, "temp_models/label_dict.pkl")
        
        # Load files
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        with open(label_path, 'rb') as f:
            label_dict = pickle.load(f)
            
        return model, le, label_dict
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def configuration_sidebar():
    """Create configuration sidebar"""
    st.sidebar.title("Model Configuration")
    
    # Model parameters
    st.sidebar.subheader("SVM Parameters")
    kernel = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly'], index=0)
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    probability = st.sidebar.checkbox("Enable probabilities", True)
    
    # Feature extraction
    st.sidebar.subheader("Feature Extraction")
    win_w = st.sidebar.slider("Window Width", 32, 128, 64, step=16)
    win_h = st.sidebar.slider("Window Height", 32, 128, 64, step=16)
    block_size = st.sidebar.slider("Block Size", 8, 32, 16, step=8)
    cell_size = st.sidebar.slider("Cell Size", 4, 16, 8, step=4)
    nbins = st.sidebar.slider("Number of Bins", 5, 12, 9)
    
    # Processing
    st.sidebar.subheader("Processing")
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)
    min_face_change = st.sidebar.slider("Min Face Change Threshold", 100, 1000, 500)
    
    return {
        'svm_params': {'kernel': kernel, 'C': C, 'probability': probability},
        'hog_params': {
            'win_size': (win_w, win_h),
            'block_size': (block_size, block_size),
            'block_stride': (block_size//2, block_size//2),
            'cell_size': (cell_size, cell_size),
            'nbins': nbins
        },
        'processing': {
            'frame_skip': frame_skip,
            'min_face_change': min_face_change
        }
    }

class FeatureExtractor:
    """Optimized feature extractor with configurable HOG parameters"""
    def __init__(self, hog_params):
        self.hog = cv2.HOGDescriptor(
            hog_params['win_size'],
            hog_params['block_size'],
            hog_params['block_stride'],
            hog_params['cell_size'],
            hog_params['nbins']
        )
        
    def extract(self, face):
        """Extract features from face image"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.hog.winSize)
        features = self.hog.compute(resized)
        return features.flatten()

def main():
    """Main application function"""
    # Load configuration
    config = configuration_sidebar()
    
    # Load models from Google Drive
    model, le, label_dict = load_models()
    if model is None:
        return
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config['hog_params'])
    
    # Main interface
    st.title("Real-Time Face Matching")
    col1, col2 = st.columns([2, 1])
    run = col1.checkbox('Start Webcam', True)
    FRAME_WINDOW = col1.empty()
    results_placeholder = col2.empty()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # State variables
    last_prob = np.zeros(len(label_dict))
    last_face = None
    frame_count = 0
    
    while run:
        start_time = time.time()
        
        # Skip frames based on configuration
        frame_count += 1
        if frame_count % config['processing']['frame_skip'] != 0:
            cap.grab()
            continue
            
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Convert to RGB for display
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            
            # Extract face with margin (with boundary checks)
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            current_face = frame[y1:y2, x1:x2]
            
            # Resize to consistent dimensions
            current_face_resized = cv2.resize(current_face, DEFAULT_FACE_SIZE)
            
            # Only process if face changed significantly
            if (last_face is None or 
                cv2.norm(last_face, current_face_resized) > config['processing']['min_face_change']):
                last_face = current_face_resized
                
                try:
                    # Extract features and predict
                    features = feature_extractor.extract(current_face_resized).reshape(1, -1)
                    proba = model.predict_proba(features)[0]
                    np.copyto(last_prob, proba)
                    
                    # Sort results from highest to lowest
                    sorted_indices = np.argsort(-last_prob)
                    sorted_results = [
                        (label_dict[i], last_prob[i]*100) 
                        for i in sorted_indices
                    ]
                    
                    # Update display with top 5 matches
                    with results_placeholder.container():
                        st.subheader("Match Percentages")
                        for name, confidence in sorted_results[:5]:
                            cols = st.columns([3, 2, 5])
                            cols[0].markdown(f"**{name}**")
                            cols[1].markdown(f"{confidence:.1f}%")
                            cols[2].progress(
                                min(100, int(confidence)),
                                text=f"{min(100, confidence):.1f}%"
                            )
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            
            # Draw on frame
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            best_idx = np.argmax(last_prob)
            cv2.putText(
                frame_display,
                f"{label_dict[best_idx]}: {last_prob[best_idx]*100:.1f}%",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Display frame
        FRAME_WINDOW.image(frame_display)
        
        # Control frame rate
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.05 - elapsed))
    
    cap.release()

if __name__ == "__main__":
    main()