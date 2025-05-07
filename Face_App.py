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
    """Download files from Google Drive with retries and version handling"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_resource
def load_models():
    """Load models with version mismatch handling"""
    try:
        os.makedirs("temp_models", exist_ok=True)
        
        # Download files with updated sklearn
        model_path = download_from_drive(MODEL_FILE_ID, "temp_models/face_model.pkl")
        le_path = download_from_drive(LABEL_ENCODER_FILE_ID, "temp_models/label_encoder.pkl")
        label_path = download_from_drive(LABEL_DICT_FILE_ID, "temp_models/label_dict.pkl")
        
        # Suppress version warnings
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        
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
    """Create configuration panel"""
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("SVM Parameters")
    kernel = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly'], index=0)
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    
    st.sidebar.subheader("Feature Extraction")
    win_size = st.sidebar.slider("Window Size", 32, 128, 64, step=16)
    block_size = st.sidebar.slider("Block Size", 8, 32, 16, step=8)
    
    st.sidebar.subheader("Processing")
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)
    
    return {
        'svm_params': {'kernel': kernel, 'C': C, 'probability': True},
        'hog_params': {
            'win_size': (win_size, win_size),
            'block_size': (block_size, block_size),
            'block_stride': (block_size//2, block_size//2),
            'cell_size': (8, 8),
            'nbins': 9
        },
        'processing': {
            'frame_skip': frame_skip,
            'min_face_change': 500
        }
    }

class FeatureExtractor:
    def __init__(self, hog_params):
        self.hog = cv2.HOGDescriptor(
            hog_params['win_size'],
            hog_params['block_size'],
            hog_params['block_stride'],
            hog_params['cell_size'],
            hog_params['nbins']
        )
    
    def extract(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.hog.winSize)
        return self.hog.compute(resized).flatten()

def main():
    # Load configuration and models
    config = configuration_sidebar()
    model, le, label_dict = load_models()
    if model is None:
        return
    
    feature_extractor = FeatureExtractor(config['hog_params'])
    
    st.title("Face Matching System")
    run = st.checkbox('Start Processing', True)
    
    # For Streamlit Cloud, we'll use file upload instead of webcam
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if run and uploaded_file is not None:
        try:
            # Read and process uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, DEFAULT_FACE_SIZE)
                
                # Extract features and predict
                features = feature_extractor.extract(face_resized).reshape(1, -1)
                proba = model.predict_proba(features)[0]
                
                # Display results
                st.image(img_rgb, caption='Uploaded Image', use_column_width=True)
                
                # Show top matches
                st.subheader("Match Percentages")
                sorted_indices = np.argsort(-proba)
                for i in sorted_indices[:5]:
                    st.progress(
                        int(proba[i]*100),
                        text=f"{label_dict[i]}: {proba[i]*100:.1f}%"
                    )
            else:
                st.warning("No faces detected in the image")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()