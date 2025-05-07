import cv2
import numpy as np
import pickle
import streamlit as st
import time
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

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

# Configuration sidebar
def configuration_sidebar():
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

# Load or create model
def setup_model(config, label_dict):
    try:
        # Try to load existing model
        with open('models/face_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Verify model compatibility with current config
        if (model.kernel != config['svm_params']['kernel'] or 
            model.probability != config['svm_params']['probability']):
            st.warning("Loaded model doesn't match current configuration. Creating new model.")
            raise Exception("Configuration mismatch")
            
        return model
    except:
        # Create new model with current configuration
        return SVC(**config['svm_params'])

# Feature extractor class
class FeatureExtractor:
    def __init__(self, hog_params):
        self.hog = cv2.HOGDescriptor(
            hog_params['win_size'],
            hog_params['block_size'],
            hog_params['block_stride'],
            hog_params['cell_size'],
            hog_params['nbins']
        )
        self.feature_length = self._calculate_feature_length(hog_params)
        
    def _calculate_feature_length(self, params):
        # Calculate expected feature vector length
        cells_per_block = (params['block_size'][0] // params['cell_size'][0]) ** 2
        blocks_per_window = ((params['win_size'][0] - params['block_size'][0]) // 
                           (params['block_stride'][0]) + 1) ** 2
        return cells_per_block * blocks_per_window * params['nbins']
    
    def extract(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.hog.winSize)
        features = self.hog.compute(resized)
        return features.flatten()

def main():
    # Load configuration
    config = configuration_sidebar()
    
    # Load label information
    try:
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('models/label_dict.pkl', 'rb') as f:
            label_dict = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading label data: {str(e)}")
        return
    
    # Setup model and feature extractor
    model = setup_model(config, label_dict)
    feature_extractor = FeatureExtractor(config['hog_params'])
    
    # Main interface
    st.title("Real-Time Face Matching")
    col1, col2 = st.columns([2, 1])
    run = col1.checkbox('Start Webcam', True)
    FRAME_WINDOW = col1.empty()
    results_placeholder = col2.empty()
    
    # Video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
    
    # Face detection
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
            
            # Extract face with margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            current_face = frame[y1:y2, x1:x2]
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
                    
                    # Sort results
                    sorted_indices = np.argsort(-last_prob)
                    sorted_results = [
                        (label_dict[i], last_prob[i]*100) 
                        for i in sorted_indices
                    ]
                    
                    # Update display
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