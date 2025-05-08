import cv2
import numpy as np
import pickle
import streamlit as st
import time
import gdown
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Google Drive file IDs
MODEL_FILE_ID = "16vjqOa4HxbPS3LcqjzXyQgZHZ7GWcfGe"
LABEL_ENCODER_FILE_ID = "1EgoI9SCkKBfbUzqmq44FtB1x7kAkP2uF"
LABEL_DICT_FILE_ID = "1ZRxuwvSQf8ErNcGURGUaC-uwN__HOETv"

# Set page config
st.set_page_config(page_title="Live Face Matching", layout="wide")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
    """Load models with version mismatch handling"""
    try:
        os.makedirs("temp_models", exist_ok=True)
        
        # Download files
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
    """Create configuration panel with all parameters from the old version"""
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
    block_stride = st.sidebar.slider("Block Stride", 4, 16, 8, step=4)
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
            'block_stride': (block_stride, block_stride),
            'cell_size': (cell_size, cell_size),
            'nbins': nbins
        },
        'processing': {
            'frame_skip': frame_skip,
            'min_face_change': min_face_change
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

class VideoProcessor:
    def __init__(self, model, le, label_dict, feature_extractor, config):
        self.model = model
        self.le = le
        self.label_dict = label_dict
        self.feature_extractor = feature_extractor
        self.config = config
        self.last_prob = None
        self.last_face = None
        self.frame_count = 0
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Skip frames based on configuration
        self.frame_count += 1
        if self.frame_count % self.config['processing']['frame_skip'] != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
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
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            current_face = img[y1:y2, x1:x2]
            current_face_resized = cv2.resize(current_face, DEFAULT_FACE_SIZE)
            
            # Only process if face changed significantly
            if (self.last_face is None or 
                cv2.norm(self.last_face, current_face_resized) > self.config['processing']['min_face_change']):
                self.last_face = current_face_resized
                
                try:
                    # Extract features and predict
                    features = self.feature_extractor.extract(current_face_resized).reshape(1, -1)
                    proba = self.model.predict_proba(features)[0]
                    self.last_prob = proba
                    
                    # Always update results regardless of confidence
                    sorted_indices = np.argsort(-proba)
                    sorted_results = [
                        (self.label_dict[i], proba[i]*100) 
                        for i in sorted_indices
                    ]
                    st.session_state['match_results'] = sorted_results
                    
                    # Debug info
                    print(f"Found matches: {len(sorted_results)}")
                    print(f"Top match: {sorted_results[0] if sorted_results else 'None'}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    print(f"Exception in prediction: {str(e)}")
            
            # Draw on frame
            if self.last_prob is not None:
                best_idx = np.argmax(self.last_prob)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{self.label_dict[best_idx]}: {self.last_prob[best_idx]*100:.1f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Load configuration and models
    config = configuration_sidebar()
    model, le, label_dict = load_models()
    if model is None:
        st.error("Failed to load models. Please check your internet connection.")
        return
    
    # Add debug info
    st.sidebar.write(f"Model loaded: {type(model).__name__}")
    st.sidebar.write(f"Number of classes: {len(label_dict)}")
    
    feature_extractor = FeatureExtractor(config['hog_params'])
    
    st.title("Live Face Matching System")
    
    # Initialize session state for results
    if 'match_results' not in st.session_state:
        st.session_state['match_results'] = []
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC streamer for live video
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(
                model, le, label_dict, feature_extractor, config
            ),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )
    
        with col2:
            # Results display container
            st.subheader("Top 5 Matches")
            
            if webrtc_ctx and webrtc_ctx.state.playing and 'match_results' in st.session_state:
                results = st.session_state.get('match_results', [])
                if results:
                    for i, (name, confidence) in enumerate(results[:5]):
                        cols = st.columns([3, 2, 5])
                        cols[0].markdown(f"**{name}**")
                        cols[1].markdown(f"{confidence:.1f}%")
                        cols[2].progress(min(100, int(confidence)), text=f"{min(100, confidence):.1f}%")
                else:
                    st.info("No faces matched yet. Please wait...")
                    
            # Auto-rerun every second instead of continuous loop
            time.sleep(1)
            
            

if __name__ == "__main__":
    main()
