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
from collections import deque
from typing import List, Tuple, Optional

# Google Drive file IDs
MODEL_FILE_ID = "16vjqOa4HxbPS3LcqjzXyQgZHZ7GWcfGe"
LABEL_ENCODER_FILE_ID = "1EgoI9SCkKBfbUzqmq44FtB1x7kAkP2uF"
LABEL_DICT_FILE_ID = "1ZRxuwvSQf8ErNcGURGUaC-uwN__HOETv"

# Set page config
st.set_page_config(page_title="Live Face Matching", layout="wide")

# WebRTC Configuration (Updated with multiple STUN servers)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]}
    ]
})

# Constants
DEFAULT_FACE_SIZE = (160, 160)
MIN_CONFIDENCE = 0.65

@st.cache_resource(ttl=3600, show_spinner=False)
def download_from_drive(file_id: str, output_path: str) -> str:
    """Download files from Google Drive with caching and retries"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        for attempt in range(3):
            try:
                gdown.download(url, output_path, quiet=True)
                break
            except Exception as e:
                if attempt == 2:
                    st.error(f"Failed to download {output_path} after 3 attempts")
                    raise
                time.sleep(2)
    return output_path

@st.cache_resource(show_spinner="Loading models...")
def load_models() -> Tuple[Optional[SVC], Optional[LabelEncoder], Optional[dict]]:
    """Load models with robust error handling"""
    try:
        os.makedirs("temp_models", exist_ok=True)
        
        with st.spinner("Downloading model files..."):
            model_path = download_from_drive(MODEL_FILE_ID, "temp_models/face_model.pkl")
            le_path = download_from_drive(LABEL_ENCODER_FILE_ID, "temp_models/label_encoder.pkl")
            label_path = download_from_drive(LABEL_DICT_FILE_ID, "temp_models/label_dict.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        with open(label_path, 'rb') as f:
            label_dict = pickle.load(f)
            
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
            
        return model, le, label_dict
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def configuration_sidebar() -> dict:
    """Create configuration panel"""
    st.sidebar.title("Settings")
    
    with st.sidebar.expander("Processing Settings"):
        frame_skip = st.slider("Frame Skip", 1, 5, 2, 
                             help="Process every nth frame to reduce CPU load")
        min_confidence = st.slider("Confidence Threshold", 0.1, 0.99, 0.65, 0.01,
                                 help="Minimum confidence to show predictions")
        min_face_change = st.slider("Face Change Threshold", 50, 500, 150,
                                  help="Pixel difference needed to re-process face")
    
    return {
        'processing': {
            'frame_skip': frame_skip,
            'min_confidence': min_confidence,
            'min_face_change': min_face_change
        }
    }

class FeatureExtractor:
    def __init__(self):
        # HOG configuration
        self.win_size = (128, 128)
        self.block_size = (32, 32)
        self.block_stride = (16, 16)
        self.cell_size = (16, 16)
        self.nbins = 9
        
        self.hog = cv2.HOGDescriptor(
            _winSize=self.win_size,
            _blockSize=self.block_size,
            _blockStride=self.block_stride,
            _cellSize=self.cell_size,
            _nbins=self.nbins
        )
        
        self._last_face = None
        self._last_features = None
    
    def extract(self, face: np.ndarray) -> np.ndarray:
        """Extract features with memoization"""
        if self._last_face is not None and cv2.norm(face, self._last_face) < 5:
            return self._last_features
            
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.win_size)
        features = self.hog.compute(resized).flatten()
        
        self._last_face = face.copy()
        self._last_features = features
        
        return features

class VideoProcessor:
    """Video processor with face detection and recognition"""
    def __init__(self, model, le, label_dict, config):
        self.model = model
        self.label_dict = label_dict
        self.config = config
        self.feature_extractor = FeatureExtractor()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.pred_history = deque(maxlen=5)
        self.last_face = None
        self.frame_count = 0
    
    def _detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frames"""
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        if self.frame_count % self.config['processing']['frame_skip'] != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        faces = self._detect_faces(img)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            current_face = img[y1:y2, x1:x2]
            current_face_resized = cv2.resize(current_face, DEFAULT_FACE_SIZE)
            
            if (self.last_face is None or 
                cv2.norm(self.last_face, current_face_resized) > 
                self.config['processing']['min_face_change']):
                
                self.last_face = current_face_resized
                
                try:
                    features = self.feature_extractor.extract(current_face_resized)
                    proba = self.model.predict_proba(features.reshape(1, -1))[0]
                    
                    if proba.max() >= self.config['processing']['min_confidence']:
                        self.pred_history.append(proba)
                        current_pred = np.mean(self.pred_history, axis=0) if self.pred_history else proba
                        
                        sorted_indices = np.argsort(-current_pred)
                        sorted_results = [
                            (self.label_dict[i], current_pred[i]*100) 
                            for i in sorted_indices
                        ]
                        
                        st.session_state['match_results'] = sorted_results
                
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
            
            if 'match_results' in st.session_state and st.session_state['match_results']:
                best_name, best_conf = st.session_state['match_results'][0]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{best_name}: {best_conf:.1f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    """Main application"""
    if 'match_results' not in st.session_state:
        st.session_state['match_results'] = []
    
    config = configuration_sidebar()
    model, le, label_dict = load_models()
    
    if model is None or le is None or label_dict is None:
        return
    
    st.title("Live Face Matching System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(model, le, label_dict, config),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%"},
                "autoPlay": True,
                "muted": True
            }
        )
    
    with col2:
        st.subheader("Recognition Results")
        
        if not webrtc_ctx or not webrtc_ctx.state.playing:
            st.info("Starting video stream...")
            return
        
        if 'match_results' in st.session_state and st.session_state['match_results']:
            results = st.session_state['match_results'][:5]
            
            for i, (name, confidence) in enumerate(results):
                cols = st.columns([1, 3, 1])
                cols[0].markdown(f"**{i+1}.**")
                cols[1].markdown(f"**{name}**")
                cols[2].markdown(f"`{confidence:.1f}%`")
                st.progress(min(confidence/100, 1.0))

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()