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

# Google Drive file IDs
MODEL_FILE_ID = "16vjqOa4HxbPS3LcqjzXyQgZHZ7GWcfGe"
LABEL_ENCODER_FILE_ID = "1EgoI9SCkKBfbUzqmq44FtB1x7kAkP2uF"
LABEL_DICT_FILE_ID = "1ZRxuwvSQf8ErNcGURGUaC-uwN__HOETv"

# Set page config
st.set_page_config(page_title="Live Face Matching", layout="wide")

# WebRTC Configuration with fallback TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        {
            "urls": "turn:global.turn.twilio.com:3478?transport=udp",
            "username": "YOUR_TURN_USERNAME",
            "credential": "YOUR_TURN_CREDENTIAL"
        }
    ],
    "iceTransportPolicy": "all"  # Try both relay and host candidates
})

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
        
        model_path = download_from_drive(MODEL_FILE_ID, "temp_models/face_model.pkl")
        le_path = download_from_drive(LABEL_ENCODER_FILE_ID, "temp_models/label_encoder.pkl")
        label_path = download_from_drive(LABEL_DICT_FILE_ID, "temp_models/label_dict.pkl")
        
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        
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
    """Create configuration panel with optimized defaults"""
    st.sidebar.title("Processing Settings")
    
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 1,
                                  help="Process every nth frame to reduce CPU load")
    min_face_change = st.sidebar.slider("Face Change Threshold", 50, 500, 150,
                                       help="Pixel difference needed to re-process face")
    
    return {
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
        self._last_face = None
        self._last_features = None
        
    def extract(self, face):
        """Extract features with memoization"""
        if self._last_face is not None and cv2.norm(face, self._last_face) < 50:
            return self._last_features
            
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.hog.winSize)
            features = self.hog.compute(resized).flatten()
            
            self._last_face = face.copy()
            self._last_features = features
            return features
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

class VideoProcessor:
    def __init__(self, model, le, label_dict, feature_extractor, config):
        self.model = model
        self.label_dict = label_dict
        self.feature_extractor = feature_extractor
        self.config = config
        self.last_prob = None
        self.last_face = None
        self.frame_count = 0
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.pred_history = deque(maxlen=3)
    
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            self.frame_count += 1
            if self.frame_count % self.config['processing']['frame_skip'] != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                margin = int(min(w, h) * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                current_face = img[y1:y2, x1:x2]
                
                if current_face.size == 0:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                current_face_resized = cv2.resize(current_face, DEFAULT_FACE_SIZE)
                
                if (self.last_face is None or 
                    cv2.norm(self.last_face, current_face_resized) > self.config['processing']['min_face_change']):
                    
                    self.last_face = current_face_resized.copy()
                    
                    try:
                        features = self.feature_extractor.extract(current_face_resized)
                        if features is not None:
                            proba = self.model.predict_proba(features.reshape(1, -1))[0]
                            self.pred_history.append(proba)
                            
                            current_pred = np.mean(self.pred_history, axis=0)
                            
                            sorted_indices = np.argsort(-current_pred)
                            sorted_results = [
                                (self.label_dict[i], current_pred[i]*100) 
                                for i in sorted_indices
                            ]
                            st.session_state['match_results'] = sorted_results
                            
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")
                
                if self.pred_history:
                    best_idx = np.argmax(current_pred)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{self.label_dict[best_idx]}: {current_pred[best_idx]*100:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    config = configuration_sidebar()
    model, le, label_dict = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check your internet connection.")
        return
    
    feature_extractor = FeatureExtractor(DEFAULT_HOG_PARAMS)
    
    st.title("Live Face Matching System")
    
    with st.sidebar.expander("System Info"):
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Number of classes: {len(label_dict)}")
    
    if 'match_results' not in st.session_state:
        st.session_state['match_results'] = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(
                model, le, label_dict, feature_extractor, config
            ),
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
            video_html_attrs={
                "autoPlay": True,
                "muted": True,
                "playsinline": True
            }
        )
    
    with col2:
        st.subheader("Recognition Results")
        
        if not webrtc_ctx or not webrtc_ctx.state.playing:
            st.info("Starting video stream...")
            return
        
        if 'match_results' in st.session_state:
            results = st.session_state['match_results']
            
            if results:
                for i, (name, confidence) in enumerate(results[:5]):
                    cols = st.columns([2, 1, 2])
                    cols[0].markdown(f"**{i+1}. {name}**")
                    cols[1].markdown(f"`{confidence:.1f}%`")
                    cols[2].progress(confidence / 100)
            else:
                st.warning("No faces detected")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Add async loop handling
    import asyncio
    try:
        asyncio.get_event_loop().run_until_complete(main())
    except RuntimeError:
        asyncio.new_event_loop().run_until_complete(main())