import cv2
import numpy as np
import pickle
import streamlit as st
import gdown
import os
import av
from sklearn.svm import SVC
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Google Drive file IDs
MODEL_FILE_ID = "16vjqOa4HxbPS3LcqjzXyQgZHZ7GWcfGe"
LABEL_ENCODER_FILE_ID = "1EgoI9SCkKBfbUzqmq44FtB1x7kAkP2uF"
LABEL_DICT_FILE_ID = "1ZRxuwvSQf8ErNcGURGUaC-uwN__HOETv"

# Page configuration
st.set_page_config(page_title="Celebrity Lookalike Detector", layout="wide")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]
})

# Constants
DEFAULT_FACE_SIZE = (128, 128)
HOG_PARAMS = {
    'win_size': (64, 64),
    'block_size': (16, 16),
    'block_stride': (8, 8),
    'cell_size': (8, 8),
    'nbins': 9
}

@st.cache_resource
def download_and_load_models():
    """Download and load ML models with error handling"""
    try:
        os.makedirs("temp_models", exist_ok=True)
        
        # Download files
        model_path = gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", 
                                  "temp_models/face_model.pkl", quiet=False)
        label_path = gdown.download(f"https://drive.google.com/uc?id={LABEL_DICT_FILE_ID}", 
                                  "temp_models/label_dict.pkl", quiet=False)

        # Load models
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(label_path, 'rb') as f:
            label_dict = pickle.load(f)

        # Clean label names (remove 'pins_' prefix)
        cleaned_labels = {}
        for key, value in label_dict.items():
            clean_name = value.replace("pins_", "").replace("_", " ").title()
            cleaned_labels[key] = clean_name
            
        return model, cleaned_labels
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

class FeatureExtractor:
    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            HOG_PARAMS['win_size'],
            HOG_PARAMS['block_size'],
            HOG_PARAMS['block_stride'],
            HOG_PARAMS['cell_size'],
            HOG_PARAMS['nbins']
        )

    def extract(self, face_img):
        """Extract HOG features from face image"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, HOG_PARAMS['win_size'])
            return self.hog.compute(resized).flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

class VideoProcessor:
    def __init__(self, model, label_dict):
        self.model = model
        self.label_dict = label_dict
        self.feature_extractor = FeatureExtractor()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_predictions = []
        self.frame_count = 0
        self.prediction_updated = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Skip every other frame for performance
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Get largest face
            
            # Extract face region with margin
            margin = int(min(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            face_roi = img[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # Resize and extract features
                face_resized = cv2.resize(face_roi, DEFAULT_FACE_SIZE)
                features = self.feature_extractor.extract(face_resized)
                
                if features is not None:
                    # Make prediction
                    proba = self.model.predict_proba(features.reshape(1, -1))[0]
                    sorted_indices = np.argsort(-proba)
                    predictions = [
                        (self.label_dict[i], proba[i]*100) 
                        for i in sorted_indices[:5]  # Top 5 predictions
                    ]
                    
                    # Update session state directly
                    st.session_state.predictions = predictions
                    
                    # Draw face rectangle and top prediction
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if predictions:
                        top_name, top_conf = predictions[0]
                        cv2.putText(img, f"{top_name}: {top_conf:.1f}%",
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    # Load models
    model, label_dict = download_and_load_models()
    if model is None or label_dict is None:
        return

    # App layout
    st.title("Celebrity Lookalike Analyzer")
    st.markdown("### Find out which celebrity you look like!")

    col1, col2 = st.columns([2, 1])

    with col1:
        # WebRTC video stream
        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(model, label_dict),
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
        )

    with col2:
        st.markdown("### Top Matches")
        results_placeholder = st.empty()
        
        # Get predictions from session state
        predictions = st.session_state.predictions
        
        # Add debug info
        st.write(f"Face detected: {len(predictions) > 0}")
        
        if len(predictions) > 0:
            with results_placeholder.container():
                for i, (name, confidence) in enumerate(predictions):
                    st.markdown(f"**{i+1}. {name}**")
                    st.progress(confidence/100)
                    st.markdown(f"`{confidence:.1f}% Similarity`")
                    st.write("---")
        else:
            results_placeholder.warning("Align your face in the camera...")

    # Footer
    st.markdown("---")
    st.markdown("*Using Haar cascades for face detection and HOG+SVM for recognition*")

if __name__ == "__main__":
    main()
