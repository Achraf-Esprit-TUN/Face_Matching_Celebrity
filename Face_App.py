import os
import cv2
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import uuid
import av
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from streamlit_webrtc import webrtc_streamer, WebRTCConfiguration

# --- Google Drive Authentication ---
def authenticate_gdrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens browser for auth
    return GoogleDrive(gauth)

# --- Storage Manager (Google Drive) ---
class StorageManager:
    def __init__(self):
        self.drive = authenticate_gdrive()
        self.user_images_folder_id = "10YxfcJDyhbTwFD4RUSTup1-5vZZrO8Kc"  # Replace with your folder ID
        self.models_folder_id = "1947SsF0d2vAo9p1N1PsionMagWA8drWA"  # Replace with your folder ID

    def upload_file(self, file_path, folder_id):
        file = self.drive.CreateFile({'parents': [{'id': folder_id}]})
        file.SetContentFile(file_path)
        file.Upload()
        return file['id']

    def download_file(self, file_id, save_path):
        file = self.drive.CreateFile({'id': file_id})
        file.GetContentFile(save_path)
        return True

    def list_files(self, folder_id):
        return self.drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()

# --- User Manager (Pickle + Google Drive) ---
class UserManager:
    def __init__(self, storage):
        self.storage = storage
        self.users_file_id = "1HpI7viug0RPGq6ywl-LViMiC222aMMoM"  # Upload a blank `users.pkl` first

    def _load_users(self):
        try:
            self.storage.download_file(self.users_file_id, "temp_users.pkl")
            with open("temp_users.pkl", "rb") as f:
                return pickle.load(f)
        except:
            return {}

    def _save_users(self, users):
        with open("temp_users.pkl", "wb") as f:
            pickle.dump(users, f)
        self.storage.upload_file("temp_users.pkl", self.models_folder_id)
        os.remove("temp_users.pkl")

    def add_user(self, username, password, image_file):
        users = self._load_users()
        if username in users:
            return False, "Username exists"
        
        user_id = str(uuid.uuid4())
        users[username] = {
            "user_id": user_id,
            "password": password,  # Insecure! Use bcrypt in production
            "images": []
        }
        self._save_users(users)
        
        # Save image to Google Drive
        img_path = f"temp_{user_id}.jpg"
        Image.open(image_file).save(img_path)
        file_id = self.storage.upload_file(img_path, self.user_images_folder_id)
        os.remove(img_path)
        
        return True, user_id

    def authenticate(self, username, password):
        users = self._load_users()
        if username in users and users[username]["password"] == password:
            return True, users[username]["user_id"]
        return False, None

# --- Model Training & Prediction ---
class FaceRecognizer:
    def __init__(self, storage):
        self.storage = storage
        self.model_file_id = "1L5z6dEw1WcGdd7UThgtJF9llp4JayasO"
        self.le_file_id = "1QjDehqvrxziI35k9aCJ3Kv-pbVf5Ae_U"
        self.label_dict_file_id = "1V7xqHmXK1EeuYKLEVn_7o2MwX-PJ6ZcM"

    def train_model(self, user_manager):
        users = user_manager._load_users()
        X, y = [], []
        label_dict = {}

        for username, data in users.items():
            user_id = data["user_id"]
            files = self.storage.list_files(self.user_images_folder_id)
            
            for file in files:
                if user_id in file["title"]:
                    self.storage.download_file(file["id"], "temp_img.jpg")
                    img = cv2.imread("temp_img.jpg")
                    
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (128, 128))
                        hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)
                        features = hog.compute(resized).flatten()
                        X.append(features)
                        y.append(user_id)
                        label_dict[user_id] = username
                    os.remove("temp_img.jpg")

        if not X:
            return False, "No data to train"

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        model = SVC(kernel="linear", probability=True)
        model.fit(X, y_encoded)

        # Save models
        with open("temp_model.pkl", "wb") as f:
            pickle.dump(model, f)
        self.storage.upload_file("temp_model.pkl", self.models_folder_id)

        with open("temp_le.pkl", "wb") as f:
            pickle.dump(le, f)
        self.storage.upload_file("temp_le.pkl", self.models_folder_id)

        with open("temp_labels.pkl", "wb") as f:
            pickle.dump(label_dict, f)
        self.storage.upload_file("temp_labels.pkl", self.models_folder_id)

        return True, f"Trained on {len(X)} images"

    def predict_face(self, frame):
        try:
            # Load models
            self.storage.download_file(self.model_file_id, "temp_model.pkl")
            self.storage.download_file(self.le_file_id, "temp_le.pkl")
            self.storage.download_file(self.label_dict_file_id, "temp_labels.pkl")
            
            with open("temp_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("temp_le.pkl", "rb") as f:
                le = pickle.load(f)
            with open("temp_labels.pkl", "rb") as f:
                label_dict = pickle.load(f)

            # Preprocess frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)
            features = hog.compute(resized).reshape(1, -1)

            # Predict
            pred = model.predict(features)
            proba = model.predict_proba(features)[0]
            user_id = le.inverse_transform(pred)[0]
            username = label_dict.get(user_id, "Unknown")

            return username, proba.max() * 100
        except:
            return "Unknown", 0

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Face Auth", layout="wide")
    storage = StorageManager()
    user_manager = UserManager(storage)
    recognizer = FaceRecognizer(storage)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # --- Login / Register Tabs ---
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                success, _ = user_manager.authenticate(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            user_image = st.file_uploader("Upload Face", type=["jpg", "png"])
            
            if st.button("Register"):
                if new_user and new_pass and user_image:
                    success, msg = user_manager.add_user(new_user, new_pass, user_image)
                    if success:
                        st.success("Registered! Please login.")
                        recognizer.train_model(user_manager)  # Retrain model
                    else:
                        st.error(msg)
                else:
                    st.error("Fill all fields")

    # --- Face Recognition (After Login) ---
    else:
        st.title("Face Recognition System")
        
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            username, confidence = recognizer.predict_face(img)
            
            cv2.putText(img, f"{username} ({confidence:.1f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="example",
            video_frame_callback=video_frame_callback,
            rtc_configuration=WebRTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

if __name__ == "__main__":
    main()