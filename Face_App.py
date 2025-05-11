import os
# Set Keras environment variables BEFORE any other imports
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_KERAS'] = '1'

# Now import other libraries
import numpy as np
import streamlit as st
from PIL import Image
import pickle
from datetime import datetime
import tempfile
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# Then import DeepFace
from deepface import DeepFace

# Load environment variables
load_dotenv()

# Initialize Supabase
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Create directories if they don't exist
os.makedirs('registered_faces', exist_ok=True)
os.makedirs('face_encodings', exist_ok=True)

# Load or initialize face encodings database
def load_face_encodings():
    try:
        with open('face_encodings/encodings.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {'encodings': [], 'labels': []}

def save_face_encodings(encodings_db):
    with open('face_encodings/encodings.pkl', 'wb') as f:
        pickle.dump(encodings_db, f)

# Upload image to Supabase Storage
def upload_to_supabase(file_path, user_id):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    file_name = f"{user_id}.jpg"
    res = supabase.storage.from_("profile-pics").upload(file=file_data, path=file_name, file_options={"content-type": "image/jpeg"})
    
    # Get public URL
    url = supabase.storage.from_("profile-pics").get_public_url(file_name)
    return url

# Register new user
def register_user(username, email, image):
    # Save image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(temp_file.name)
    
    # Generate unique user ID
    user_id = str(uuid.uuid4())
    
    # Upload to Supabase
    try:
        image_url = upload_to_supabase(temp_file.name, user_id)
    except Exception as e:
        os.unlink(temp_file.name)
        return False, f"Upload failed: {str(e)}"
    
    try:
        # Face detection and encoding with DeepFace
        img = np.array(Image.open(temp_file.name))
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name='Facenet',
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if not embedding_objs:
            os.unlink(temp_file.name)
            return False, "No face detected in the image. Please upload a clear face photo."
            
        face_enc = embedding_objs[0]['embedding']
        
        # Update face encodings database
        encodings_db = load_face_encodings()
        encodings_db['encodings'].append(face_enc)
        encodings_db['labels'].append(user_id)
        save_face_encodings(encodings_db)
        
        # Save user data to Supabase
        user_data = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'image_url': image_url,
            'registration_date': datetime.now().isoformat(),
            'last_login': None
        }
        
        supabase.table('users').insert(user_data).execute()
        
        os.unlink(temp_file.name)
        return True, "Registration successful!"
    except Exception as e:
        os.unlink(temp_file.name)
        return False, f"Face processing failed: {str(e)}"

# Authenticate user using Streamlit's camera input
def authenticate_user():
    encodings_db = load_face_encodings()
    if not encodings_db['encodings']:
        return None, "No registered users found"
    
    st.write("Please look at the camera for authentication")
    picture = st.camera_input("Take a picture")
    
    if picture:
        try:
            # Convert to numpy array
            image = Image.open(picture)
            img_array = np.array(image)
            
            # Get face embedding
            embedding_objs = DeepFace.represent(
                img_path=img_array,
                model_name='Facenet',
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            if not embedding_objs:
                return None, "No face detected"
                
            current_encoding = embedding_objs[0]['embedding']
            
            # Compare with stored encodings
            face_distances = []
            for known_encoding in encodings_db['encodings']:
                distance = np.linalg.norm(np.array(known_encoding) - np.array(current_encoding))
                face_distances.append(distance)
                
            min_distance = min(face_distances)
            best_match_index = face_distances.index(min_distance)
            
            if min_distance < 0.6:  # Threshold for face matching
                user_id = encodings_db['labels'][best_match_index]
                return user_id, None
            else:
                return None, "Authentication failed. No matching face found."
                
        except Exception as e:
            return None, f"Authentication error: {str(e)}"
    
    return None, "No image captured"

# Streamlit UI
def main():
    st.title("Face Authentication System")
    
    menu = ["Home", "Register", "Authenticate", "Admin"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Face Authentication System")
        
    elif choice == "Register":
        st.subheader("Register New User")
        username = st.text_input("Username")
        email = st.text_input("Email")
        image_file = st.file_uploader("Upload Profile Picture", type=['jpg', 'png', 'jpeg'])
        
        if st.button("Register"):
            if username and email and image_file:
                success, message = register_user(username, email, Image.open(image_file))
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please fill all fields")
    
    elif choice == "Authenticate":
        st.subheader("Authenticate User")
        user_id, error = authenticate_user()
        if user_id:
            user_data = supabase.table('users').select("*").eq('user_id', user_id).execute().data[0]
            st.success(f"Authentication successful! Welcome {user_data['username']}")
            st.image(user_data['image_url'], caption=user_data['username'], width=200)
            
            # Update last login time
            supabase.table('users').update({'last_login': datetime.now().isoformat()}).eq('user_id', user_id).execute()
        elif error:
            st.error(error)
    
    elif choice == "Admin":
        st.subheader("Admin Panel")
        if st.button("View Registered Users"):
            users = supabase.table('users').select("*").execute().data
            if users:
                for user in users:
                    with st.expander(user['username']):
                        st.write(f"Email: {user['email']}")
                        st.write(f"Registered: {user['registration_date']}")
                        st.write(f"Last Login: {user['last_login'] or 'Never'}")
                        st.image(user['image_url'], width=150)
            else:
                st.warning("No users registered yet")

if __name__ == "__main__":
    main()