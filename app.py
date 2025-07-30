import os
import shutil
from pathlib import Path
import face_recognition
from PIL import Image
import numpy as np
import streamlit as st
import cv2
import logging

# Setup logging to capture errors
logging.basicConfig(filename='error_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Preprocess image for face recognition
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        # Convert to RGB (face_recognition expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to grayscale for histogram equalization
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
        # Resize to reduce memory usage
        img_rgb = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_LANCZOS4)
        return img_rgb
    except Exception as e:
        logging.info(f"Preprocessing {image_path}: {e}")
        st.error(f"Error preprocessing {image_path}: {e}")
        return None

# Get face encodings from an image
def get_face_encodings(image_path):
    try:
        image = preprocess_image(image_path)
        if image is None:
            return [], image_path
        # Detect face locations using HOG model
        face_locations = face_recognition.face_locations(image, model="hog")
        if not face_locations:
            logging.info(f"No faces detected in {image_path}")
            return [], image_path
        # Generate encodings
        encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=50)
        return encodings, image_path
    except Exception as e:
        logging.info(f"Processing {image_path}: {e}")
        st.error(f"Error processing {image_path}: {e}")
        return [], image_path

# Find matching faces in images
def find_matching_faces(reference_paths, image_paths, tolerance):
    ref_encodings = []
    for ref_path in reference_paths:
        encodings, _ = get_face_encodings(ref_path)
        if encodings:
            ref_encodings.extend(encodings)
        else:
            logging.info(f"No faces in reference image {ref_path}")
    
    if not ref_encodings:
        st.error("No faces detected in any reference image!")
        return []
    
    # Average encodings for robustness
    ref_encoding = np.mean(ref_encodings, axis=0)
    matches = []
    
    total = len(image_paths)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, path in enumerate(image_paths):
        status_text.text(f"Processing image {i+1}/{total}: {os.path.basename(path)}")
        progress_bar.progress((i+1)/total)
        encodings, _ = get_face_encodings(path)
        for enc in encodings:
            distance = face_recognition.face_distance([ref_encoding], enc)[0]
            if distance <= tolerance:
                matches.append(path)
                logging.info(f"Match found for {path}, distance: {distance}")
                break
            else:
                logging.info(f"No match for {path}, distance: {distance}")
    
    return matches

# Main Streamlit app
def main():
    st.title("Find Person in Photos (Glasses & Age Support)")
    
    # Reference images upload
    st.header("Upload Reference Images")
    reference_files = st.file_uploader("Choose reference images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    reference_paths = []
    
    if reference_files:
        os.makedirs("temp/references", exist_ok=True)
        for ref_file in reference_files:
            ref_path = os.path.join("temp/references", ref_file.name)
            with open(ref_path, "wb") as f:
                f.write(ref_file.getbuffer())
            reference_paths.append(ref_path)
            st.image(ref_path, width=100, caption=ref_file.name)
    
    # Images to search
    st.header("Upload Images to Search")
    image_files = st.file_uploader("Choose images to search", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    image_paths = []
    
    if image_files:
        os.makedirs("temp/images", exist_ok=True)
        for img_file in image_files:
            img_path = os.path.join("temp/images", img_file.name)
            with open(img_path, "wb") as f:
                f.write(img_file.getbuffer())
            image_paths.append(img_path)
    
    # Tolerance setting
    st.header("Settings")
    tolerance = st.slider("Tolerance (0.4-0.9, higher is looser)", min_value=0.4, max_value=0.9, value=0.65, step=0.01)
    
    # Process images
    if st.button("Find Matching Faces"):
        if not reference_paths:
            st.error("Please upload at least one reference image!")
            return
        if not image_paths:
            st.error("Please upload images to search!")
            return
        
        st.info("Processing images...")
        matches = find_matching_faces(reference_paths, image_paths, tolerance)
        
        # Display results
        st.subheader(f"Found {len(matches)} Matching Images")
        if matches:
            for match in matches:
                col1, col2, col3 = st.columns([1, 2, 1])
                col1.image(match, width=100)
                col2.write(f"**Name**: {os.path.basename(match)}")
                col2.write(f"**Path**: {match}")
                with open(match, "rb") as file:
                    col3.download_button(
                        label="Download",
                        data=file,
                        file_name=os.path.basename(match),
                        mime="image/jpeg"
                    )
        else:
            st.write("No matching images found.")
        
        # Log matches
        with open("matches_log.txt", "w") as f:
            f.write(f"Matched Images:\n{', '.join(matches)}\n")
        
        st.success("Face recognition completed! Check error_log.txt for issues.")

if __name__ == "__main__":
    main()