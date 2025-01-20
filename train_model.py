import face_recognition
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

def train_face_recognition_model():
    print("Starting face recognition training...")
    
    # Ensure directories exist
    os.makedirs("database", exist_ok=True)
    if not os.path.exists("data"):
        print("Error: 'data' directory not found!")
        return False

    # Initialize lists to store encodings
    known_encodings = []
    known_names = []
    
    # Get list of student folders
    student_folders = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]
    
    if not student_folders:
        print("No student folders found in 'data' directory!")
        return False
    
    print(f"Found {len(student_folders)} student folders")
    
    # Process each student folder
    for student_id in tqdm(student_folders, desc="Processing students"):
        student_path = os.path.join("data", student_id)
        
        # Get all images for this student
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in os.listdir(student_path) 
                      if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if not image_files:
            print(f"No valid images found for student {student_id}")
            continue
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(student_path, image_file)
            
            try:
                # Load and convert image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image: {image_path}")
                    continue
                    
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                if not face_locations:
                    print(f"No face found in {image_path}")
                    continue
                
                # Generate encodings
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if not encodings:
                    print(f"Could not generate encoding for {image_path}")
                    continue
                
                # Add encoding and student ID
                known_encodings.append(encodings[0])
                known_names.append(student_id)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    # Check if we got any encodings
    if not known_encodings:
        print("No face encodings were generated! Check your images.")
        return False
    
    # Save encodings to file
    print("\nSaving encodings...")
    data = {
        "encodings": known_encodings,
        "names": known_names
    }
    
    try:
        with open("database/encodings.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"\nTraining completed successfully!")
        print(f"Total encodings saved: {len(known_encodings)}")
        print(f"Total unique students: {len(set(known_names))}")
        print(f"Encodings file saved to: database/encodings.pkl")
        return True
        
    except Exception as e:
        print(f"Error saving encodings file: {str(e)}")
        return False

if __name__ == "__main__":
    # Install tqdm if not already installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing required package: tqdm")
        os.system("pip install tqdm")
        from tqdm import tqdm
    
    # Run training
    success = train_face_recognition_model()
    
    if success:
        print("\nYou can now run the face recognition system.")
    else:
        print("\nTraining failed! Please check the errors above.")
