import cv2
import os
from datetime import datetime
import numpy as np

def check_face_quality(face_img, min_size=(100, 100)):
    """Balance between speed and quality checks"""
    # Check minimum size
    height, width = face_img.shape[:2]
    if height < min_size[0] or width < min_size[1]:
        return False, "Size too small"
    
    # Quick brightness and contrast check
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    if brightness < 40 or brightness > 230:
        return False, "Poor lighting"
    if contrast < 25:
        return False, "Low contrast"
    
    # Basic blur check (faster than Laplacian)
    blur_score = cv2.Sobel(gray, cv2.CV_64F, 1, 0).var()
    if blur_score < 100:
        return False, "Too blurry"
        
    return True, "Good"

def capture_images():
    # Initialize webcam with balanced resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    student_id = input("Enter Student ID: ").strip()
    if not student_id:
        print("Error: Student ID cannot be empty")
        cap.release()
        return

    folder_path = f"data/{student_id}"
    os.makedirs(folder_path, exist_ok=True)

    print("\nCapturing images for Student ID:", student_id)
    print("Instructions:")
    print("- Keep your face centered and well-lit")
    print("- Slowly turn your head (left, right, up, down)")
    print("- Press 'q' to quit\n")

    count = 0
    required_images = 20  # Balanced number of images
    last_save_time = datetime.now()
    min_save_interval = 0.3  # Reduced interval for faster capture
    last_position = None
    min_movement = 20  # Minimum pixels to move between captures

    while count < required_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimized face detection parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(100, 100),
            maxSize=(400, 400)
        )
        
        current_time = datetime.now()
        time_elapsed = (current_time - last_save_time).total_seconds()

        if len(faces) == 1 and time_elapsed >= min_save_interval:
            (x, y, w, h) = faces[0]
            center = (x + w//2, y + h//2)
            
            # Check if face has moved enough
            if last_position is None or \
               abs(center[0] - last_position[0]) > min_movement or \
               abs(center[1] - last_position[1]) > min_movement:
                
                # Add margin to face crop
                margin = int(0.15 * w)  # Slightly reduced margin
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                face_img = frame[y1:y2, x1:x2]

                is_quality, status = check_face_quality(face_img)
                if is_quality:
                    timestamp = current_time.strftime("%H%M%S")
                    face_filename = f"{folder_path}/{student_id}_{timestamp}_{count}.jpg"
                    cv2.imwrite(face_filename, face_img)
                    count += 1
                    last_save_time = current_time
                    last_position = center
                    
                    # Green rectangle for success
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Captured: {count}/{required_images}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Yellow rectangle with status
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, status, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Blue rectangle for waiting movement
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # Red rectangle if face not detected properly
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Capturing Images", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= required_images:
        print(f"\nSuccessfully captured {required_images} images for Student ID: {student_id}")
    else:
        print(f"\nCapture interrupted. Got {count} out of {required_images} images.")

if __name__ == "__main__":
    capture_images()
