import face_recognition
import cv2
import pickle
import sqlite3
from datetime import datetime
import numpy as np

def load_encodings():
    try:
        with open("database/encodings.pkl", "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None

def initialize_database():
    try:
        conn = sqlite3.connect("database/attendance.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id TEXT, 
            date TEXT, 
            time TEXT,
            confidence REAL,
            status TEXT,
            PRIMARY KEY (id, date)
        )
        """)
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None, None

def process_face(face_encoding, data):
    # Calculate face distances for better matching
    face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
    matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
    
    name = "Unknown"
    confidence = 0

    if True in matches:
        # Get the index of best match
        best_match_idx = np.argmin(face_distances)
        if matches[best_match_idx]:
            confidence = (1 - face_distances[best_match_idx]) * 100
            if confidence > 60:  # Confidence threshold
                name = data["names"][best_match_idx]
    
    return name, confidence

def main():
    # Load face encodings
    data = load_encodings()
    if data is None:
        print("Failed to load encodings. Exiting.")
        return

    # Initialize database
    conn, cursor = initialize_database()
    if conn is None:
        print("Failed to initialize database. Exiting.")
        return

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create window with fixed size
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Recognition', 640, 480)

    # Track recognized faces to prevent duplicate attendance
    attendance_recorded = set()
    frame_count = 0
    recognition_history = {}

    print("Face recognition started. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:  # Process every other frame
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG method (faster)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name, confidence = process_face(face_encoding, data)
                
                # Track recognition history
                if name != "Unknown":
                    if name not in recognition_history:
                        recognition_history[name] = []
                    recognition_history[name].append(confidence)
                    
                    # Keep only recent recognitions
                    if len(recognition_history[name]) > 5:
                        recognition_history[name].pop(0)
                    
                    # Check if consistently recognized with high confidence
                    if (len(recognition_history[name]) >= 3 and 
                        np.mean(recognition_history[name]) > 70):
                        
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        if (name, current_date) not in attendance_recorded:
                            try:
                                current_time = datetime.now().strftime("%H:%M:%S")
                                cursor.execute("""
                                    INSERT OR REPLACE INTO attendance 
                                    (id, date, time, confidence, status) 
                                    VALUES (?, ?, ?, ?, ?)
                                """, (name, current_date, current_time, confidence, "Present"))
                                conn.commit()
                                attendance_recorded.add((name, current_date))
                                print(f"Attendance recorded for {name}")
                            except sqlite3.Error as e:
                                print(f"Database error: {e}")

                # Draw face rectangle and name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Display name and confidence
                if name != "Unknown":
                    label = f"{name} ({confidence:.1f}%)"
                else:
                    label = name
                    
                cv2.putText(frame, label, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
