import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from google.cloud import datastore
import tkinter as tk
from tkinter import simpledialog, messagebox
import dlib
from scipy.spatial import distance

# Set the path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Projects\FaceRecognitionProject (1)\FaceRecognitionProject\serviceAccountKey.json"

# Initialize Datastore client
datastore_client = datastore.Client()

# Load face detector and facial landmark predictor for blink detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate the Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Indices for the eyes in the 68-point facial landmark model
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))

# Blink detection thresholds
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 3
blink_counter = 0

# CLAHE function for enhancing image contrast
def apply_CLAHE(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

# Store face encoding in Datastore
def store_face_encoding(student_id, name, encoding):
    key = datastore_client.key('FaceEncodings', student_id)
    entity = datastore.Entity(key)
    entity.update({
        'name': name,
        'encoding': encoding
    })
    datastore_client.put(entity)

# Retrieve all face encodings from Datastore
def load_face_encodings():
    query = datastore_client.query(kind='FaceEncodings')
    results = list(query.fetch())
    encodings = []
    names = []
    for result in results:
        encodings.append(np.array(result['encoding']))
        names.append(result['name'])
    return encodings, names

# Store attendance in Datastore
def store_attendance(student_id, present, time, day):
    key = datastore_client.key('Attendance', student_id)
    entity = datastore.Entity(key)
    entity.update({
        'present': present,
        'time': time,
        'day': day
    })
    datastore_client.put(entity)

# Mark attendance
def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString = now.strftime('%Y-%m-%d')
    student_id = name.lower()  # assuming student_id is the lowercase name
    store_attendance(student_id, True, dtString, dateString)
    print(f"Marked attendance for {name} at {dtString} on {dateString}")

# Function to add a new face to the database
def add_new_face(name):
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Please check the camera.")
            break
        cv2.imshow('Add New Face - Press "s" to Save, "q" to Quit', img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:  # Ensure a face was detected
                face_encoding = face_encodings[0]
                student_id = name.lower()
                store_face_encoding(student_id, name, face_encoding.tolist())
                print(f"Added {name} to the database")
                break
            else:
                print("No face detected, please try again.")
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to handle attendance with blink detection and CLAHE preprocessing
def take_attendance():
    global blink_counter
    encodeListKnown, classNamesKnown = load_face_encodings()
    print('Loaded encodings from Datastore')

    attendance_time_tracker = {}
    cap = cv2.VideoCapture(0)
    frame_counter = 0

    # Variables for temporal consistency checks
    previous_face_location = None
    previous_frame = None
    static_frame_counter = 0
    static_threshold = 5  # Number of frames to consider static
    movement_detected = False
    consecutive_blinks = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Please check the camera.")
            break

        display_img = img.copy()  # Store a copy of the original image for display
        img = apply_CLAHE(img)  # Apply CLAHE for lighting condition adjustments

        if frame_counter % 5 == 0:  # Process every 5th frame
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)

            if len(faces) > 0:
                face = faces[0]  # Assume the first detected face is the target
                landmarks = predictor(gray_frame, face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])

                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                if avg_EAR < EAR_THRESHOLD:
                    blink_counter += 1
                    consecutive_blinks += 1  # Count consecutive blinks
                else:
                    if blink_counter >= CONSEC_FRAMES:
                        print("Blink detected! Liveness confirmed.")
                    blink_counter = 0

                # Check for temporal consistency
                if previous_face_location is not None:
                    current_face_location = (face.left(), face.top(), face.right(), face.bottom())
                    if current_face_location == previous_face_location:
                        static_frame_counter += 1
                    else:
                        static_frame_counter = 0
                        movement_detected = True  # Reset motion flag when movement is detected
                    previous_face_location = current_face_location
                else:
                    previous_face_location = (face.left(), face.top(), face.right(), face.bottom())

                # If the face has been static for too long, flag as suspicious
                if static_frame_counter > static_threshold:
                    print("Warning: Face position has been static for too long. Possible use of a static image.")
                    movement_detected = False  # No movement detected

                # Background analysis
                if previous_frame is not None:
                    diff = cv2.absdiff(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), previous_frame)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    non_zero_count = cv2.countNonZero(thresh)

                    # If there is little to no movement in the background
                    if non_zero_count < 500:  # Threshold can be adjusted
                        print("Warning: No significant background movement detected. Possible use of a static image.")
                        movement_detected = False  # No movement detected
                    else:
                        movement_detected = True  # Background motion detected

                # Store the current frame for the next iteration
                previous_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgSmall)
                encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNamesKnown[matchIndex].upper()
                        current_time = datetime.now()

                        # Check if both blink detection and movement detection are positive before marking attendance
                        if movement_detected and (consecutive_blinks > 1 and
                                                  (name not in attendance_time_tracker or 
                                                  (current_time - attendance_time_tracker[name]) > timedelta(seconds=10))):
                            markAttendance(name)
                            attendance_time_tracker[name] = current_time  # Update last attendance time

                        # Draw rectangle and label on the frame
                        y1, x1, y2, x2 = faceLoc
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

        # Display the image with detections
        cv2.imshow('Attendance System', display_img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


# Function to launch the GUI for user actions
def launch_gui():
    def on_add_face():
        name = simpledialog.askstring("Input", "Enter student's name:")
        if name:
            add_new_face(name)

    def on_take_attendance():
        take_attendance()

    root = tk.Tk()
    root.title("Attendance System")
    root.geometry("300x200")

    add_face_button = tk.Button(root, text="Add New Face", command=on_add_face)
    add_face_button.pack(pady=20)

    take_attendance_button = tk.Button(root, text="Take Attendance", command=on_take_attendance)
    take_attendance_button.pack(pady=20)

    root.mainloop()

# Call the GUI launch function
launch_gui()
