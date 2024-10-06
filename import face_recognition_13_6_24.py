import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from google.cloud import datastore
import tkinter as tk
from tkinter import simpledialog, messagebox

# Set the path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Projects\FaceRecognitionProject (1)\FaceRecognitionProject\serviceAccountKey.json"

# Initialize Datastore client
datastore_client = datastore.Client()

# Find encodings for the images (can be used for new faces)
def findEncodings(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(image)[0]
    return encode

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

# Mark attendance by storing it in Datastore
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
        cv2.imshow('Add New Face - Press "s" to Save, "q" to Quit', img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            face_encoding = face_recognition.face_encodings(img)[0]
            student_id = name.lower()
            store_face_encoding(student_id, name, face_encoding.tolist())
            print(f"Added {name} to the database")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to handle attendance
def take_attendance():
    encodeListKnown, classNamesKnown = load_face_encodings()
    print('Loaded encodings from Datastore')

    # Maintain a dictionary to track attendance timestamps
    attendance_time_tracker = {}

    # Start video capture and perform face recognition
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Please check the camera.")
            break

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

                if name not in attendance_time_tracker or (current_time - attendance_time_tracker[name]) > timedelta(seconds=10):  # adjust the time delta as needed
                    markAttendance(name)
                    attendance_time_tracker[name] = current_time

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to prompt for a name and add a new face
def prompt_add_new_face():
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    if name:
        add_new_face(name)
    else:
        messagebox.showwarning("Input Error", "Name cannot be empty!")

# Initialize the Tkinter main window
root = tk.Tk()
root.title("Face Recognition System")

# Create and place buttons in the main window
attendance_button = tk.Button(root, text="Take Attendance", command=take_attendance)
attendance_button.pack(pady=10)

new_face_button = tk.Button(root, text="Add New Face", command=prompt_add_new_face)
new_face_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
