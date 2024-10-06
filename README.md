
# Facial Recognition Attendance System

## Project Overview
This project is a **Facial Recognition Attendance System** developed using Python. It employs computer vision techniques to automate the process of recording attendance based on facial recognition. The system captures live video feed, detects faces, and matches them against a stored database of known faces. Upon recognition, it logs attendance details, including timestamps, into Google Cloud Datastore.

## Technologies Used
- **Python**: The primary programming language for developing the application.
- **OpenCV**: For video capture and image processing.
- **face_recognition**: For facial detection and encoding.
- **Google Cloud Datastore**: For storing face encodings and attendance records.
- **Tkinter**: For creating the graphical user interface (GUI).

## Features
- **Real-time Face Detection**: Utilizes a webcam to capture live video and detect faces.
- **Attendance Logging**: Automatically marks attendance for recognized individuals, avoiding duplicate entries within a short time frame.
- **Adding New Faces**: Allows the user to input new individuals into the system, capturing and encoding their faces for future recognition.
- **User-Friendly Interface**: The GUI provides buttons to take attendance or add new faces.


