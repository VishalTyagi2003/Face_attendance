# Facial Recognition Attendance System

## Project Overview
This project is a **Facial Recognition Attendance System** developed using Python. It employs computer vision techniques to automate the process of recording attendance based on facial recognition. The system captures live video feed, detects faces, and matches them against a stored database of known faces. Upon recognition, it logs attendance details, including timestamps, into Google Cloud Datastore.

## Technologies Used
- **Python**: The primary programming language for developing the application.
- **OpenCV**: For video capture and image processing.
- **face_recognition**: For facial detection and encoding.
- **Google Cloud Datastore**: For storing face encodings and attendance records.
- **Tkinter**: For creating the graphical user interface (GUI).

## Requirements
- `shape_predictor_68_face_landmarks.dat`: This file is essential for facial landmark detection and is required for the blink detection feature to work effectively. Please ensure that it is available in the project directory.

## Features
- **Real-time Face Detection**: Utilizes a webcam to capture live video and detect faces.
  
- **Blink Detection for Liveness Verification**: One of the critical improvements in my system is the integration of blink detection to verify liveness. This feature helps ensure that the system can distinguish between a live person and a photo or video, reducing the risk of spoofing attacks. By incorporating this biometric verification method, I am enhancing the security and accuracy of attendance tracking.

- **Attendance Logging**: Automatically marks attendance for recognized individuals, avoiding duplicate entries within a short time frame.

- **Integration with Google Cloud Datastore**: I have implemented a backend system using Google Cloud Datastore for storing facial encodings and attendance records. This allows for scalable and efficient data management. Utilizing cloud infrastructure ensures that the system can handle a large number of users and provides real-time updates without compromising performance.

- **Adding New Faces**: Allows the user to input new individuals into the system, capturing and encoding their faces for future recognition.

- **Customizable Contrast Adjustment with CLAHE**: To improve the system's performance under varying lighting conditions, I have integrated Contrast Limited Adaptive Histogram Equalization (CLAHE). This feature enhances image contrast in real time, allowing for better facial recognition accuracy in different environments. It ensures that the system remains robust and reliable, regardless of the lighting conditions during attendance taking.

- **User-Friendly Interface**: The GUI provides buttons to take attendance or add new faces.
