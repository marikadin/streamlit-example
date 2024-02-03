import cv2
import face_recognition
import streamlit as st
import numpy as np
import time

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize session state variables
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'known_face_labels' not in st.session_state:
    st.session_state.known_face_labels = []
if 'scanning_face' not in st.session_state:
    st.session_state.scanning_face = False
if 'stop_loop' not in st.session_state:
    st.session_state.stop_loop = False

def recognize_face(frame, known_face_encodings, known_face_labels):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = frame[y:y+h, x:x+w]

        # Convert BGR image to RGB
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Encode the face using face_recognition library
        current_face_encoding = face_recognition.face_encodings(rgb_face_image)

        if current_face_encoding:
            # Compare the current face with known faces
            results = face_recognition.compare_faces(known_face_encodings, current_face_encoding[0])
            labels = np.array(known_face_labels)

            # Check if there is a match
            if True in results:
                matching_label = labels[results.index(True)]
                return matching_label

    return None

def main():
    st.title("Face Recognition App")

    # Create a webcam instance
    video_capture = cv2.VideoCapture(0)

    # Checkbox to save face
    save_face_button = st.button("Save Face")

    # Checkbox to recognize face
    recognize_face_button = st.button("Recognize Face")

    while not st.session_state.stop_loop:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop over detected faces
        for (x, y, w, h) in faces:
            # Extract the face region
            face_image = frame[y:y+h, x:x+w]

            # Convert BGR image to RGB
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Encode the face using face_recognition library
            face_encoding = face_recognition.face_encodings(rgb_face_image)

            if face_encoding:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Save the face encoding and label if "Save Face" button is clicked
                if save_face_button and not st.session_state.scanning_face:
                    st.session_state.known_face_encodings.append(face_encoding[0])
                    name = st.text_input("Enter the name for the person:")
                    if name:
                        st.session_state.known_face_labels.append(name)
                        st.write(f"Face saved for {name}!")
                        st.session_state.stop_loop = True
                    st.session_state.scanning_face = True

        # Display the frame with faces
        st.image(frame, channels="BGR", use_column_width=True)

        # Check if "Recognize Face" button is clicked
        if recognize_face_button:
            # Capture another frame
            ret, frame = video_capture.read()

            # Call the function to recognize the face
            recognized_name = recognize_face(frame, st.session_state.known_face_encodings, st.session_state.known_face_labels)

            # Display the recognized face
            if recognized_name:
                st.image(frame, channels="BGR", use_column_width=True, caption=f"Recognized as: {recognized_name}")
                st.write(f"Face recognized as {recognized_name}")
                # Stop the loop after recognizing the face
                st.session_state.stop_loop = True

    st.warning("Face recognition stopped. Stopping face recognition.")

if __name__ == "__main__":
    main()
