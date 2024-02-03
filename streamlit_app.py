import cv2
import face_recognition
import streamlit as st
import numpy as np
import time

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main():
    st.title("Face Recognition App")

    # Create a webcam instance
    video_capture = cv2.VideoCapture(0)

    # List to store face encodings and corresponding labels
    known_face_encodings = []
    known_face_labels = []

    while True:
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

                # Save the face encoding and label
                known_face_encodings.append(face_encoding[0])
                known_face_labels.append("Person " + str(len(known_face_encodings)))

        # Display the frame with faces
        st.image(frame, channels="BGR", use_column_width=True)

        # Check if there are known faces
        if known_face_encodings:
            # Pause for one second
            time.sleep(1)

            # Capture another frame
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
                current_face_encoding = face_recognition.face_encodings(rgb_face_image)

                if current_face_encoding:
                    # Compare the current face with known faces
                    results = face_recognition.compare_faces(known_face_encodings, current_face_encoding[0])
                    labels = np.array(known_face_labels)

                    # Check if there is a match
                    if True in results:
                        matching_label = labels[results.index(True)]
                        st.success(f"Face recognized as {matching_label}")

if __name__ == "__main__":
    main()
