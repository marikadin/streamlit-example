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

            # Check if there is a match
            if True in results:
                matching_label = None
                if known_face_labels:  # Check if known_face_labels is not empty
                    labels = np.array(known_face_labels)
                    matching_label = labels[results.index(True)]
                return matching_label, face_image

    return None, None


def main():
    st.title("Face Recognition App")

    # Create a webcam instance
    video_capture = cv2.VideoCapture(0)

    # Checkbox to save face
    save_face_button = st.button("Save Face")

    # Checkbox to recognize face
    recognize_face_button = st.button("Recognize Face")

    recognized_name = None
    recognized_image = None

    # Face recognition loop
    while not recognized_name:
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

                # Display the frame with faces
                st.image(frame, channels="BGR", use_column_width=True)

                # Save the face encoding and label if "Save Face" button is clicked
                if save_face_button and not st.session_state.scanning_face:
                    st.session_state.known_face_encodings.append(face_encoding[0])
                    name = st.text_input("Enter the name for the person:")
                    if name:
                        st.session_state.known_face_labels.append(name)
                        st.write(f"Face saved for {name}!")
                        st.session_state.scanning_face = True
                        break

        # Check if "Recognize Face" button is clicked
        if recognize_face_button:
            # Call the function to recognize the face
            recognized_name, recognized_image = recognize_face(frame, st.session_state.known_face_encodings, st.session_state.known_face_labels)

            # Display the recognized face
            if recognized_name and recognized_image is not None:
                st.image(recognized_image, channels="BGR", use_column_width=True, caption=f"Recognized as: {recognized_name}")
                st.write(f"Face recognized as {recognized_name}")
                break

        # Pause for 5 seconds before capturing the next frame
        time.sleep(5)

    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    main()

