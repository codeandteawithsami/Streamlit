import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("Real-time Pokémon Detection")

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")

# Function to process video and annotate frames
def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        st.error("Error: Could not open video.")
        return

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        results = model(frame)
        if len(results) > 0:
            annotated_frame = results[0].plot()  # assuming results[0] contains the annotations
        else:
            annotated_frame = frame

        # Convert the frame to RGB (OpenCV uses BGR by default)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Yield the frame
        yield annotated_frame

    video_capture.release()

# File uploader for video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    # Process the video and display the annotated frames
    video_frames = process_video(tfile.name)

    # Display video frames in Streamlit
    stframe = st.empty()
    for frame in video_frames:
        stframe.image(frame)

    # Clean up temporary files
    os.remove(tfile.name)