import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_flickering(video_path, threshold=50):
    cap = cv2.VideoCapture(video_path)
    brightness_changes = []
    ret, previous_frame = cap.read()
    if not ret:
        st.write("Failed to read video.")
        cap.release()
        return [], []

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(previous_gray, gray)
        mean_diff = np.mean(diff)
        brightness_changes.append(mean_diff)
        previous_gray = gray
        frame_count += 1

    cap.release()
    flickering_frames = [i for i, change in enumerate(brightness_changes) if change > threshold]
    return flickering_frames, brightness_changes

st.title("Video Flickering Detector")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
threshold = st.slider("Threshold for flickering detection", 0, 100, 50)

if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    flickering_frames, brightness_changes = detect_flickering("temp_video.mp4", threshold)
    st.write("Detected flickering frames:", flickering_frames)
    plt.plot(brightness_changes)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold})")
    plt.xlabel("Frame")
    plt.ylabel("Brightness Change")
    plt.legend()
    st.pyplot(plt)
