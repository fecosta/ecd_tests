import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_flickering(video_path, threshold=50):
    """
    Detects flickering in a video by measuring brightness changes between frames.

    Parameters:
    - video_path (str): Path to the video file.
    - threshold (float): Threshold for detecting significant brightness change, indicating flickering.

    Returns:
    - flickering_frames (list): List of frame indices where flickering is detected.
    - brightness_changes (list): List of mean brightness changes for each frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}. Please check the path.")
        return [], []

    brightness_changes = []

    # Read the first frame
    ret, previous_frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame of the video.")
        cap.release()
        return [], []

    # Convert the first frame to grayscale
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Loop through each frame in the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or read error

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(previous_gray, gray)
        mean_diff = np.mean(diff)
        brightness_changes.append(mean_diff)

        # Update previous frame
        previous_gray = gray
        frame_count += 1

    cap.release()

    # Check if brightness_changes has valid data
    if not brightness_changes:
        print("Warning: No brightness changes detected, possibly due to video read errors.")
        return [], []

    # Plot brightness changes with a threshold line for flickering detection
    plt.plot(brightness_changes, label="Brightness Change")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold})")
    plt.xlabel("Frame")
    plt.ylabel("Brightness Change")
    plt.title("Brightness Change Between Frames")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Detect flickering frames by checking if brightness change exceeds the threshold
    flickering_frames = [i for i, change in enumerate(brightness_changes) if change > threshold]

    print("Detected flickering frames:", flickering_frames)
    return flickering_frames, brightness_changes
    
# use the video folder
video_path = "./video/avengers.mp4"
#adjust the threshold
flickering_frames, brightness_changes = detect_flickering(video_path, threshold=30)