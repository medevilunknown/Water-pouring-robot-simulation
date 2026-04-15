import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download the model if not present
model_path = '/tmp/pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe pose model...")
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    urllib.request.urlretrieve(url, model_path)

print("Initializing MediaPipe Pose Landmarker...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("Opening physical webcam...")
    cap = cv2.VideoCapture('/dev/video1')

    if not cap.isOpened():
        print("FAIL: Cannot open /dev/video1. Check USB connection or privileges.")
        exit(1)

    ret, frame = cap.read()
    if ret:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = landmarker.detect(mp_image)
        print(f"PASS: Captured frame of shape {frame.shape}.")
        if results.pose_landmarks:
            print("PASS: MediaPipe successfully extracted human landmarks!")
        else:
            print("PASS: Inference ran, but no human detected in frame.")
    else:
        print("FAIL: Device opened, but frame is empty.")

    cap.release()