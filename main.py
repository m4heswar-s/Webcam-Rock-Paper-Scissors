import time
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.VIDEO,
    num_hands = 1
)

while True:
    intro_video = cv.VideoCapture("videos/video.mp4")
    last_frame = None
    while True:
        ret, frame = intro_video.read()
        if not ret:
            break
        last_frame = frame
        cv.imshow("ROCK PAPER SCISSORS",frame)
        cv.waitKey(30)
    cv.imshow("ROCK PAPER SCISSORS", last_frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        pass

print("Ended")