import time
import sys
import random
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 1
)

video_intro = cv.VideoCapture("videos/video-intro.mp4")
round_start_img = cv.VideoCapture("videos/video1.mp4")
ret, start_frame  = round_start_img.read()
round_start_img.release()
if not ret:
    sys.exit("Failed to load assets")


while True:
    ret, frame = video_intro.read()
    if not ret:
        video_intro.set(cv.CAP_PROP_POS_FRAMES, 71)
        continue
    cv.imshow("ROCK PAPER SCISSORS", frame)

    if cv.waitKey(32) & 0xFF == ord('q'):
        cv.imshow("ROCK PAPER SCISSORS", start_frame)
        video_intro.release()
        break
    if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1:
        video_intro.release()
        cv.destroyAllWindows()
        sys.exit("Window closed")

webcam = cv.VideoCapture(0)


for i in range(1,6):
    rand = random.randrange(1,4)
    video_path = f"videos/video{rand}.mp4"
    round_cap = cv.VideoCapture(video_path)
    cv.waitKey(2000)
    while True:
        ret, frame = round_cap.read()
        if not ret:
            with vision.HandLandmarker.create_from_options(options) as recognizer:
                cv.waitKey(100)
                ret, webcamframe = webcam.read()
                rgb_frame = cv.cvtColor(webcamframe, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = recognizer.detect(mp_image)
            
            cv.imshow("ROCK PAPER SCISSORS", start_frame)
            break
        cv.imshow("ROCK PAPER SCISSORS", frame)
        cv.waitKey(42)
        if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1:
            video_intro.release()
            cv.destroyAllWindows()
            sys.exit("Window closed")