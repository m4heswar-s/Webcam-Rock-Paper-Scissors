import sys
import random
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def score_determiner(result, video_num):
    pass

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

    key_press = cv.waitKey(32) & 0xFF 
    if key_press == ord('q'):
        video_intro.release()
        break
    if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1 or key_press == ord('e'):
        video_intro.release()
        cv.destroyAllWindows()
        sys.exit("Window closed")

while True:
    win = True
    webcam = cv.VideoCapture(0)

    score = 0

    for i in range(1,6):
        text_start_frame = start_frame.copy()
        cv.putText(text_start_frame, f"SCORE: {score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv.putText(text_start_frame, f"Round {i}",(int(start_frame.shape[1]*0.5-200), int(start_frame.shape[1]*0.1)),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
        cv.imshow("ROCK PAPER SCISSORS", text_start_frame)
        rand = random.randrange(1,4)
        video_path = f"videos/video{rand}.mp4"
        round_cap = cv.VideoCapture(video_path)
        cv.waitKey(2000)
        while True:
            ret, frame = round_cap.read()
            cv.putText(frame, f"SCORE: {score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            if not ret:
                with vision.HandLandmarker.create_from_options(options) as recognizer:
                    ret, webcamframe = webcam.read()
                    rgb_frame = cv.cvtColor(webcamframe, cv.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = recognizer.detect(mp_image)
                    # score_determiner(result, rand)
                text_start_frame = start_frame.copy()
                cv.putText(text_start_frame, f"SCORE: {score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                cv.imshow("ROCK PAPER SCISSORS", text_start_frame)
                break
            cv.imshow("ROCK PAPER SCISSORS", frame)
            key_press = cv.waitKey(42)
            if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1 or key_press == ord('e'):
                video_intro.release()
                cv.destroyAllWindows()
                sys.exit("Window closed")

    webcam.release()
    if win:
        video_outro = cv.VideoCapture("videos/video-outro1.mp4")
    else:
        video_outro = cv.VideoCapture("videos/video-outro2.mp4")

    while True:
        ret, frame = video_outro.read()
        if not ret:
            video_outro.set(cv.CAP_PROP_POS_FRAMES, 24)
            continue
        cv.imshow("ROCK PAPER SCISSORS", frame)

        key_press = cv.waitKey(50) & 0xFF 
        if key_press == ord('q'):
            video_outro.release()
            break
        if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1 or key_press == ord('e'):
            video_outro.release()
            cv.destroyAllWindows()
            sys.exit("Window closed")