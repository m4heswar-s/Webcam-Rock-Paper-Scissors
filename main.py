import sys
import math
import random
import cv2 as cv
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def dist_calculator(a, b, hand):
    """This will return the distance between two handlandmarks"""
    point1 = hand[a]
    point2 = hand[b]
    return math.sqrt((point1.x -point2.x)**2 + (point1.y -point2.y)**2)


def score_determiner(result, video_num, robo_score, player_score):
    """
    This will return robo score, player score and points

    point => 1 => player win
    point => 2 => robo win
    point => -1 => draw
    point => -2 => Hand not detected
    point => -3 => Invalid hand gesture
    """

    if not result.hand_landmarks:
        robo_score += 1
        return robo_score, player_score, -2
    
    hand = result.hand_landmarks[0]
    # thumb_ratio = dist_calculator(4, 0, hand) / (dist_calculator(4, 3, hand) + dist_calculator(3, 0, hand))
    indexfing_ratio = dist_calculator(8, 0, hand) / (dist_calculator(8, 6, hand) + dist_calculator(6, 0, hand))
    middlefing_ratio = dist_calculator(12, 0, hand) / (dist_calculator(12, 10, hand) + dist_calculator(10, 0, hand))
    ringfing_ratio = dist_calculator(16, 0, hand) / (dist_calculator(16, 14, hand) + dist_calculator(14, 0, hand))
    pinkyfing_ratio = dist_calculator(20, 0, hand) / (dist_calculator(20, 18, hand) + dist_calculator(18, 0, hand))

    if  indexfing_ratio <= 0.90 and middlefing_ratio <= 0.90 and ringfing_ratio <= 0.90 and pinkyfing_ratio <= 0.90:
        if video_num == 2:
            return robo_score +1, player_score, 0
        elif video_num == 3:
            return robo_score, player_score + 1, 1
        else:
            return robo_score, player_score, -1
    elif indexfing_ratio > 0.90 and middlefing_ratio > 0.90 and ringfing_ratio > 0.90 and pinkyfing_ratio > 0.90:
        if video_num == 3:
            return robo_score +1, player_score, 0
        elif video_num == 1:
            return robo_score, player_score + 1, 1
        else:
            return robo_score, player_score, -1
    elif indexfing_ratio > 0.90 and middlefing_ratio > 0.90 and ringfing_ratio <= 0.90 and pinkyfing_ratio <= 0.90:
        if video_num == 1:
            return robo_score +1, player_score, 0
        elif video_num == 2:
            return robo_score, player_score + 1, 1
        else:
            return robo_score, player_score, -1
    else:
        robo_score += 1
        return robo_score, player_score, -3


#Initialize recognizer options
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 1
)


#Initialize Intro Window
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


# Rounds Loop
while True:
    webcam = cv.VideoCapture(0)

    player_score = 0
    robo_score = 0
    text_result = ""

    #Initialize Rounds Window
    for i in range(1,6):
        text_start_frame = start_frame.copy()
        cv.putText(text_start_frame, f"PLAYER SCORE: {player_score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv.putText(text_start_frame, f"ROBO SCORE: {robo_score}",(int(start_frame.shape[1]-250),30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        start_frame_result = text_start_frame.copy()
        cv.putText(start_frame_result, text_result, (int(start_frame.shape[1]*0.5), int(start_frame.shape[0]*0.2)),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv.imshow("ROCK PAPER SCISSORS",start_frame_result)
        cv.waitKey(1000)
        cv.putText(text_start_frame, f"Round {i}",(int(start_frame.shape[1]*0.5-200), int(start_frame.shape[0]*0.2)),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
        cv.imshow("ROCK PAPER SCISSORS", text_start_frame)
        rand = random.randrange(1,4)
        video_path = f"videos/video{rand}.mp4"
        round_cap = cv.VideoCapture(video_path)
        cv.waitKey(2000)
        while True:
            ret, frame = round_cap.read()
            if not ret:
                with vision.HandLandmarker.create_from_options(options) as recognizer:
                    ret, webcamframe = webcam.read()
                    if not ret:
                        sys.exit("Failed to open webcam")
                    rgb_frame = cv.cvtColor(webcamframe, cv.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = recognizer.detect(mp_image)
                    robo_score, player_score, point = score_determiner(result, rand, robo_score, player_score)
                    print(robo_score, player_score, point)
                    if point == 1:
                        text_result = "WIN"
                    elif point == 0:
                        text_result = "LOSE"
                    elif point == -1:
                        text_result = "DRAW"
                    elif point == -2:
                        text_result = "HAND NOT DETECTED"
                    else:
                        text_result = "INVALID GESTURE"


                text_start_frame = start_frame.copy()
                cv.putText(text_start_frame, f"PLAYER SCORE: {player_score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                cv.putText(text_start_frame, f"ROBO SCORE: {robo_score}",(int(start_frame.shape[1]-250),30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                cv.imshow("ROCK PAPER SCISSORS", text_start_frame)
                break

            cv.putText(text_start_frame, f"PLAYER SCORE: {player_score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv.putText(text_start_frame, f"ROBO SCORE: {robo_score}",(int(start_frame.shape[1]-250),30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv.imshow("ROCK PAPER SCISSORS", frame)
            key_press = cv.waitKey(42)
            if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1 or key_press == ord('e'):
                video_intro.release()
                cv.destroyAllWindows()
                sys.exit("Window closed")

    webcam.release()

    text_start_frame = start_frame.copy()
    cv.putText(text_start_frame, f"PLAYER SCORE: {player_score}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv.putText(text_start_frame, f"ROBO SCORE: {robo_score}",(int(start_frame.shape[1]-250),30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv.putText(text_start_frame, text_result, (int(start_frame.shape[1]*0.5), int(start_frame.shape[0]*0.2)),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv.imshow("ROCK PAPER SCISSORS",text_start_frame)
    cv.waitKey(1000)

    video_outro = cv.VideoCapture("videos/video-outro1.mp4")
    if player_score > robo_score:
        text_result = "WIN"
    elif player_score == robo_score:
        text_result = "DRAW"
    else:
        text_result = "LOSE"
        video_outro = cv.VideoCapture("videos/video-outro2.mp4")


    while True:
        ret, frame = video_outro.read()
        if not ret:
            video_outro.set(cv.CAP_PROP_POS_FRAMES, 24)
            continue
        cv.putText(frame, text_result, (int(start_frame.shape[1]*0.5-80), int(start_frame.shape[0]*0.2)),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
        cv.imshow("ROCK PAPER SCISSORS", frame)

        key_press = cv.waitKey(50) & 0xFF 
        if key_press == ord('q'):
            video_outro.release()
            break
        if cv.getWindowProperty("ROCK PAPER SCISSORS", cv.WND_PROP_VISIBLE) < 1 or key_press == ord('e'):
            video_outro.release()
            cv.destroyAllWindows()
            sys.exit("Window closed")