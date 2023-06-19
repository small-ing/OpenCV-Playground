import hand_tracking_module as htm
import cv2
import mediapipe as mp
import random

cap = cv2.VideoCapture(0)
tracker = htm.handTracker()

def letter_parser():
    letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return letter

while True:
    success, image = cap.read()
    image = tracker.hands_finder(image)
    lmList = tracker.position_finder(image)
    tracker.letter_display(image, letter_parser())
    #if len(lmList) != 0:
        #print(lmList[4])

    cv2.imshow("Video",image)
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
 
