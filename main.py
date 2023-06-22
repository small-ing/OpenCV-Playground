import hand_tracking_module as htm
import cv2
import mediapipe as mp
from data_translations import *
cap = cv2.VideoCapture(0)
tracker = htm.handTracker()

#Leo am exist
# brandon was here
#Austin is here
#Aloha from William
while True:
    try:
        success, image = cap.read()
    except:
        print("Camera not found")
        break

    image = tracker.hands_finder(image)
    lmList = tracker.position_finder(image)
    tracker.update_letter()
    letter=tracker.estimate_letter()
    #print(letter)
    tracker.letter_display(image,letter=letter)

    cv2.imshow("Video",image)
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
 
