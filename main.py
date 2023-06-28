from ASLWebsite.hand_tracking_module import handTracker, CNN
import ASLWebsite.hand_tracking_module as htm
import cv2

cap = cv2.VideoCapture(0)

tracker = htm.handTracker(asl=True)

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
    letter=tracker.estimate_letter()
    tracker.letter_display(image,letter=letter)

    cv2.imshow("Video",image)
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
 
