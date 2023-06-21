
    #Programa visual de identificacion de manops al aire libre
import cv2
import mediapipe as mp
from data_translations import *
import torch

class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.idSelX_20 = 0
        self.idSelY_20 = 0
        self.idSelX_0 = 0
        self.idSelY_0 = 0
        self.stringOut_20 = ""
        self.stringOut_0 = ""
        self.landmark_tensor = torch.zeros(1, 21, 2)
        
    def hands_finder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def position_finder(self,image, handNo=0, draw=True):
        lmlist = [] # use list to create real time update coordinate list
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                self.landmark_tensor[0][id][0] = cx
                self.landmark_tensor[0][id][1] = cy
                
            if id == 20 :
                   self.idSelX_20 = cx
                   self.idSelY_20 = cy
                   cv2.circle(image, (cx, cy), 25, (25, 255, 25), cv2.FILLED)  
            if id == 0 :
                   self.idSelX_0 = cx
                   self.idSelY_0 = cy
                   cv2.circle(image, (cx, cy), 25, (5, 255, 250), cv2.FILLED)         
                     
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist
    
    def letter_display(self, image, letter="", x=50, y=50):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        if letter == "":
            letter = self.stringOut_20
        # Using cv2.putText() method
        cv2.rectangle(image, (x-10, y+10), (x+300, y-100), (0,0,0), cv2.FILLED)
        cv2.putText(image, letter, (x,y-25), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        #cv2.putText(image, str(self.landmark_tensor), (x,y+500), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # cv2.putText(image, letter_0, (x,y+100), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    def update_letter(self):
        self.stringOut_20 = "ID_20Coord( " + str(self.idSelX_20) + "," + str(self.idSelY_20) + " )"
        self.stringOut_0 = "ID_0Coord( " + str(self.idSelX_0) + "," + str(self.idSelY_0) + " )"

    def estimate_letter(self):
        pass


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success, image = cap.read()
        image = tracker.hands_finder(image)
        lmList = tracker.position_finder(image)
        tracker.update_letter()
        # stringOut_0 = "ID_0Coord( " + str(idSelX_0) + "," + str(idSelY_0) + " )"
        tracker.letter_display(image, tracker.stringOut_20,tracker.stringOut_0)
        #if len(lmList) != 0:
            #print(lmList[4])
      
        cv2.imshow("Video",image)
        cv2.waitKey(2)
        # cv2.destroyAllWindows() # killer of cpu

if __name__ == "__main__":
    main()