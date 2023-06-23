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
        
        self.landmark_tensor = torch.zeros(1, 1, 21, 2)
        self.asl_model = torch.load("asl_cnn_model_fin.pth")
        self.asl_model.load_state_dict(torch.load("asl_cnn_model_weights_fin.pth"))
        self.asl_model.eval()
        
    def hands_finder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def normalize_hand(self, data):
        tensor_return = torch.zeros(data.shape)
        zero_node = data[0][0][0] # saves 0 node before changes
        for j in range(1, 21): # 1-20 nodes iterated
            joint = data[0][0][j]
            tensor_return[0][0][j][0] = (zero_node[0] - joint[0]) / 640
            tensor_return[0][0][j][1] = (zero_node[1] - joint[1]) / 360
        tensor_return[0][0][0][0] = 0 # reset zero node x
        tensor_return[0][0][0][1] = 0 # reset zero node y
        #print(tensor_return)
        return tensor_return

    def position_finder(self, image, handNo=0, draw=True):
        lmlist = [] # use list to create real time update coordinate list
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                self.landmark_tensor[0][0][id][0] = cx
                self.landmark_tensor[0][0][id][1] = cy
            self.landmark_tensor = self.normalize_hand(self.landmark_tensor)
                
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
    
    def letter_display(self, image, letter="", x=25, y=50):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Using cv2.putText() method
        #cv2.rectangle(image, (x-10, y-50), (x+400, y+150), (0,0,0), cv2.FILLED)
        cv2.putText(image, "I think it's " + letter[0], (x,y-25), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, "But it could instead be:", (x,y), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, letter[1], (x+12,y+25), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, letter[2], (x+12,y+45), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, letter[3], (x+12,y+60), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, letter[4], (x+12,y+75), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #cv2.putText(image, str(self.landmark_tensor), (x,y+500), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # cv2.putText(image, letter_0, (x,y+100), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    def update_letter(self):
        self.stringOut_20 = "ID_20Coord( " + str(self.idSelX_20) + "," + str(self.idSelY_20) + " )"
        self.stringOut_0 = "ID_0Coord( " + str(self.idSelX_0) + "," + str(self.idSelY_0) + " )"

    def estimate_letter(self):
        alphabet = "abcdefghiklmnopqrstuvwxy"
        letters = torch.topk(self.asl_model(self.landmark_tensor), 5).indices.tolist()[0]
        confidence = torch.topk(self.asl_model(self.landmark_tensor), 5).values.tolist()[0]
        #print(letters)
        #print(confidence)
        letters = [alphabet[i] for i in letters]
        for i in range(len(letters)):
            letters[i] = letters[i] + " " + str(round(confidence[i], 2)) + "%"
        return letters


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
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

if __name__ == "__main__":
    main()