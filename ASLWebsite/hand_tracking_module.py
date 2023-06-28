import cv2
import mediapipe as mp
import torch
import __main__, os

class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5,asl=False):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.landmark_tensor = torch.zeros(1, 1, 21, 2)
        if asl:
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
        data_splice = torch.split(data, 1, dim=3)
        x = data_splice[0]
        y = data_splice[1]
        width = x.max() - x.min()
        height = y.max() - y.min()
        width, height = int(width), int(height)
        zero_node = data[0][0][0] # saves 0 node before changes
        for j in range(1, 21): # 1-20 nodes iterated
            joint = data[0][0][j]
            tensor_return[0][0][j][0] = (zero_node[0] - joint[0]) / width
            tensor_return[0][0][j][1] = (zero_node[1] - joint[1]) / height
        tensor_return[0][0][0][0] = 0 # reset zero node x
        tensor_return[0][0][0][1] = 0 # reset zero node y
        return tensor_return

    def position_finder(self, image, handNo=0, draw=True):
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.landmark_tensor[0][0][id][0] = cx
                self.landmark_tensor[0][0][id][1] = cy
                if id == 20 and draw:
                    cv2.circle(image, (cx, cy), 15, (25, 255, 25), cv2.FILLED)  
                if id == 0 and draw:
                    cv2.circle(image, (cx, cy), 15, (5, 255, 250), cv2.FILLED)
            self.landmark_tensor = self.normalize_hand(self.landmark_tensor)
                
            if draw:
                cv2.circle(image,(cx,cy), 5 , (255,0,255), cv2.FILLED)
    
    def letter_display(self, image, letter="", x=25, y=50):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Using cv2.putText() method
        cv2.rectangle(image, (1, y - 50), (x + 250, y + 60), (171, 68, 14), cv2.FILLED)
        cv2.rectangle(image, (1, y - 50), (x + 250, y + 60), (59,1,1), 2)
        cv2.putText(image, "I think it's " + letter[0], (x, y - 25), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, "But it could instead be:", (x, y), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(image, letter[1], (x + 12, y + 25), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(image, letter[2], (x + 12, y + 45), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return image

    def estimate_letter(self):
        alphabet = "abcdefghiklmnopqrstuvwxy"
        letters = torch.topk(self.asl_model(self.landmark_tensor), 3).indices.tolist()[0]
        confidence = torch.topk(self.asl_model(self.landmark_tensor), 3).values.tolist()[0]
        letters = [alphabet[i] for i in letters]
        for i in range(len(letters)):
            letters[i] = letters[i] + " " + str(round(confidence[i]/10, 2)) + "%"
        return letters

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        keep_prob = 0.5
        # L1 ImgIn shape=(?, 21, 2, 1)
        # Conv -> (?, 21, 2, 32)
        # Pool -> (?, 10, 1, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 10, 1, 32)
        # Conv      ->(?, 10, 1, 64)
        # Pool      ->(?, 5, 0, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 5, 0, 128)
        # Pool ->(?, 2, 0, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(1408, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.Linear(625, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 24, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

setattr(__main__, "CNN", CNN)