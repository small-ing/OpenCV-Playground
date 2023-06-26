from hand_tracking_module import *
import data_translations as dt
from data_translations import *
import cv2
import os
import torch
# import mediapipe as mp
import numpy
from PIL import Image

landmarks = torch.zeros(87000, 1, 21, 2)
labels = torch.zeros(87000)
tracker = handTracker()
#alphabetList = "ABCDEFGHIKLMNOPQRSTUVWXY"
alphabetList = "A"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

j = 0
errors = 0
#print(os.listdir("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train"))
for i in alphabetList:
    files = os.listdir("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i)
    for file_name in files:
        with Image.open("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i + "/" + file_name) as fileObject:
            fileObject = cv2.cvtColor(numpy.array(fileObject), cv2.COLOR_BGR2RGB)
            tracker.hands_finder(fileObject, False)
            hand_landmarks = tracker.results.multi_hand_landmarks
            if hand_landmarks is not None:
                hand_landmarks = hand_landmarks[0]
                landmarks[j][0] = torch.tensor([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=torch.float32)
                labels[j] = ord(i) - ord("A") + 1
            else:
                errors += 1
        j += 1
        #print("iteration: " + str(j), "errors: " + str(errors))
temp = labels.view(-1)
zero_index = len(temp.nonzero())
labels = labels[:zero_index]
landmarks = landmarks[:zero_index]
#print(len(labels))
print("Errors: " + str(errors))
#print(j - errors)

print("Successfully created tensor")

model = dt.CNN()
model.to(device)
criteron = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
dataset = dt.ImageDataset(landmarks, labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=130, shuffle=True, num_workers=0)
print("Successfully created dataset")

dt.train_model(model, train_loader, criteron, optimizer, 10, landmarks, labels)

# Iterate through all the images in the folder
# For each image, run the hand tracking module
# Build a tensor of the hand landmarks [87000, 1, 21, 2]
# Run the tensor through the model