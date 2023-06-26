from hand_tracking_module import *
import data_translations as dt
from data_translations import *
import cv2
import os
import torch
import random
# import mediapipe as mp
import numpy
from PIL import Image

landmarks = torch.zeros(87000, 1, 21, 2)
labels = torch.zeros(87000)
tracker = handTracker()
#alphabetList = "ABCDEFGHIKLMNOPQRSTUVWXY"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(os.listdir("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train"))
# if you have the images locally, you should only need /asl_alphabet_train
def collect_train_files():
    j = 0
    errors = 0
    for i in "ABCDEFGHIKLMNOPQRSTUVWXY":
        print("Current Letter is " + i)
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
            #print("iteration: " + str(j), "errors: " + str(errors))
            j += 1
    return landmarks, labels, errors
        
def collect_test_files(train_landmarks, train_labels, num_files=100):
    landmarks = torch.zeros(num_files, 1, 21, 2)
    labels = torch.zeros(num_files)
    
    for idx in range(num_files):
        random_idx = random.randint(0, len(train_landmarks) - 1)
        landmarks[idx] = train_landmarks[random_idx]
        labels[idx] = train_labels[random_idx]
        
    return landmarks, labels
    
def main():
    print("Starting main")
    train_landmarks, train_labels, train_errors = collect_train_files()
    print("Finished collecting train files")
    
    temp = train_labels.view(-1)
    zero_index = len(temp.nonzero())
    train_labels = train_labels[:zero_index]
    train_labels = train_labels.long()
    train_landmarks = train_landmarks[:zero_index]
    
    test_landmarks, test_labels = collect_test_files(train_landmarks, train_labels)
    print("Finished collecting test files")
    

    
    #print(len(labels))
    print( "Training Errors: " + str(train_errors) + "out of " + str(len(train_labels)) + " images")
    #print(j - errors)

    print("Successfully created tensors")

    model = dt.CNN()
    model.to(device)
    criteron = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    dataset = dt.ImageDataset(train_landmarks, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=130, shuffle=True, num_workers=0)
    print("Successfully created dataset")

    if dt.train_model(model, train_loader, criteron, optimizer, 200, test_landmarks, test_labels):
        torch.save(model, "asl_cnn_bulk_model.pth")
        torch.save(model.state_dict(), "asl_cnn_bulk_model_weights.pth")
    

# Iterate through all the images in the folder
# For each image, run the hand tracking module
# Build a tensor of the hand landmarks [87000, 1, 21, 2]
# Run the tensor through the model

if __name__ == "__main__":
    main()