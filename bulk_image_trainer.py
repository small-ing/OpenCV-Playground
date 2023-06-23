from hand_tracking_module import *
from data_translations import *
import cv2
import os
import torch
import mediapipe as mp


# For each image, run the hand tracking module
# Build a tensor of the hand landmarks [87000, 1, 21, 2]
# Run the tensor through the model

def main():
    landmark_tensor = torch.zeros(87000, 1, 21, 2)
    label_tensor = torch.zeros(8700, 1)

    # Iterate through all the images in the folder
    path = ""
    tracker = handTracker()
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for index in range(len(os.listdir(path + letter))):
            # open the image -> image
            image = cv2.imread(path + letter + "/" + letter + index + ".jpg")

            # run the hand module on the image -> hand_landmarks ([21, 2])
            tracker.hands_finder(image, draw=False)
            hand_landmarks = tracker.results.multi_hand_landmarks[0]


            # add to the tensor the hand landmarks -> tensor[index] = hand_landmarks
            landmark_tensor[index][0] = hand_landmarks
            label_tensor[index][0] = letter

    # now with the giant filled-in tensor
    criteron = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    train_loader = torch.utils.data.DataLoader(landmark_tensor, batch_size = 130, shuffle=True, num_workers=2)


    model = CNN()
    train_model(model,criteron, optimizer, 10, landmark_tensor, label_tensor)

if __name__ == "__main__":
    main()
