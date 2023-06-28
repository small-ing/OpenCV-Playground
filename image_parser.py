import os
import cv2
import torch
from alive_progress import alive_bar
from PIL import Image
import numpy as np
from ASLWebsite.hand_tracking_module import handTracker, CNN

tracker = handTracker()
alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY"

def collect_train_files():
    landmarks = torch.zeros(80000, 21, 2)
    labels = torch.zeros(87000)
    j = 0
    work = 0
    errors = 0
    for i in "ABCDEFGHIKLMNOPQRSTUVWXY":
        #print("    Current Letter is " + i)
        files = os.listdir("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i)
        with alive_bar(len(files), title=i) as bar:
            for file_name in files:
                with Image.open("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i + "/" + file_name) as fileObject:
                    fileObject = cv2.cvtColor(np.array(fileObject), cv2.COLOR_BGR2RGB)
                    tracker.hands_finder(fileObject, False)
                    hand_landmarks = tracker.results.multi_hand_landmarks
                    if hand_landmarks is not None:
                        hand_landmarks = hand_landmarks[0]
                        landmarks[j] = torch.tensor([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=torch.float32)
                        labels[j] = alphabet.index(i)
                    else:
                        fileObject = cv2.cvtColor(np.array(fileObject), cv2.COLOR_BGR2RGB)
                        tracker.hands_finder(fileObject, False)
                        hand_landmarks = tracker.results.multi_hand_landmarks
                        if hand_landmarks is not None:
                            hand_landmarks = hand_landmarks[0]
                            landmarks[j] = torch.tensor([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=torch.float32)
                            labels[j] = alphabet.index(i)
                            work += 1
                        else:
                            errors += 1
                bar()
            # print("iteration: " + str(j), "errors: " + str(errors))
            j += 1
    print("retrying helped with " + str(work) + " images")
    return landmarks, labels, errors