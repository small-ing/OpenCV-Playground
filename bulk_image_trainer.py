from hand_tracking_module import *
from data_translations import *
import cv2
import os
import torch
import mediapipe as mp

# Iterate through all the images in the folder
# For each image, run the hand tracking module
# Build a tensor of the hand landmarks [87000, 1, 21, 2]
# Run the tensor through the model
