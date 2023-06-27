import json
import os
import numpy as np
from PIL import Image
import cv2
from ASLWebsite.hand_tracking_module import handTracker, CNN
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init
import time


#                0    1    2   3  4  5  6    7   8  9   10  11   12   13  14  15  16   17   18  19  20  21  22 23 24 25
landmark_map = [None, 0, None, 5, 6, 7, 8, None, 9, 10, 11, 12, None, 17, 18, 19, 20, None, 13, 14, 15, 16, 1, 2, 3, 4]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tracker = handTracker()
alphabet = "abcdefghiklmnopqrstuvwxy"

def collect_data(batch_size=24, offset=0):
    empty_marks = np.zeros((batch_size, 21, 2))
    empty_labels = np.zeros((batch_size))
    letter_batch = batch_size / 24
    letter_batch = int(letter_batch)

    for letter in alphabet:
        offset = offset // 24
        if letter == "z" or letter == "j":
            pass
        else:
            for index in range(offset, offset+letter_batch):
                total_index = index + alphabet.index(letter)*letter_batch
                empty_labels[total_index-offset] = alphabet.index(letter)

                try:
                    with open(os.path.join("data", letter + "_annotation", '{0:06d}'.format(index) + ".json")) as file:
                        data = json.load(file)
                        try:
                            assert letter == data["Letter"]
                        except AssertionError:
                            print("Letter mismatch: " + letter + " != " + data["Letter"] + "in File " + os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json"))
                        for joint in range(26): # splits original joints into new joints  
                            if landmark_map[joint] != None: # checks if joint is in new landmark map
                                if empty_marks[total_index-offset][landmark_map[joint]][0] == 0 and empty_marks[total_index-offset][landmark_map[joint]][1] == 0:
                                    empty_marks[total_index-offset][landmark_map[joint]] = data["Landmarks"][joint]
                                else: # break everything if there is already a value
                                    assert False

                except FileNotFoundError:
                    print("File not found: " + os.path.join("data", letter + "_annotation", '{0:06d}'.format(index) + str(index) + ".json"))
    return empty_marks, empty_labels.astype(int)

def normalize_data(data):
    tensor_return = torch.zeros(data.shape)
    for i in range(len(data)): # ASL Letters iterated
        for j in range(1, 21): # 1-20 nodes iterated
            width = data[i][:][0].max() - data[i][:][0].min()
            height = data[i][:][1].max() - data[i][:][1].min()
            width, height = int(width), int(height)
            zero_node = data[i][0] # saves 0 node before changes
            for k in range(2): # x/y iteration
                if k == 0:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / width
                if k == 1:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / height
            tensor_return[i][0][0] = 0 # reset zero node x
            tensor_return[i][0][1] = 0 # reset zero node y
    return tensor_return

def normalize_image_data(data):
    tensor_return = torch.zeros(data.shape)
    for i in range(len(data)): # ASL Letters iterated
        for j in range(1, 21): # 1-20 nodes iterated
            width = data[i][:][0].max() - data[i][:][0].min()
            height = data[i][:][1].max() - data[i][:][1].min()
            width, height = int(width), int(height)
            zero_node = data[i][0] # saves 0 node before changes
            for k in range(2): # x/y iteration
                if k == 0:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / width
                if k == 1:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / height
            tensor_return[i][0][0] = 0 # reset zero node x
            tensor_return[i][0][1] = 0 # reset zero node y
    return tensor_return

def collect_train_files():
    landmarks = torch.zeros(80000, 21, 2)
    labels = torch.zeros(87000)
    j = 0
    work = 0
    errors = 0
    for i in "ABCDEFGHIKLMNOPQRSTUVWXY":
        print("    Current Letter is " + i)
        files = os.listdir("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i)
        for file_name in files:
            with Image.open("../../../Downloads/asl_images/asl_alphabet_train/asl_alphabet_train" + "/" + i + "/" + file_name) as fileObject:
                fileObject = cv2.cvtColor(np.array(fileObject), cv2.COLOR_BGR2RGB)
                tracker.hands_finder(fileObject, False)
                hand_landmarks = tracker.results.multi_hand_landmarks
                if hand_landmarks is not None:
                    hand_landmarks = hand_landmarks[0]
                    landmarks[j] = torch.tensor([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=torch.float32)
                    labels[j] = ord(i) - ord("A") + 1
                else:
                    fileObject = cv2.cvtColor(np.array(fileObject), cv2.COLOR_BGR2RGB)
                    tracker.hands_finder(fileObject, False)
                    hand_landmarks = tracker.results.multi_hand_landmarks
                    if hand_landmarks is not None:
                        hand_landmarks = hand_landmarks[0]
                        landmarks[j] = torch.tensor([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=torch.float32)
                        labels[j] = ord(i) - ord("A") + 1
                        work += 1
                    else:
                        errors += 1
            # print("iteration: " + str(j), "errors: " + str(errors))
            j += 1
    print("retrying helped with " + str(work) + " images")
    return landmarks, labels, errors
        
def collect_test_files(train_landmarks, train_labels, num_files=100):
    landmarks = torch.zeros(num_files, 1, 21, 2)
    labels = torch.zeros(num_files)
    
    for idx in range(num_files):
        random_idx = random.randint(0, len(train_landmarks) - 1)
        landmarks[idx][0] = train_landmarks[random_idx]
        labels[idx] = train_labels[random_idx]
        
    return landmarks, labels


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

def train_model(model, train_loader, loss_fn, optimizer, epochs, test_images, test_labels):
    should_save = False
    for i in range(epochs):
        for img, label in train_loader:
            img = img.to(device)
            img = img.to(torch.float)
            label = label.to(device) 
            pred = model(img)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
        test_images = test_images.to(torch.float).to(device)
        pred = model(test_images)
        digit = torch.argmax(pred, dim=1)
        test_labels = test_labels.to(device)
        acc = torch.sum(digit == test_labels)/len(test_labels)
        if acc > 0.9 and loss < 0.15:
            print("Good enough to save")
            should_save = True
            if acc > 0.98 and loss < 0.05:
                print(f"Accuracy - {acc} and Loss - {loss} are ideal")
                print("Model is Ideal, saving now...")
                break
        print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")
    return should_save



def main():
    print("Starting...")
    start_time = time.time()
    train_data, train_labels = collect_data(24000)
    test_data, test_labels = collect_data(2400, 21600)
    print("Collected JSON Data")
    bit_time = time.time()
    print("Collecting Image Data...")
    train_more_data, train_more_labels, errors = collect_train_files()
    print("There were ", errors, " errors in collecting the image data")
    print("It took ", (time.time() - bit_time)/60, " minutes to collect the image data")
    print("Collected Image Data")
    temp = train_more_labels.view(-1)
    zero_index = len(temp.nonzero())
    train_more_labels = train_more_labels[:zero_index]
    train_more_labels = train_more_labels.long()
    train_more_data = train_more_data[:zero_index]
    test_more_data, test_more_labels = collect_test_files(train_more_data, train_more_labels)
    test_more_data = test_more_data
    
    print("Successfully collected all data")
    print("Time Elapsed: ", (time.time() - start_time)/60, " minutes")
    
    train_data = torch.from_numpy(train_data)
    train_data = normalize_data(train_data)
    test_data = torch.from_numpy(test_data)
    test_data = normalize_data(test_data)
    
    train_more_data = normalize_image_data(train_more_data)
    train_data = train_data.reshape(-1, 1, 21, 2)
    test_data = test_data.reshape(-1, 1, 21, 2)
    train_more_data = train_more_data.reshape(-1, 1, 21, 2)
    
    print("Successfully normalized data")
    
    train_data = torch.cat((train_data, train_more_data), 0)
    test_data = torch.cat((test_data, test_more_data), 0)
    
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    train_labels = torch.cat((train_labels, train_more_labels), 0)
    test_labels = torch.cat((test_labels, test_more_labels), 0)
    
    train_data = train_data.float()

    print("Successfully converted to tensors")
    
    train_dataset = ImageDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130, shuffle=True, num_workers=2)

    print("Successfully created dataset")
    
    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    criteron = nn.CrossEntropyLoss()

    print("Successfully created model")
    print("Time Elapsed: ", (time.time() - start_time)/60, " minutes")
    
    model.to(device)
    if train_model(model, train_loader, criteron, optimizer, 500, test_data, test_labels):
        torch.save(model, "asl_cnn_model.pth")
        torch.save(model.state_dict(), "asl_cnn_model_weights.pth")
    print("Total Time Elapsed: ", (time.time() - start_time)/60, " minutes")



if __name__ == "__main__":
    main()