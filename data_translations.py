import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import matplotlib.pyplot as plt


#                0    1    2   3  4  5  6    7   8  9   10  11   12   13  14  15  16   17   18  19  20  21  22 23 24 25
landmark_map = [None, 0, None, 5, 6, 7, 8, None, 9, 10, 11, 12, None, 17, 18, 19, 20, None, 13, 14, 15, 16, 1, 2, 3, 4]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alphabet = "abcdefghijklmnopqrstuvwxyz"

def collect_data(batch_size=26, offset=0):
    empty_marks = np.zeros((batch_size, 21, 2))
    empty_labels = np.zeros((batch_size))
    letter_batch = batch_size / 26
    letter_batch = int(letter_batch)

    for letter in alphabet:
        #print("starting " + letter  + " batch")
        offset = offset // 26
        for index in range(offset, offset+letter_batch):
            #print("starting " + letter + " " + str(index))
            total_index = index + alphabet.index(letter)*letter_batch
            empty_labels[total_index-offset] = alphabet.index(letter)
            
            zero_fill = ""
            if index < 10:
                zero_fill = "00000"
            if index < 100 and index >= 10:
                zero_fill = "0000"
            if index < 1000 and index >= 100:
                zero_fill = "000"
            if index < 10000 and index >= 1000: 
                zero_fill = "00"

            try:
                with open(os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json")) as file:
                    data = json.load(file)
                    
                    try:
                        assert letter == data["Letter"]
                    except AssertionError:
                        print("Letter mismatch: " + letter + " != " + data["Letter"] + "in File " + os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json"))
                    
                    for joint in range(26): # splits original joints into new joints  
                        if landmark_map[joint] != None: # checks if joint is in new landmark map
                            #print("Landmark: " + str(joint) + " " + str(landmark_map[joint]))
                            #print("Total Index: " + str(total_index))
                            #print(empty_marks[total_index-offset][landmark_map[joint]])
                            if empty_marks[total_index-offset][landmark_map[joint]][0] == 0 and empty_marks[total_index-offset][landmark_map[joint]][1] == 0:
                                empty_marks[total_index-offset][landmark_map[joint]] = data["Landmarks"][joint]
                                #print("Total Index: " + str(total_index))
                            else: # break everything if there is already a value
                                assert 1 + 1 == 3

            except FileNotFoundError:
                print("File not found: " + os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json"))
    return empty_marks, empty_labels.astype(int)

def normalize_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][0] = data[i][j][0] / 640
            data[i][j][1] = data[i][j][1] / 480 
    return data

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
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
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
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 26, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# def create_model():
#     m = nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Dropout(p=1)
    
    
#     )
#     print(m)
#     return m

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
            #print(img.shape)
            #print(label.shape)
            img = img.to(device)
            img = img.to(torch.float)
            label = label.to(device)
            
            pred = model(img)
            #print("Prediction: " + str(pred))
            #print("Actual: " + str(label))

            loss = loss_fn(pred, label)
            #print(loss)

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
        test_images = test_images.to(torch.float).to(device)
        pred = model(test_images)
        digit = torch.argmax(pred, dim=1)
        # print(test_labels)
        test_labels = test_labels.to(device)
        acc = torch.sum(digit == test_labels)/len(test_labels)
        if acc > 0.2:
            should_save = True
        print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")
    return should_save

def test_model(model, test_images, test_labels):
    test_images = test_images.to(torch.float).to(device)
    pred = model(test_images)
    digit = torch.argmax(pred, dim=1)
    test_labels = test_labels.to(device)
    acc = torch.sum(digit == test_labels)/len(test_labels)
    print(f"Test accuracy: {acc}")

def main():
    train_data, train_labels = collect_data(2340)
    test_data, test_labels = collect_data(260, 2340)

    train_data = torch.from_numpy(train_data)
    train_data = normalize_data(train_data)
    test_data = torch.from_numpy(test_data)
    test_data = normalize_data(test_data)
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)
    print(train_data.shape)
    print(train_labels.shape)
    train_data = train_data.reshape(-1, 1, 21, 2)
    print(train_data.shape)
    test_data = test_data.reshape(-1, 1, 21, 2)
    
    train_data = train_data.float()
    train_dataset = ImageDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130, shuffle=True, num_workers=1)

    model = CNN()


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss()

    model.to(device)
    train_model(model, train_loader, criteron, optimizer, 8, test_data, test_labels)
        # torch.save(model, "model.pth")
        # torch.load("model.pth")




if __name__ == "__main__":
    main()