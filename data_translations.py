import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#                0    1    2   3  4  5  6    7   8  9   10  11   12   13  14  15  16   17   18  19  20  21  22 23 24 25
landmark_map = [None, 0, None, 5, 6, 7, 8, None, 9, 10, 11, 12, None, 17, 18, 19, 20, None, 13, 14, 15, 16, 1, 2, 3, 4]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_data(batch_size=26, offset=0):
    empty_marks = np.zeros((batch_size, 21, 2))
    empty_labels = np.zeros((batch_size))
    letter_batch = batch_size / 26
    letter_batch = int(letter_batch)

    for letter in "abcdefghijklmnopqrstuvwxyz":
        print("starting " + letter  + " batch")
        offset = offset // 26
        for index in range(offset, offset+letter_batch):
            print("starting " + letter + " " + str(index))
            total_index = index + "abcdefghijklmnopqrstuvwxyz".index(letter)*letter_batch
            empty_labels[total_index-offset] = "abcdefghijklmnopqrstuvwxyz".index(letter)
            
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
    return empty_marks, empty_labels

def normalize_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][0] = data[i][j][0] / 640
            data[i][j][1] = data[i][j][1] / 480 
    return data

def create_model():
    m = nn.Sequential(
    nn.Conv2d(1, 28, (3, 3), padding=1),
      nn.ReLU(),
    #   nn.MaxPool2d((2, 2)),

      nn.Conv2d(28, 56, (3, 3), padding=1),
      nn.ReLU(),
    #   nn.MaxPool2d((2, 2)),

      nn.Conv2d(56, 56, (3, 3), padding=1),
      nn.ReLU(),
    #   nn.MaxPool2d((2, 2)),

      nn.Flatten(),
      nn.Linear(504, 64),
      nn.ReLU(),
      nn.Linear(64, 26)
    )
    print(m)
    return m

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

def train_model(model, train_loader, loss_fn, optimizer, epochs, test_images, test_labels):
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
        print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")
    
def main():
    train_data, train_labels = collect_data(23400)
    train_data = normalize_data(train_data)
    train_data = torch.from_numpy(train_data)
    train_dataset = ImageDataset(train_data, train_labels)
    test_data, test_labels = collect_data(2600, 23400)
    test_data = normalize_data(test_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130, shuffle=True, num_workers=2)

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criteron = nn.CrossEntropyLoss()

    model.to(device)
    train_model(model, train_loader, criteron, optimizer, 10, test_data, test_labels)

if __name__ == "__main__":
    main()