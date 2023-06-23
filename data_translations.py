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
            zero_node = data[i][0] # saves 0 node before changes
            #print(zero_node)
            for k in range(2): # x/y iteration
                if k == 0:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / 320
                if k == 1:
                    tensor_return[i][j][k] = (zero_node[k] - data[i][j][k]) / 270
            tensor_return[i][0][0] = 0 # reset zero node x
            tensor_return[i][0][1] = 0 # reset zero node y
    return tensor_return

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
            should_save = True
            if acc > 0.925 and loss < 0.07:
                break
        print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")
    return should_save



def main():
    train_data, train_labels = collect_data(24000)
    test_data, test_labels = collect_data(2400, 21600)
    
    print("Successfully collected data")

    train_data = torch.from_numpy(train_data)
    train_data = normalize_data(train_data)
    test_data = torch.from_numpy(test_data)
    test_data = normalize_data(test_data)
    
    print("Successfully normalized data")
    
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    train_data = train_data.reshape(-1, 1, 21, 2)
    test_data = test_data.reshape(-1, 1, 21, 2)
    train_data = train_data.float()

    print("Successfully converted to tensors")
    
    train_dataset = ImageDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=130, shuffle=True, num_workers=2)

    print("Successfully created dataset")
    
    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-6)
    criteron = nn.CrossEntropyLoss()

    print("Successfully created model")
    
    model.to(device)
    if train_model(model, train_loader, criteron, optimizer, 500, test_data, test_labels):
        torch.save(model, "asl_cnn_model.pth")
        torch.save(model.state_dict(), "asl_cnn_model_weights.pth")



if __name__ == "__main__":
    main()