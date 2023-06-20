import json
import os
import numpy as np

#                0    1    2   3  4  5  6    7   8  9   10  11   12   13  14  15  16   17   18  19  20  21  22 23 24 25
landmark_map = [None, 0, None, 5, 6, 7, 8, None, 9, 10, 11, 12, None, 17, 18, 19, 20, None, 13, 14, 15, 16, 1, 2, 3, 4]

def collect_data(batch_size=10, offset=0):
    empty_marks = np.zeros((21, 26, batch_size, 2))
    for index in range(offset, offset+batch_size):
        print
        zero_fill = ""
        if index < 10:
            zero_fill = "00000"
        if index < 100 and index >= 10:
            zero_fill = "0000"
        if index < 1000 and index >= 100:
            zero_fill = "000"
        if index < 10000 and index >= 1000: 
            zero_fill = "00"

        for letter in "abcdefghijklmnopqrstuvwxyz":
            try:
                with open(os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json")) as file:
                    data = json.load(file)
                    try:
                        assert letter == data["Letter"]
                    except AssertionError:
                        print("Letter mismatch: " + letter + " != " + data["Letter"] + "in File " + os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json"))
                    for joint in range(26): # splits original joints
                        for dim in range(2): # splits x/y
                            if landmark_map[joint] != None: # checks if joint is in new landmark map
                                if empty_marks[landmark_map[joint]]["abcdefghijklmnopqrstuvwxyz".index(letter)][index-offset][dim] == 0:
                                    empty_marks[landmark_map[joint]]["abcdefghijklmnopqrstuvwxyz".index(letter)][index-offset][dim] = data["Landmarks"][joint][dim]  
                                else: # break everything if there is already a value
                                    assert 1 + 1 == 3
            except FileNotFoundError:
                print("File not found: " + os.path.join("data", letter + "_annotation", zero_fill + str(index) + ".json"))
    return empty_marks

def main():
    print(collect_data(5, 500))



if __name__ == "__main__":
    main()