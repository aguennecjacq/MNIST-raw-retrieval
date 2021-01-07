""" this is a simple script designed to retrieve the mnist dataset in a raw format
    A csv file is used to store the corresponding labels
"""
import pandas as pd
import torchvision
import torch
import os

# create the corresponding folders in case they do not exist
path_train = "./data_raw/MNIST/train_images"
path_test = "./data_raw/MNIST/test_images"

try:
    os.makedirs(path_train)
except OSError:
    print ("Creation of the directory %s failed" % path_train)
    exit()
else:
    print ("Successfully created the directory %s" % path_train)

try:
    os.makedirs(path_test)
except OSError:
    print ("Creation of the directory %s failed" % path_test)
    exit()
else:
    print ("Successfully created the directory %s" % path_test)


# First, get the dataset in their original format
train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True)
test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True)

#  1 - Get training images
gt_map = []
N = len(train_dataset)
progress = 0
# start of iteration over the training dataset
print("retrieving training dataset")
for i in range(0,N):
    # first save image
    img = train_dataset[i][0]
    img.save("./data_raw/MNIST/train_images/"+str(i)+".png")
    
    # next get the ground_truth / label
    label = train_dataset[i][1]
    gt_map.append({"ground_truth": label })
    
    # show progress
    if int((i / N) * 100) == progress:
        print(str(progress)+'%')
        progress+= 10
print("100%")
df_train = pd.DataFrame(gt_map)


# 2 - Get test images
gt_map = []
N = len(test_dataset)
progress = 0
# start of iteration over the test dataset
print("retrieving test dataset")
for i in range(0, N):
    # first save image
    img = test_dataset[i][0]
    img.save("./data_raw/MNIST/test_images/" + str(i) + ".png")

    # next get the ground_truth / label
    label = test_dataset[i][1]
    gt_map.append({"ground_truth": label})

    # show progress
    if int((i / N) * 100) == progress:
        print(str(progress) + '%')
        progress += 10
print("100%")
df_test = pd.DataFrame(gt_map)

# 3 - export labels to csv
df_train.to_csv("./data_raw/MNIST/train_labels.csv")
df_test.to_csv("./data_raw/MNIST/test_labels.csv")
