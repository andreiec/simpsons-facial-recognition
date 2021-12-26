import os
import cv2 as cv
import numpy as np
import pandas as pd

TRAIN_IMAGES_PATH = "./files/antrenare/bart/"
TRAIN_IMAGES_LABELS = "./files/antrenare/bart.txt"

images = []
labels = {}


# Show image with all bounding boxes for faces
def image_show_with_bounding_boxes(path, image_labels):
    img = cv.imread(path)
    for label in image_labels:
        img = cv.rectangle(img, (label[0], label[1]), (label[2], label[3]), (0, 255, 0), 1)
    cv.imshow(path, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Read images for bart
for image in os.listdir(TRAIN_IMAGES_PATH):
    images.append(os.path.join(TRAIN_IMAGES_PATH, image))


# Load labels
with open(TRAIN_IMAGES_LABELS) as f:
    lines = f.readlines()

    for line in lines:
        data = line.split()

        key = data[0]
        x1, y1, x2, y2, name = int(data[1]), int(data[2]), int(data[3]), int(data[4]), data[5]

        # Check if key is already in dict. If yes then append the coords to the list
        if key in labels.keys():
            dict_value = labels.get(key)
            dict_value.append((x1, y1, x2, y2, name))
            labels[key] = dict_value
        else:
            labels[key] = [(x1, y1, x2, y2, name)]


for i in range(5):
    image_show_with_bounding_boxes(images[i], labels[images[i].split("/")[-1]])
