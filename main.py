import os
import time
import cv2 as cv
import numpy as np
import random
import pickle

from skimage import data, exposure
from skimage.feature import hog
from sklearn import svm

# Constants
SAVE_MODEL = False
LOAD_MODEL = True
TRAIN_MODEL = False
LOAD_IMAGES = False

# Paths
TRAIN_IMAGES_PATH = "./files/antrenare/bart/"
TRAIN_IMAGES_LABELS = "./files/antrenare/bart.txt"

# Sliding window parameters
sliding_window_size = (128, 128)
sliding_window_step_size = 32

sliding_window_predict_size = (64, 64)
sliding_window_predict_step_size = 8

# Data gathering parameters
image_resize_factor = 1
random_window_max_tries = 5

# Training data arrays
images_with_faces_train_data = []
images_without_faces_train_data = []

# Save images paths and their labels
images = []
labels = {}


# Function to yield the complete pyramid of an image
def image_pyramid(img, scale=1.5, minsize=(128, 128)):
    yield img

    while True:
        w = int(img.shape[1] / scale)
        h = int(img.shape[0] / scale)
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

        if img.shape[0] < minsize[1] or img.shape[1] < minsize[0]:
            break

        yield img


# Function to yield a window of an image
def image_sliding_window(img, step_size, window_size):
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            yield x, y, img[y:y + window_size[1], x:x + window_size[0]]


# Show image with all bounding boxes for faces
def show_image_with_bounding_boxes(path, img_labels):
    img = cv.imread(path)
    for label in img_labels:
        img = cv.rectangle(img, (label[0], label[1]), (label[2], label[3]), (0, 255, 0), 1)
    cv.imshow(path, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function to check if two rectangles overlap
def check_if_rectangles_overlap(l1, r1, l2, r2):
    return l1[0] < r2[0] and r1[0] > l2[0] and l1[1] < r2[1] and r1[1] > l2[1]


# Function to return the bounding image of face from label
def get_image_from_label(img, label):
    return img[label[1]:label[3], label[0]:label[2]]


# Read images from folder
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

# show_image_with_bounding_boxes("./files/antrenare/bart/pic_0016.jpg", labels['pic_0016.jpg'])

if LOAD_IMAGES:
    print("Loading images..")

for image_path in images:
    if not LOAD_IMAGES:
        break

    image_labels = labels[image_path.split("/")[-1]]

    # Read image from path and convert it to grayscale
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.blur(image, (5, 5), 1)

    # Append faces from labels (face is resized to match window size)
    for image_label in image_labels:
        face = get_image_from_label(image, image_label)
        face = cv.resize(face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))

        # Get HOG and append
        _, face_hog_image = hog(face, visualize=True)
        images_with_faces_train_data.append(face_hog_image.flatten())

    # Foreach face append a random non face image to the images_without_faces_training_data to even the training data
    for random_image in range(len(image_labels)):
        found_non_face = False
        tries = 0

        # Get random coords while find some that fit the criteria
        while not found_non_face:
            # Select random coords
            rand_y, rand_x = random.randint(10, image.shape[0] - sliding_window_size[1]), random.randint(10, image.shape[1] - sliding_window_size[0])

            # If number of tries is exceeded return
            tries += 1

            # Check if the square would intersect any face from label, if yes, break
            intersects = False
            for image_label in image_labels:
                if check_if_rectangles_overlap((rand_x, rand_y), (rand_x + sliding_window_size[0], rand_y + sliding_window_size[1]), (image_label[0], image_label[1]), (image_label[2], image_label[3])):
                    intersects = True

            if not intersects:
                # Get the HOG and append
                non_face = image[rand_y:rand_y + sliding_window_size[1], rand_x:rand_x + sliding_window_size[0]]
                non_face = cv.resize(non_face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
                _, non_face_hog_image = hog(non_face, visualize=True)
                images_without_faces_train_data.append(non_face_hog_image.flatten())
                found_non_face = True

            if tries >= random_window_max_tries:
                found_non_face = True

if LOAD_IMAGES:
    print("Images loaded!")

# Build classifier labels
train_y_faces = [1] * len(images_with_faces_train_data)
train_y_non_faces = [0] * len(images_without_faces_train_data)
train_y_faces.extend(train_y_non_faces)
train_y = train_y_faces

# Combine training data
images_with_faces_train_data.extend(images_without_faces_train_data)
train_x = np.array(images_with_faces_train_data)

classifier = svm.SVC()

if TRAIN_MODEL:
    print("Training SVM..")
    classifier.fit(train_x, train_y)
    print("SVM Trained!")

# Save SVM model
if SAVE_MODEL:
    print("Saving SVM..")
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    print("SVM Saved!")

# Load SVM model
if LOAD_MODEL:
    print("Loading SVM..")
    filename = 'finalized_model.sav'
    classifier = pickle.load(open(filename, 'rb'))
    print("SVM Loaded!")

# Sliding window for image
predict_image = cv.imread('./files/validare/simpsons_validare/bart_simpson_4.jpg')
predict_image = cv.cvtColor(predict_image, cv.COLOR_BGR2GRAY)
predict_image = cv.blur(predict_image, (3, 3), 1)

for (x, y, window) in image_sliding_window(predict_image, sliding_window_predict_step_size, sliding_window_predict_size):

    if window.shape[0] != sliding_window_predict_size[1] or window.shape[1] != sliding_window_predict_size[0]:
        continue

    window = cv.resize(window, sliding_window_size)
    _, predict_hog_window = hog(window, visualize=True)
    predict_hog_window_np = np.array(predict_hog_window)
    predict_hog_window_np_flatten = predict_hog_window_np.flatten()
    prediction = classifier.predict([predict_hog_window_np_flatten])

    if prediction:
        cv.imshow(str(x) + " " + str(y), window)

    clone = predict_image.copy()
    cv.rectangle(clone, (x, y), (x + sliding_window_predict_size[0], y + sliding_window_predict_size[1]), (0, 255, 0), 2)
    cv.imshow("Window", clone)
    cv.waitKey(1)
    # time.sleep(0.02)

cv.waitKey(0)
