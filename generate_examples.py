import os
import cv2 as cv
import random

GENERATE_FACES = True
GENERATE_NON_FACES = True

# Paths
TRAIN_IMAGES_PATH = ["./files/antrenare/bart/", "./files/antrenare/homer/", "./files/antrenare/lisa/", "./files/antrenare/marge/"]
TRAIN_IMAGES_LABELS = ["./files/antrenare/bart.txt", "./files/antrenare/homer.txt", "./files/antrenare/lisa.txt", "./files/antrenare/marge.txt"]

# Sliding window parameters
sliding_window_size = (36, 36)

# Data gathering parameters
image_resize_factor = 1
random_window_max_tries = 5

# Training data arrays
images_with_faces_train_data = []
images_without_faces_train_data = []

# Save images paths and their labels
images = []
labels = {}


# Function to return the bounding image of face from label
def get_image_from_label(img, label):
    return img[label[1]:label[3], label[0]:label[2]]


# Function to check if two rectangles overlap
def check_if_rectangles_overlap(l1, r1, l2, r2):
    return l1[0] < r2[0] and r1[0] > l2[0] and l1[1] < r2[1] and r1[1] > l2[1]


# Read images from folders
for folder in TRAIN_IMAGES_PATH:
    for image in os.listdir(folder):
        images.append(os.path.join(folder, image))


# Load labels
for label_file in TRAIN_IMAGES_LABELS:
    with open(label_file) as f:
        lines = f.readlines()

        for line in lines:
            data = line.split()

            key = label_file.split("/")[-1].split(".")[0] + "_" + data[0]
            x1, y1, x2, y2, name = int(data[1]), int(data[2]), int(data[3]), int(data[4]), data[5]

            # Check if key is already in dict. If yes then append the coords to the list
            if key in labels.keys():
                dict_value = labels.get(key)
                dict_value.append((x1, y1, x2, y2, name))
                labels[key] = dict_value
            else:
                labels[key] = [(x1, y1, x2, y2, name)]


for image_path in images:

    # Load labels
    image_labels = labels[image_path.split("/")[-2] + "_" + image_path.split("/")[-1]]

    # Read image from path
    image = cv.imread(image_path)

    # Append faces from labels (face is resized to match window size)
    if GENERATE_FACES:
        for image_label in image_labels:
            face = get_image_from_label(image, image_label)
            face = cv.resize(face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
            images_with_faces_train_data.append(face)

    # Foreach face append a random non face image to the images_without_faces_training_data to even the training data (Added the plus 1 to make up for margin errors)
    if GENERATE_NON_FACES:
        for i in range(len(image_labels) + random.randint(0, 1)):
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
                    non_face = image[rand_y:rand_y + sliding_window_size[1], rand_x:rand_x + sliding_window_size[0]]
                    non_face = cv.resize(non_face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
                    images_without_faces_train_data.append(non_face)
                    found_non_face = True

                if tries >= random_window_max_tries:
                    found_non_face = True

if GENERATE_FACES:
    for i, image in enumerate(images_with_faces_train_data):
        cv.imwrite("./positiveExamples/" + str(i) + ".jpg", image)

if GENERATE_NON_FACES:
    for i, image in enumerate(images_without_faces_train_data):
        cv.imwrite("./negativeExamples/" + str(i) + ".jpg", image)
