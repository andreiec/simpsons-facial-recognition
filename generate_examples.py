import os
import cv2 as cv
import random

GENERATE_FACES = False
GENERATE_NON_FACES = False
GENERATE_CHARACTER_FACES = True

# Paths
TRAIN_IMAGES_PATH = ["./files/antrenare/bart/", "./files/antrenare/homer/", "./files/antrenare/lisa/", "./files/antrenare/marge/"]
TRAIN_IMAGES_LABELS = ["./files/antrenare/bart.txt", "./files/antrenare/homer.txt", "./files/antrenare/lisa.txt", "./files/antrenare/marge.txt"]

# Sliding window parameters
sliding_window_size = (36, 36)

# Data gathering parameters
image_resize_factor = 1
random_window_max_tries = 5

# Yellow filter
low_yellow = (19, 90, 190)
high_yellow = (90, 255, 255)

# Training data arrays
images_with_faces_train_data = []
images_without_faces_train_data = []

bart = []
homer = []
lisa = []
marge = []

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

    # Try to get around 30 negative examples per image
    if GENERATE_NON_FACES:
        for i in range(7):
            for j in range(4):
                # Select random coords
                rand_y, rand_x = random.randint(0, image.shape[0] - int(sliding_window_size[1] * (1.5**j) + 1)), random.randint(0, image.shape[1] - int(sliding_window_size[0] * (1.5**j) + 1))

                # Check if the square would intersect any face from label, if yes, break
                intersects = False
                for image_label in image_labels:
                    if check_if_rectangles_overlap((rand_x, rand_y), (rand_x + int(sliding_window_size[0] * (1.5**j) + 1), rand_y + int(sliding_window_size[1] * (1.5**j) + 1)), (image_label[0], image_label[1]), (image_label[2], image_label[3])):
                        intersects = True

                if not intersects:
                    negative_example = image[rand_y:rand_y + int(sliding_window_size[1] * (1.5**j) + 1), rand_x:rand_x + int(sliding_window_size[0] * (1.5**j) + 1)]
                    patch_hsv = cv.cvtColor(negative_example, cv.COLOR_BGR2HSV)
                    yellow_patch = cv.inRange(patch_hsv, low_yellow, high_yellow)

                    # Check if image contains some yellow, if yes, append to negative examples
                    if yellow_patch.mean() >= 50:
                        non_face = image[rand_y:rand_y + int(sliding_window_size[1] * (1.5**j) + 1), rand_x:rand_x + int(sliding_window_size[0] * (1.5**j) + 1)]
                        non_face = cv.resize(non_face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
                        images_without_faces_train_data.append(non_face)

    if GENERATE_CHARACTER_FACES:
        for image_label in image_labels:
            if image_label[-1] == "unknown":
                continue

            face = get_image_from_label(image, image_label)
            face = cv.resize(face, sliding_window_size)

            if image_label[-1] == "bart":
                bart.append(face)
            elif image_label[-1] == "homer":
                homer.append(face)
            elif image_label[-1] == "lisa":
                lisa.append(face)
            elif image_label[-1] == "marge":
                marge.append(face)

if GENERATE_FACES:
    for i, image in enumerate(images_with_faces_train_data):
        cv.imwrite("./positiveExamples/" + str(i) + ".jpg", image)

if GENERATE_NON_FACES:
    for i, image in enumerate(images_without_faces_train_data):
        cv.imwrite("./negativeExamples/" + str(i) + ".jpg", image)

if GENERATE_CHARACTER_FACES:
    for i, image in enumerate(bart):
        cv.imwrite("./separatedExamples/bart/" + str(i) + ".jpg", image)

    for i, image in enumerate(homer):
        cv.imwrite("./separatedExamples/homer/" + str(i) + ".jpg", image)

    for i, image in enumerate(lisa):
        cv.imwrite("./separatedExamples/lisa/" + str(i) + ".jpg", image)

    for i, image in enumerate(marge):
        cv.imwrite("./separatedExamples/marge/" + str(i) + ".jpg", image)
