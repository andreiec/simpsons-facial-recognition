import os
import cv2 as cv
import numpy as np
import pickle
import timeit

from skimage.feature import hog
from sklearn import svm

import matplotlib.pyplot as plt

# Constants
SAVE_MODEL = False
LOAD_MODEL = True
TRAIN_MODEL = False
LOAD_IMAGES = False
DISPLAY_SLIDING_WINDOW = False

# Paths
TRAIN_IMAGES_PATH = ["./files/antrenare/bart/", "./files/antrenare/homer/", "./files/antrenare/lisa/", "./files/antrenare/marge/"]
TRAIN_IMAGES_LABELS = ["./files/antrenare/bart.txt", "./files/antrenare/homer.txt", "./files/antrenare/lisa.txt", "./files/antrenare/marge.txt"]

POSITIVE_EXAMPLES_PATH = "./positiveExamples/"
NEGATIVE_EXAMPLES_PATH = "./negativeExamples/"
VALIDATION_PATH = "./files/validare/simpsons_validare/"
GROUND_TRUTH_PATH = "./files/validare/task1_gt.txt"

# Sliding window parameters
sliding_window_size = (8, 8)
sliding_window_step_size = 1

train_window_size = (36, 36)
hog_pixels_per_cell = (6, 6)

# Save images paths and their labels
images = []
labels = {}

# Yellow filter
low_yellow = (20, 105, 105)
high_yellow = (35, 255, 255)


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


# Functii luate din codul de evaluare solutie
def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


# Functii luate din codul de evaluare solutie
def compute_average_precision(rec, prec):
    # functie adaptata din 2010 Pascal VOC development kit
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision


# Functii luate din codul de evaluare solutie
def eval_detections(detections, scores, file_names, ground_truth_path):
    ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
    ground_truth_file_names = np.array(ground_truth_file[:, 0])
    ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

    num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
    gt_exists_detection = np.zeros(num_gt_detections)
    # sorteazam detectiile dupa scorul lor
    sorted_indices = np.argsort(scores)[::-1]
    file_names = file_names[sorted_indices]
    scores = scores[sorted_indices]
    detections = detections[sorted_indices]

    num_detections = len(detections)
    true_positive = np.zeros(num_detections)
    false_positive = np.zeros(num_detections)
    duplicated_detections = np.zeros(num_detections)

    for detection_idx in range(num_detections):
        indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

        gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
        bbox = detections[detection_idx]
        max_overlap = -1
        index_max_overlap_bbox = -1
        for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
            overlap = intersection_over_union(bbox, gt_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                index_max_overlap_bbox = indices_detections_on_image[gt_idx]

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if max_overlap >= 0.3:
            if gt_exists_detection[index_max_overlap_bbox] == 0:
                true_positive[detection_idx] = 1
                gt_exists_detection[index_max_overlap_bbox] = 1
            else:
                false_positive[detection_idx] = 1
                duplicated_detections[detection_idx] = 1
        else:
            false_positive[detection_idx] = 1

    cum_false_positive = np.cumsum(false_positive)
    cum_true_positive = np.cumsum(true_positive)

    rec = cum_true_positive / num_gt_detections
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('All faces: average precision %.3f' % average_precision)
    plt.savefig('precizie_medie_all_faces.png')
    plt.show()


def non_maximal_suppression(image_detections, image_scores, image_size):
    # xmin, ymin, xmax, ymax
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


# Main function
def main():
    # Training data arrays
    images_with_faces_train_data = []
    images_without_faces_train_data = []

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

    if LOAD_IMAGES:
        print("Loading images..")

        # Load positive examples files
        positive_descriptors_path = "./descriptori/positive/positive.npy"
        if os.path.exists(positive_descriptors_path):
            images_with_faces_train_data = np.load(positive_descriptors_path)
        else:
            for file in os.listdir(POSITIVE_EXAMPLES_PATH):
                image = cv.imread(POSITIVE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                face_hog_image = hog(image, pixels_per_cell=(6, 6), cells_per_block=(2, 2), feature_vector=True)
                images_with_faces_train_data.append(face_hog_image)

                face_hog_image = hog(np.fliplr(image), pixels_per_cell=(6, 6), cells_per_block=(2, 2), feature_vector=True)
                images_with_faces_train_data.append(face_hog_image)

            images_with_faces_train_data = np.array(images_with_faces_train_data)
            np.save(positive_descriptors_path, images_with_faces_train_data)

        # Load negative examples files
        negative_descriptors_path = "./descriptori/negative/negative.npy"
        if os.path.exists(negative_descriptors_path):
            images_without_faces_train_data = np.load(negative_descriptors_path)
        else:
            for file in os.listdir(NEGATIVE_EXAMPLES_PATH):
                image = cv.imread(NEGATIVE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                non_face_hog_image = hog(image, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
                images_without_faces_train_data.append(non_face_hog_image)

                non_face_hog_image = hog(np.fliplr(image), pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
                images_without_faces_train_data.append(non_face_hog_image)

            images_without_faces_train_data = np.array(images_without_faces_train_data)
            np.save(negative_descriptors_path, images_without_faces_train_data)

        print("Images loaded!")

        # Build classifier labels
        print(f"Faces: {len(images_with_faces_train_data)}, Non faces: {len(images_without_faces_train_data)}")
        train_y = np.concatenate((np.ones(images_with_faces_train_data.shape[0]), np.zeros(images_without_faces_train_data.shape[0])))

        # Combine training data
        train_x = np.concatenate((np.squeeze(images_with_faces_train_data), np.squeeze(images_without_faces_train_data)), axis=0)

    classifier = svm.LinearSVC(C=1)

    if TRAIN_MODEL:
        print("Training SVM..")

        # Train test split
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

    final_detections = []
    final_file_paths = []
    final_scores = []
    length_of_files = len(os.listdir(VALIDATION_PATH))

    for file_no, file in enumerate(os.listdir(VALIDATION_PATH)):
        start_time = timeit.default_timer()

        # Sliding window for image
        loaded_image = cv.imread('./files/validare/simpsons_validare/' + file)
        loaded_image_copy = loaded_image.copy()
        loaded_image_hsv = cv.cvtColor(loaded_image, cv.COLOR_BGR2HSV)
        loaded_image_hsv_yellow = cv.inRange(loaded_image_hsv, low_yellow, high_yellow)

        sliding_window_scale = 1.2
        detections = []
        scores = []

        scale = 0.18
        scale_y = 0.15

        # Sliding window
        while scale <= 1 and scale_y <= 1:
            image_resize = cv.resize(loaded_image, (0, 0), fx=scale, fy=scale_y)
            image_resize_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
            image_resize_hog = hog(image_resize_gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=False)

            number_of_cols = image_resize.shape[1] // hog_pixels_per_cell[0] - 1
            number_of_rows = image_resize.shape[0] // hog_pixels_per_cell[0] - 1
            number_of_cell_in_template = train_window_size[0] // hog_pixels_per_cell[0] - 1

            # Slide across hog cells
            for y in range(0, number_of_rows - number_of_cell_in_template, sliding_window_step_size):
                for x in range(0, number_of_cols - number_of_cell_in_template, sliding_window_step_size):
                    x_min = int(x * hog_pixels_per_cell[1] * 1 // scale)
                    y_min = int(y * hog_pixels_per_cell[0] // scale_y)
                    x_max = int((x * hog_pixels_per_cell[1] + train_window_size[1]) * 1 // scale)
                    y_max = int((y * hog_pixels_per_cell[0] + train_window_size[0]) * 1 // scale_y)

                    # Check if image contains some yellow
                    if loaded_image_hsv_yellow[y_min:y_max, x_min:x_max].mean() >= 70:
                        descriptor = image_resize_hog[y:y + number_of_cell_in_template, x:x + number_of_cell_in_template].flatten()
                        score = np.dot(descriptor, classifier.coef_.T) + classifier.intercept_[0]

                        # Append score
                        if score[0] > 0:
                            scores.append(score[0])
                            detections.append((x_min, y_min, x_max, y_max))

                    # Display sliding window
                    if DISPLAY_SLIDING_WINDOW:
                        clone = image_resize.copy()
                        cv.rectangle(clone, (x, y), (x + sliding_window_size[1], y + sliding_window_size[0]), (0, 255, 0), 2)
                        cv.imshow("Window", clone)
                        cv.waitKey(1)

            scale *= 1.04
            scale_y *= 1.04

        # for detection in detections:
            # cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 1)

        if len(detections) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(detections), np.array(scores), loaded_image.shape)

            for detection in image_detections:
                cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), 1)
                final_detections.append(detection)
                final_file_paths.append(file)

            for score in image_scores:
                final_scores.append(score)

        end_time = timeit.default_timer()
        print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.' % (file_no + 1, length_of_files, end_time - start_time))

        # cv.imshow("a", loaded_image_copy)
        # cv.waitKey(0)

    final_detections = np.asarray(final_detections)
    final_file_paths = np.asarray(final_file_paths)
    final_scores = np.asarray(final_scores)

    print(f"Detections: {len(final_detections)}")

    eval_detections(final_detections, final_scores, final_file_paths, GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main()
