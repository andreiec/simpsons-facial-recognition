import os
import time
import cv2 as cv
import numpy as np
import random
import pickle
import timeit

from skimage import data, exposure
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Constants
SAVE_MODEL = False
LOAD_MODEL = True
TRAIN_MODEL = False
LOAD_IMAGES = False
SCORE_MODEL = False
DISPLAY_SLIDING_WINDOW = False

# Paths
TRAIN_IMAGES_PATH = ["./files/antrenare/bart/", "./files/antrenare/homer/", "./files/antrenare/lisa/", "./files/antrenare/marge/"]
TRAIN_IMAGES_LABELS = ["./files/antrenare/bart.txt", "./files/antrenare/homer.txt", "./files/antrenare/lisa.txt", "./files/antrenare/marge.txt"]

POSITIVE_EXAMPLES_PATH = "./positiveExamples/"
NEGATIVE_EXAMPLES_PATH = "./negativeExamples/"
VALIDATION_PATH = "./files/validare/simpsons_validare/"
GROUND_TRUTH_PATH = "./files/validare/task1_gt.txt"

# Sliding window parameters
sliding_window_size = (36, 36)
sliding_window_step_size = 1

sliding_window_predict_size = (36, 64)
sliding_window_predict_step_size = 1

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


# Main function
def main():

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
        for file in os.listdir(POSITIVE_EXAMPLES_PATH):
            image = cv.imread(POSITIVE_EXAMPLES_PATH + file)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, face_hog_image = hog(image, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
            images_with_faces_train_data.append(face_hog_image.flatten())

        # Load negative examples files
        for file in os.listdir(NEGATIVE_EXAMPLES_PATH):
            image = cv.imread(NEGATIVE_EXAMPLES_PATH + file)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, non_face_hog_image = hog(image, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
            images_without_faces_train_data.append(non_face_hog_image.flatten())

        print("Images loaded!")

    # Build classifier labels
    train_y_faces = [1] * len(images_with_faces_train_data)
    train_y_non_faces = [0] * len(images_without_faces_train_data)
    # print(f"Faces: {len(train_y_faces)}, Non faces: {len(train_y_non_faces)}")
    train_y_faces.extend(train_y_non_faces)
    train_y = train_y_faces

    # Combine training data
    images_with_faces_train_data.extend(images_without_faces_train_data)
    train_x = np.array(images_with_faces_train_data)

    # Define classifier
    classifier = svm.SVC()

    if LOAD_IMAGES:
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    if TRAIN_MODEL:
        print("Training SVM..")

        # Train test split
        classifier.fit(X_train, y_train)

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

    # 92% for bart only dataset, 96% for all datasets
    if SCORE_MODEL:
        correct = 0
        predictions = classifier.predict(X_test)
        for i, prediction in enumerate(predictions):
            if prediction == y_test[i]:
                correct += 1

        print(f"Score on train test split:{correct / len(y_test)}")

    final_detections = []
    final_file_paths = []
    final_scores = []
    length_of_files = len(os.listdir(VALIDATION_PATH))

    for file_no, file in enumerate(os.listdir(VALIDATION_PATH)):
        start_time = timeit.default_timer()

        # Sliding window for image
        loaded_image = cv.imread('./files/validare/simpsons_validare/' + file)
        loaded_image_copy = loaded_image.copy()
        loaded_image = cv.cvtColor(loaded_image, cv.COLOR_BGR2GRAY)

        sliding_window_scale = 1.2
        # loaded_image = cv.blur(loaded_image, (3, 3), 1)
        # loaded_image = cv.fastNlMeansDenoising(loaded_image, None, 3, 7, 21)

        detections = []

        for scale_factor, image_resize in enumerate(image_pyramid(loaded_image, scale=sliding_window_scale, minsize=(80, 80))):
            for (x, y, window) in image_sliding_window(image_resize, sliding_window_predict_step_size, sliding_window_predict_size):
                if window.shape[0] != sliding_window_predict_size[1] or window.shape[1] != sliding_window_predict_size[0]:
                    continue

                window = cv.resize(window, sliding_window_size)
                _, predict_hog_window = hog(window, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
                predict_hog_window_np = np.array(predict_hog_window)
                predict_hog_window_np_flatten = predict_hog_window_np.flatten()
                # prediction = classifier.predict([predict_hog_window_np_flatten])
                score = classifier.decision_function([predict_hog_window_np_flatten])

                if score > 0:
                    if scale_factor > 0:
                        detections.append((x, y, x + window.shape[0] * int(sliding_window_scale ** scale_factor), y + window.shape[1] * int(sliding_window_scale ** scale_factor)))
                    else:
                        detections.append((x, y, x + window.shape[0], y + window.shape[1]))
                    # final_scores.append(score[0])

                # Display sliding window
                if DISPLAY_SLIDING_WINDOW:
                    clone = image_resize.copy()
                    cv.rectangle(clone, (x, y), (x + sliding_window_predict_size[0], y + sliding_window_predict_size[1]), (0, 255, 0), 2)
                    cv.imshow("Window", clone)
                    cv.waitKey(1)

        # print("Initial Detections:")
        # print(detections)

        # for detection in detections:
        #     cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 1)

        # for detection in detections:
        #     print(detection)
        #     cv.imshow("a", loaded_image[detection[2]:detection[3], detection[0]:detection[1]])
        #     cv.waitKey(0)

        # If there is only one detection save it as a final detection
        if len(detections) == 1:
            detection = detections[0]
            final_detections.append(detection)
            final_file_paths.append(file)
            # cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), 1)

            # Score detection again and save it
            detection_window = loaded_image[detection[1]:detection[3], detection[0]:detection[2]]
            window_resized = cv.resize(detection_window, sliding_window_size)
            _, predict_hog_window = hog(window_resized, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
            predict_hog_window_np = np.array(predict_hog_window)
            predict_hog_window_np_flatten = predict_hog_window_np.flatten()
            score = classifier.decision_function([predict_hog_window_np_flatten])
            final_scores.append(score[0])

        elif len(detections) > 1:
            # Else, group rectangles and save the rectangle as a final detecion
            grouped_rectangles_detections = cv.groupRectangles(detections, 1, 0.5)

            # If there are no grouped rectangles, but there are detections, mark every detection as final
            if len(grouped_rectangles_detections[0]) == 0:
                for detection in detections:
                    final_detections.append(detection)
                    final_file_paths.append(file)

                    # Score detection again and save it
                    detection_window = loaded_image[detection[1]:detection[3], detection[0]:detection[2]]
                    window_resized = cv.resize(detection_window, sliding_window_size)
                    _, predict_hog_window = hog(window_resized, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
                    predict_hog_window_np = np.array(predict_hog_window)
                    predict_hog_window_np_flatten = predict_hog_window_np.flatten()
                    score = classifier.decision_function([predict_hog_window_np_flatten])
                    final_scores.append(score[0])

                    # cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), 1)
            else:
                for detection in grouped_rectangles_detections[0]:
                    final_detections.append(detection.tolist())
                    final_file_paths.append(file)

                    # Score detection again and save it
                    detection_window = loaded_image[detection[1]:detection[3], detection[0]:detection[2]]
                    window_resized = cv.resize(detection_window, sliding_window_size)
                    _, predict_hog_window = hog(window_resized, pixels_per_cell=(6, 6), cell_block=(2, 2), visualize=True)
                    predict_hog_window_np = np.array(predict_hog_window)
                    predict_hog_window_np_flatten = predict_hog_window_np.flatten()
                    score = classifier.decision_function([predict_hog_window_np_flatten])
                    final_scores.append(score[0])

                    # cv.rectangle(loaded_image_copy, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), 1)
        else:
            print("No detection in file under")

        end_time = timeit.default_timer()
        print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.' % (file_no + 1, length_of_files, end_time - start_time))

    final_detections = np.asarray(final_detections)
    final_file_paths = np.asarray(final_file_paths)
    final_scores = np.asarray(final_scores)

    eval_detections(final_detections, final_scores, final_file_paths, GROUND_TRUTH_PATH)

    # cv.imshow("Final", loaded_image_copy)
    # cv.waitKey(0)

    # positive_scores = []
    # negative_scores = []
    #
    # scores = classifier.decision_function(X_train)
    #
    # for score in scores:
    #     if score > 0:
    #         positive_scores.append(score)
    #     else:
    #         negative_scores.append(score)
    #
    # plt.plot(np.sort(positive_scores))
    # plt.plot(np.zeros(len(negative_scores) + 20))
    # plt.plot(np.sort(negative_scores))
    # plt.xlabel('Nr example antrenare')
    # plt.ylabel('Scor clasificator')
    # plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
    # plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
    # plt.show()


if __name__ == "__main__":
    main()
