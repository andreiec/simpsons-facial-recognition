import os
import cv2 as cv
import numpy as np
import pickle
import timeit

from skimage.feature import hog
from sklearn import svm


# Constants
LOAD_IMAGES = False
LOAD_FACE_DETECTOR_MODEL = True
TRAIN_MODEL = False
SAVE_MODEL = False
LOAD_MODEL = True
SAVE_FILES = True

# Paths
TRAIN_IMAGES_PATH = ["./files/antrenare/bart/", "./files/antrenare/homer/", "./files/antrenare/lisa/", "./files/antrenare/marge/"]
TRAIN_IMAGES_LABELS = ["./files/antrenare/bart.txt", "./files/antrenare/homer.txt", "./files/antrenare/lisa.txt", "./files/antrenare/marge.txt"]

VALIDATION_PATH = "./files/validare/simpsons_validare/"
GROUND_TRUTH_PATHS = ["./files/validare/task2_bart_gt.txt", "./files/validare/task2_homer_gt.txt", "./files/validare/task2_lisa_gt.txt", "./files/validare/task2_marge_gt.txt"]

BART_EXAMPLES_PATH = "./separatedExamples/bart/"
HOMER_EXAMPLES_PATH = "./separatedExamples/homer/"
LISA_EXAMPLES_PATH = "./separatedExamples/lisa/"
MARGE_EXAMPLES_PATH = "./separatedExamples/marge/"

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
def non_maximal_suppression(image_detections, image_scores, image_size):

    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]

    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]

    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3

    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False

    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


# Main function
def main():

    # Training data arrays
    train_images_bart = []
    train_images_homer = []
    train_images_lisa = []
    train_images_marge = []

    if LOAD_IMAGES:
        print("Loading images..")

        # Load bart faces
        bart_path = "./descriptori/characters/bart.npy"

        if os.path.exists(bart_path):
            train_images_bart = np.load(bart_path)
        else:
            for file in os.listdir(BART_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(BART_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                train_images_bart.append(image)

                # Flip image to do small augmentation and append it
                train_images_bart.append(np.fliplr(image))

            train_images_bart = np.array(train_images_bart)
            np.save(bart_path, train_images_bart)

        # Load homer faces
        homer_path = "./descriptori/characters/homer.npy"

        if os.path.exists(homer_path):
            train_images_homer = np.load(homer_path)
        else:
            for file in os.listdir(HOMER_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(HOMER_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                train_images_homer.append(image)

                # Flip image to do small augmentation and append it
                train_images_homer.append(np.fliplr(image))

            train_images_homer = np.array(train_images_homer)
            np.save(homer_path, train_images_homer)

        # Load lisa faces
        lisa_path = "./descriptori/characters/lisa.npy"

        if os.path.exists(lisa_path):
            train_images_lisa = np.load(lisa_path)
        else:
            for file in os.listdir(LISA_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(LISA_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                train_images_lisa.append(image)

                # Flip image to do small augmentation and append it
                train_images_lisa.append(np.fliplr(image))

            train_images_lisa = np.array(train_images_lisa)
            np.save(lisa_path, train_images_lisa)

        # Load marge faces
        marge_path = "./descriptori/characters/marge.npy"

        if os.path.exists(marge_path):
            train_images_marge = np.load(marge_path)
        else:
            for file in os.listdir(MARGE_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(MARGE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                train_images_marge.append(image)

                # Flip image to do small augmentation and append it
                train_images_marge.append(np.fliplr(image))

            train_images_marge = np.array(train_images_marge)
            np.save(marge_path, train_images_marge)

        print("Images loaded!")

        # Build classifier labels
        print(f"Bart: {len(train_images_bart)}, Homer: {len(train_images_homer)}, Lisa: {len(train_images_lisa)}, Merge: {len(train_images_marge)}")

        # Define labels for each character
        bart_y = [0] * len(train_images_bart)
        homer_y = [1] * len(train_images_homer)
        lisa_y = [2] * len(train_images_lisa)
        marge_y = [3] * len(train_images_marge)

        # Combine labels
        train_y = np.array(bart_y + homer_y + lisa_y + marge_y)

        # Combine training data
        train_x = np.concatenate((train_images_bart.flatten().reshape(len(bart_y), 1296), train_images_homer.flatten().reshape(len(homer_y), 1296), train_images_lisa.flatten().reshape(len(lisa_y), 1296), train_images_marge.flatten().reshape(len(marge_y), 1296)), axis=0)

    # classifier = MLPClassifier(random_state=1, max_iter=300)
    classifier = svm.SVC(C=10)

    # Define model for face classification and train test split
    if TRAIN_MODEL:
        print("Training SVC..")
        classifier.fit(train_x, train_y)
        print("SVC Trained!")

    if SAVE_MODEL:
        print("Saving SVC..")
        filename = 'finalized_model_task2.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        print("MLP Saved!")

    if LOAD_MODEL:
        print("Loading SVC..")
        filename = 'finalized_model_task2.sav'
        classifier = pickle.load(open(filename, 'rb'))
        print("SVC Loaded!")

    # Load SVM from task 1
    face_detector_SVC = svm.LinearSVC(C=1)

    if LOAD_FACE_DETECTOR_MODEL:
        print("Loading Face Detector..")
        filename = 'finalized_model_task1.sav'

        try:
            face_detector_SVC = pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            print("Could not find classifier")
            return

        print("Face Detector Loaded!")

    final_detections = []
    final_file_paths = []
    final_classifications = []
    final_scores = []
    length_of_files = len(os.listdir(VALIDATION_PATH))

    for file_no, file in enumerate(os.listdir(VALIDATION_PATH)):
        start_time = timeit.default_timer()

        # Sliding window for image
        loaded_image = cv.imread('./files/validare/simpsons_validare/' + file)
        loaded_image_hsv = cv.cvtColor(loaded_image, cv.COLOR_BGR2HSV)
        loaded_image_hsv_yellow = cv.inRange(loaded_image_hsv, low_yellow, high_yellow)

        detections = []
        scores = []

        scale_x = 0.15
        scale_y = 0.12

        # Sliding window
        while scale_x <= 1.75 and scale_y <= 1.75:
            image_resize = cv.resize(loaded_image, (0, 0), fx=scale_x, fy=scale_y)
            image_resize_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
            image_resize_hog = hog(image_resize_gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=False)

            number_of_cols = image_resize.shape[1] // hog_pixels_per_cell[0] - 1
            number_of_rows = image_resize.shape[0] // hog_pixels_per_cell[0] - 1
            number_of_cell_in_template = train_window_size[0] // hog_pixels_per_cell[0] - 1

            # Slide across hog cells
            for y in range(0, number_of_rows - number_of_cell_in_template, sliding_window_step_size):
                for x in range(0, number_of_cols - number_of_cell_in_template, sliding_window_step_size):
                    x_min = int(x * hog_pixels_per_cell[1] * 1 // scale_x)
                    y_min = int(y * hog_pixels_per_cell[0] // scale_y)
                    x_max = int((x * hog_pixels_per_cell[1] + train_window_size[1]) * 1 // scale_x)
                    y_max = int((y * hog_pixels_per_cell[0] + train_window_size[0]) * 1 // scale_y)

                    # Check if image contains some yellow
                    if loaded_image_hsv_yellow[y_min:y_max, x_min:x_max].mean() >= 70:
                        score = np.dot(image_resize_hog[y:y + number_of_cell_in_template, x:x + number_of_cell_in_template].flatten(), face_detector_SVC.coef_.T) + face_detector_SVC.intercept_[0]

                        # Append score
                        if score[0] > 0:
                            scores.append(score[0])
                            detections.append((x_min, y_min, x_max, y_max))

            scale_x *= 1.02
            scale_y *= 1.02

        # If there is a detection save it, if there are multiple detections run non_maximal_suppression to get the greatest square
        if len(detections) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(detections), np.array(scores), loaded_image.shape)

            # Save final detections and file paths
            for detection in image_detections:
                # Get detection window from image to predict it
                classify_detection = cv.cvtColor(loaded_image[detection[1]:detection[3], detection[0]:detection[2]], cv.COLOR_BGR2GRAY)
                classify_detection = cv.resize(classify_detection, train_window_size)
                classify_detection = classify_detection.flatten()

                # Append detection
                final_classifications.append(classifier.predict([classify_detection])[0])
                final_detections.append(detection)
                final_file_paths.append(file)

            # Save scores
            for score in image_scores:
                final_scores.append(score)

        end_time = timeit.default_timer()

        if len(detections) > 0:
            print(f'Time to process test image {file_no + 1:3}/{length_of_files},    with detecion, is {end_time - start_time:f} sec.')
        else:
            print(f'Time to process test image {file_no + 1:3}/{length_of_files}, without detecion, is {end_time - start_time:f} sec.')

    # Convert to numpy array final lists
    final_detections = np.asarray(final_detections)
    final_file_paths = np.asarray(final_file_paths)
    final_scores = np.asarray(final_scores)
    final_classifications = np.asarray(final_classifications)

    print(f"Total detections: {len(final_detections)}")

    bart_detections = []
    bart_paths = []
    bart_scores = []

    homer_detections = []
    homer_paths = []
    homer_scores = []

    lisa_detections = []
    lisa_paths = []
    lisa_scores = []

    marge_detections = []
    marge_paths = []
    marge_scores = []

    # Split classifications
    for i, classification in enumerate(final_classifications):
        if classification == 0:
            bart_detections.append(final_detections[i])
            bart_paths.append(final_file_paths[i])
            bart_scores.append(final_scores[i])
        elif classification == 1:
            homer_detections.append(final_detections[i])
            homer_paths.append(final_file_paths[i])
            homer_scores.append(final_scores[i])
        elif classification == 2:
            lisa_detections.append(final_detections[i])
            lisa_paths.append(final_file_paths[i])
            lisa_scores.append(final_scores[i])
        elif classification == 3:
            marge_detections.append(final_detections[i])
            marge_paths.append(final_file_paths[i])
            marge_scores.append(final_scores[i])

    # Convert to numpy arrays
    bart_detections = np.array(bart_detections)
    bart_paths = np.array(bart_paths)
    bart_scores = np.array(bart_scores)

    homer_detections = np.array(homer_detections)
    homer_paths = np.array(homer_paths)
    homer_scores = np.array(homer_scores)

    lisa_detections = np.array(lisa_detections)
    lisa_paths = np.array(lisa_paths)
    lisa_scores = np.array(lisa_scores)

    marge_detections = np.array(marge_detections)
    marge_paths = np.array(marge_paths)
    marge_scores = np.array(marge_scores)

    # Evaluate detections
    # eval_detections(bart_detections, bart_scores, bart_paths, GROUND_TRUTH_PATHS[0], 'Bart')
    # eval_detections(homer_detections, homer_scores, homer_paths, GROUND_TRUTH_PATHS[1], 'Homer')
    # eval_detections(lisa_detections, lisa_scores, lisa_paths, GROUND_TRUTH_PATHS[2], 'Lisa')
    # eval_detections(marge_detections, marge_scores, marge_paths, GROUND_TRUTH_PATHS[3], 'Marge')

    if SAVE_FILES:
        np.save("./Constantinescu_Andrei-Eduard_344/task2/detections_bart.npy", bart_detections)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/file_names_bart.npy", bart_paths)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/scores_bart.npy", bart_scores)

        np.save("./Constantinescu_Andrei-Eduard_344/task2/detections_homer.npy", homer_detections)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/file_names_homer.npy", homer_paths)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/scores_homer.npy", homer_scores)

        np.save("./Constantinescu_Andrei-Eduard_344/task2/detections_lisa.npy", lisa_detections)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/file_names_lisa.npy", lisa_paths)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/scores_lisa.npy", lisa_scores)

        np.save("./Constantinescu_Andrei-Eduard_344/task2/detections_marge.npy", marge_detections)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/file_names_marge.npy", marge_paths)
        np.save("./Constantinescu_Andrei-Eduard_344/task2/scores_marge.npy", marge_scores)


if __name__ == "__main__":
    main()
