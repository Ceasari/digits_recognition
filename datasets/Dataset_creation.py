
import gzip
import os
import random
import shutil
import struct

from datasets.ds_config import *
import cv2

import yaml
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def flip_hor(img):
    if random.random() < H_FLIP_CHANCE:
        img = cv2.flip(img, 1)
    return img


def rotate_image(image):
    if random.random() < ROTATE_CHANCE:
        angle = random.randint(0, MAX_ROTATE_ANGLE)
        direction = random.choice([-1, 1])

        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle * direction, 1)
        image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))

    return image


def noisy(img):
    height, width = img.shape[:2]

    # Compute maximum number of dots
    max_dots = int(height * width * MAX_NOICE)

    # Generate random dots
    for i in range(random.randint(0, max_dots)):
        # Generate random coordinates
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Generate random gray intensity
        gray_intensity = random.randint(0, 255)

        # Add dot to image
        img[y, x] = [gray_intensity, gray_intensity, gray_intensity]
    return img


def compute_iou(box1, box2):
    # xmin, ymin, xmax, ymax
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax:
        return 0
    return ((xmax - xmin) * (ymax - ymin)) / (a1 + a2)


def save_images(images, labels, folder):
    os.makedirs(folder, exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        filename = f"/{label}_{i:05d}.pgm"
        with open(folder + filename, 'wb') as f:
            f.write(b'P5\n28 28\n255\n')
            f.write(image)


def make_image(data, image_path, ratio=1):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    idn = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.bitwise_not(image)
    image = flip_hor(image)
    image = rotate_image(image)
    image = cv2.resize(image, (int(28 * ratio), int(28 * ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, SIZE - w, 1)[0]
        ymin = np.random.randint(0, SIZE - h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(idn)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    # cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    # blank = noisy(blank)
    return blank


def yalm_creation(save_abs_path, name):
    yaml_data = {
        "nc": CLASSES,
        "test": f"{save_abs_path}test/images",
        "train": f"{save_abs_path}train/images",
        "val": f"{save_abs_path}valid/images",
        "names": CLASSES_NAMES
    }

    with open(save_abs_path + '/' + name + '.yaml', 'w') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False)


def create_dataset() -> None:
    """
       Generates a custom dataset of images and corresponding bounding box annotations, based on the MNIST dataset.
       Uses Parameters from ds_config.py:
       DATASET_NAME (str): Name of the dataset, used for the folder name to save the dataset.
       TOTAL_SET (int): Total number of images to generate in the dataset. Defaults to 1000.
       SPLIT (list[float]): List of three float values indicating the percentage of images to allocate to the train,
       validation, and test sets, respectively. Defaults to [0.7, 0.15, 0.15].
       """
    if DATASET_NAME is None or DATASET_NAME == '':
        raise ValueError("The 'DATASET_NAME' argument must be a string.")
    # create folders for train and test sets
    save_abs_path = os.path.abspath("datasets/") + '/' + DATASET_NAME + '/'
    abs_path_with_mnist = os.path.abspath('datasets/') + '/mnist' + '/'
    train_folder = abs_path_with_mnist + 'train_origin/'
    test_folder = abs_path_with_mnist + 'test_origin/'
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    if not os.path.exists(save_abs_path):
        os.makedirs(save_abs_path)

    # # download and extract MNIST dataset
    url = 'http://yann.lecun.com/exdb/mnist/'
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    image_sizes = [3, 6, 3]
    if len(SPLIT) != 3:
        raise ValueError("The 'SPLIT' argument must be a list of three float values.")
    for i in SPLIT:
        if i < 0 or i > 1:
            raise ValueError("The values in the 'SPLIT' argument must be between 0 and 1.")

    if not os.path.exists(train_folder + 'train-images-idx3-ubyte.gz') and not os.path.exists(
            train_folder + 'train-labels-idx1-ubyte.gz') and not os.path.exists(
        test_folder + 't10k-images-idx3-ubyte.gz') and not os.path.exists(
        test_folder + 't10k-labels-idx1-ubyte.gz'):

        for filename in filenames:
            print(f"Downloading {filename}...")
            os.system(f"curl -O {url}/{filename}")
            os.system(f"gunzip -f {filename}")
        if os.path.exists("train-images-idx3-ubyte.gz"):
            shutil.move(f"{os.getcwd()}/train-images-idx3-ubyte.gz", train_folder)
        if os.path.exists("train-labels-idx1-ubyte"):
            shutil.move(f"{os.getcwd()}/train-labels-idx1-ubyte", train_folder)

        if os.path.exists("train-labels-idx1-ubyte.gz"):
            shutil.move(f"{os.getcwd()}/train-labels-idx1-ubyte.gz", test_folder)
        if os.path.exists("train-labels-idx1-ubyte"):
            shutil.move(f"{os.getcwd()}/train-labels-idx1-ubyte", test_folder)

        if os.path.exists("t10k-images-idx3-ubyte.gz"):
            shutil.move(f"{os.getcwd()}/t10k-images-idx3-ubyte.gz", test_folder)
        if os.path.exists("t10k-images-idx3-ubyte"):
            shutil.move(f"{os.getcwd()}/t10k-images-idx3-ubyte", test_folder)

        if os.path.exists("t10k-labels-idx1-ubyte.gz"):
            shutil.move(f"{os.getcwd()}/t10k-labels-idx1-ubyte.gz", test_folder)
        if os.path.exists("t10k-labels-idx1-ubyte"):
            shutil.move(f"{os.getcwd()}/t10k-labels-idx1-ubyte", test_folder)


    # Load train images
    if not os.path.exists(abs_path_with_mnist + 'train_pgm') or len(
            os.listdir(abs_path_with_mnist + 'train_pgm/')) != 60000:
        with gzip.open(train_folder + 'train-images-idx3-ubyte.gz', 'rb') as f:
            magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        # Load train labels
        with gzip.open(train_folder + 'train-labels-idx1-ubyte.gz', 'rb') as f:
            _, _ = np.frombuffer(f.read(8), dtype=np.uint32)
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Save train images as PGM files
        print("Saving train images as PGM files...")
        save_images(train_images, train_labels, abs_path_with_mnist + 'train')

    # Load test images
    if not os.path.exists(abs_path_with_mnist + 'test') or len(
            os.listdir(abs_path_with_mnist + 'test/')) != 10000:
        with gzip.open(test_folder + 't10k-images-idx3-ubyte.gz', 'rb') as f:
            magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        # Load test labels
        with gzip.open(test_folder + 't10k-labels-idx1-ubyte.gz', 'rb') as f:
            _, _ = np.frombuffer(f.read(8), dtype=np.uint32)
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Save test images as PGM files
        print("Saving test images as PGM files...")
        save_images(test_images, test_labels, abs_path_with_mnist + 'test')

    # move valid images to valid folder
    os.makedirs(abs_path_with_mnist + '/valid/', exist_ok=True)
    if len(os.listdir(abs_path_with_mnist + '/valid/')) <= 8000:
        class_counts = {str(i): 0 for i in range(10)}
        for filename in os.listdir(abs_path_with_mnist + '/train/'):
            if not filename.endswith('.pgm'):
                continue

            # extract the first digit from the filename
            digit_class = filename[0]

            # check if we've already copied enough of this class
            if class_counts[digit_class] >= 10000 / 10:
                continue

            # determine the full path to the source and destination files
            src_path = os.path.join(abs_path_with_mnist + '/train/', filename)
            dest_path = os.path.join(abs_path_with_mnist + '/valid/', filename)

            # copy the file to the appropriate class folder in valid
            shutil.copyfile(src_path, dest_path)

            # update the counter for this class
            class_counts[digit_class] += 1

    images_num_train = int(SPLIT[0] * TOTAL_SET)
    images_num_valid = int(SPLIT[1] * TOTAL_SET)
    images_num_test = int(SPLIT[2] * TOTAL_SET)

    for file in DATASETS_SETS:
        images_path = save_abs_path + file + "/" + "images"
        labels_path = save_abs_path + file + "/" + "labels"

        if file == 'train':
            images_num = images_num_train
        if file == 'valid':
            images_num = images_num_valid
        if file == 'test':
            images_num = images_num_test

        if os.path.exists(images_path):
            shutil.rmtree(images_path)
        os.makedirs(images_path, exist_ok=True)

        if os.path.exists(labels_path):
            shutil.rmtree(labels_path)
        os.makedirs(labels_path, exist_ok=True)

        image_paths = [abs_path_with_mnist + file + "/" + image_name
                       for image_name in os.listdir(abs_path_with_mnist + file)]

        image_num = 0
        while image_num < images_num:
            image_path = os.path.realpath(os.path.join(images_path, "%06d.jpg" % (image_num + 1)))
            blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
            data = [blanks, bboxes, labels]
            bboxes_num = 0

            # add digits to the image
            for i in range(len(RATIOS)):
                n = random.randint(0, image_sizes[i])
                if n != 0:
                    bboxes_num += 1
                for _ in range(n):
                    ratio = random.choice(RATIOS[i])
                    idx = random.randint(0, len(image_paths) - 1)
                    data[0] = make_image(data, image_paths[idx], ratio)
                    data[0] = noisy(data[0])

            if bboxes_num == 0:
                continue
            cv2.imwrite(image_path, data[0])

            # create label file for the image
            label_file = os.path.realpath(os.path.join(labels_path, os.path.basename(image_path)[:-4] + '.txt'))
            with open(label_file, 'w') as label_f:
                lines = []
                for i in range(len(labels)):
                    if i == 0:
                        continue
                    xmin, ymin, xmax, ymax = bboxes[i]
                    width = xmax - xmin
                    height = ymax - ymin
                    x_center = xmin + (width / 2)
                    y_center = ymin + (height / 2)
                    normalized_x_center = x_center / SIZE
                    normalized_y_center = y_center / SIZE
                    normalized_width = width / SIZE
                    normalized_height = height / SIZE
                    class_ind = str(labels[i])
                    lines.append(
                        '{} {} {} {} {}'.format(class_ind, normalized_x_center, normalized_y_center, normalized_width,
                                                normalized_height))
                label_f.write('\n'.join(lines))
            image_num += 1

    yalm_creation(save_abs_path, DATASET_NAME)

    print("train : \t|", f"{images_num_train}\t|")
    print("valid : \t|", f"{images_num_valid}\t|")
    print("test : \t\t|", f"{images_num_test}\t|")
    print("Datasets are created successfully")


if __name__ == '__main__':
    create_dataset()
