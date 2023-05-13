import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datasets.ds_config import DATASET_NAME, DATASETS_SETS


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        thickness = max(2, int(w / 275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image


# Function to plot images with the bounding boxes.
def plot(num_samples=4):
    path_to_datasets = os.path.abspath("datasets/") + '/' + DATASET_NAME + '/'
    for folder_path in DATASETS_SETS:

        image_paths = f'{path_to_datasets}/{folder_path}/images/'
        label_paths = f'{path_to_datasets}/{folder_path}/labels/'

        all_images = []
        all_images.extend(glob.glob(image_paths + '/*.jpg'))
        all_images.extend(glob.glob(image_paths + '/*.JPG'))

        all_images.sort()

        num_images = len(all_images)

        plt.figure(figsize=(14, 5))
        plt.suptitle(folder_path, fontsize=20, fontweight='bold')

        rows = (num_samples + 3) // 4
        cols = 4

        for i in range(num_samples):
            j = random.randint(0, num_images - 1)
            image_name = all_images[j]
            image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
            image = cv2.imread(all_images[j])
            with open(os.path.join(label_paths, image_name + '.txt'), 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()
                for label_line in label_lines:
                    label = label_line[0]
                    bbox_string = label_line[2:]
                    x_c, y_c, w, h = bbox_string.split(' ')
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)
            result_image = plot_box(image, bboxes, labels)

            # Add black border to the image
            result_image = cv2.copyMakeBorder(result_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Plot the image
            plt.subplot(rows, cols, i + 1)
            plt.subplots_adjust(top=0.9)
            plt.imshow(result_image[:, :, ::-1])
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# Function to collect the class counts from the annotations.
def collect_data_from_ann():
    path_to_datasets = os.path.abspath("datasets/") + '/' + DATASET_NAME + '/'
    class_counts = {i: 0 for i in range(10)}
    class_counts_folders = {}

    for folder_path in DATASETS_SETS:
        full_path = f'{path_to_datasets}/{folder_path}/labels/'
        class_counts_folder = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for filename in os.listdir(full_path):
            if filename.endswith('.txt'):
                annotation_path = os.path.join(full_path, filename)
                with open(annotation_path, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts_folder[class_id] += 1

        class_counts_folders[folder_path] = class_counts_folder
        for class_id in class_counts:
            class_counts[class_id] += class_counts_folder[class_id]

    return class_counts_folders


# Function to plot the class counts.
def plot_class_counts():
    classes = pd.DataFrame(collect_data_from_ann())
    class_counts_df = classes.copy()
    class_counts_df.rows = [str(i) for i in range(10)]
    class_counts_df['Total_of_class'] = class_counts_df.sum(axis=1)
    class_counts_df.loc['Total_digits'] = class_counts_df.sum()
    class_counts_df_rates = classes.copy()
    class_counts_df_rates = class_counts_df_rates.apply(lambda x: x / x.sum() * 100, axis=0)
    # class_counts_df_rates.loc['Total_digits'] = class_counts_df_rates.sum(axis=0)
    class_counts_df_rates['Total_of_class'] = class_counts_df_rates.mean(axis=1)
    # class_counts_df_rates
    sns.heatmap(class_counts_df_rates, annot=True, cmap='YlGnBu')

    # set axis labels
    plt.xlabel('Datasets')
    plt.ylabel('Classes')

    # set plot title
    plt.title('Class Frequencies as Percentage of Total')

    # show plot
    plt.show()


def check_datasets():
    path_to_datasets = os.path.abspath("datasets/") + '/' + DATASET_NAME + '/'
    # Iterate over datasets
    for folder_path in DATASETS_SETS:
        img_dir = f'{path_to_datasets}/{folder_path}/images/'
        ann_dir = f'{path_to_datasets}/{folder_path}/labels/'
        img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f.endswith(".jpg")]
        total_imgs = 0
        total_anns = 0
        total_classes = set()
        # Iterate over images
        for img_file in img_files:
            ann_file = os.path.join(ann_dir, img_file.replace(".jpg", ".txt"))
            # Read annotations file
            with open(ann_file, "r") as f:
                # Check that file ends with .jpg
                if not img_file.endswith(".jpg"):
                    continue
                # Update count of images
                total_imgs += 1
                ann_lines = f.readlines()
            num_anns = len(ann_lines)
            total_anns += num_anns
            # Iterate over annotations and count classes
            for ann in ann_lines:
                class_id = int(ann.strip().split()[0])
                total_classes.add(class_id)
        print(f"Dataset: {folder_path}")
        print(f"Total images: {total_imgs}")
        print(f"Total annotations (boxes): {total_anns}")
        print(f"Total classes: {len(total_classes)}")
        print("=" * 20)


def plot_hist():
    path_to_datasets = os.path.abspath("datasets/") + '/' + DATASET_NAME + '/'
    # Initialize an empty dictionary to store the data
    data_dict = {}
    # Loop over each folder path
    for folder_path in DATASETS_SETS:
        full_path = f'{path_to_datasets}/{folder_path}/labels/'
        # Initialize an empty dictionary for the current data set
        dataset_dict = {}
        # Loop over each file in the folder
        for filename in os.listdir(full_path):
            if filename.endswith('.txt'):
                annotation_path = os.path.join(full_path, filename)
                # Get the image name from the annotation file name
                image_name = os.path.splitext(filename)[0] + '.jpg'
                # Initialize a list to store the bounding boxes and classes for the current image
                bbox_list = []
                # Read the annotation file
                with open(annotation_path, 'r') as f:
                    for line in f:
                        # Get the class ID and bounding box coordinates
                        class_id, x, y, w, h = map(float, line.split())
                        # Add the class ID to the list
                        bbox_list.append(int(class_id))
                # Get the number of bounding boxes for the current image
                num_boxes = len(bbox_list)
                # Add the current image data to the dictionary for the current data set
                dataset_dict[image_name] = {'num_boxes': num_boxes, 'classes': bbox_list}
        # Add the dictionary for the current data set to the main dictionary
        data_dict[folder_path] = dataset_dict
    # ploting Number of (digits)boxes per image
    test_num_boxes = [data_dict['test'][img]['num_boxes'] for img in data_dict['test']]
    train_num_boxes = [data_dict['train'][img]['num_boxes'] for img in data_dict['train']]
    valid_num_boxes = [data_dict['valid'][img]['num_boxes'] for img in data_dict['valid']]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    fig.suptitle('Distribution of (digits)boxes per image in Different Sets', fontsize=14, fontweight='bold')
    sns.histplot(test_num_boxes, ax=axes[0], bins=range(11), stat='count')
    sns.histplot(train_num_boxes, ax=axes[1], bins=range(11), stat='count')
    sns.histplot(valid_num_boxes, ax=axes[2], bins=range(11), stat='count')
    # axes block
    axes[0].set_xlabel('Number of boxes')
    axes[1].set_xlabel('Number of boxes')
    axes[2].set_xlabel('Number of boxes')
    axes[0].set_ylabel('Quantity')
    axes[0].set_title('Test set')
    axes[1].set_title('Train set')
    axes[2].set_title('Validation set')
    # plotting
    plt.tight_layout()
    plt.show()
    # ploting Distribution of Unique Classes in Different Sets
    test_num_classes = [len(set(data_dict['test'][img]['classes'])) for img in data_dict['test']]
    train_num_classes = [len(set(data_dict['train'][img]['classes'])) for img in data_dict['train']]
    valid_num_classes = [len(set(data_dict['valid'][img]['classes'])) for img in data_dict['valid']]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    fig.suptitle('Distribution of Unique Classes in Different Sets', fontsize=14, fontweight='bold')
    sns.histplot(test_num_classes, ax=axes[0], bins=range(11), stat='count')
    axes[0].set_xticks(range(11))
    sns.histplot(train_num_classes, ax=axes[1], bins=range(11), stat='count')
    axes[1].set_xticks(range(11))
    sns.histplot(valid_num_classes, ax=axes[2], bins=range(11), stat='count')
    axes[2].set_xticks(range(11))

    axes[0].set_xlabel('Number of classes')
    axes[1].set_xlabel('Number of classes')
    axes[2].set_xlabel('Number of classes')
    axes[0].set_ylabel('Quantity')
    axes[0].set_title('Test set')
    axes[1].set_title('Train set')
    axes[2].set_title('Validation set')

    plt.tight_layout()
    plt.show()


def full_check_ds(num=4):
    """
    Function to check the dataset with all the function
    :param num: how many images to plot from each set

    """
    check_datasets()
    plot(num)
    plot_class_counts()
    plot_hist()
