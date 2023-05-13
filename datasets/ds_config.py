#================================================================

# Dataset options
DATASET_NAME            = 'new_data'  # Name of the dataset, used for the folder name to save the dataset
TOTAL_SET               = 10  # Total number of images to create
SPLIT                   = [0.7, 0.15, 0.15]  # SPLIT of the dataset (train, validation, test)
SIZE                    = 640  # size of the images (square)
RATIOS                  = [[0.5, 0.8], [1., 1.5, 2.], [3., 4.]]  # variety of digits sizes (small, medium, big objects)
MAX_NOICE               = 0.02  # noice level on each image
H_FLIP_CHANCE           = 0.15  # horizontal flip chance
MAX_ROTATE_ANGLE        = 30  # max rotate angle
ROTATE_CHANCE           = 0.7  # rotate chance


#================================================================

# YAML options
CLASSES                 = 10  # quantity of classes
CLASSES_NAMES           = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # name of the classes

#================================================================

# Options not to changing
DATASETS_SETS           = ['train', 'valid', 'test']  # datasets sets