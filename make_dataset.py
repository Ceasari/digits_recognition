from datasets.Dataset_creation import create_dataset
from datasets.Dataset_utils import full_check_ds
import os
from datasets.ds_config import DATASET_NAME

if __name__ == '__main__':
    create_dataset()
    full_check_ds()

