# IMPORT LIBRARIES
import os
import my_utils

# SET PARAMETERS
CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH, 'data')
ALL_IMAGES_PATH = os.path.join(DATA_PATH, 'images')
LABEL_CSV_FILE_LOCATION = os.path.join(DATA_PATH, 'labels.csv')
SPLITTED_IMAGES_PATH = os.path.join(DATA_PATH, 'splitted_images')  # folder to create
TRAINING_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'training\\')  # folder to create
VALIDATION_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'validation\\')  # folder to create
validation_rate = 0.2
seed = 33


my_utils.label_data(ALL_IMAGES_PATH, os.path.join(DATA_PATH, 'train.csv'))
my_utils.split_data(SPLITTED_IMAGES_PATH, ALL_IMAGES_PATH, validation_rate, seed=seed)
my_utils.balance_classes(TRAINING_DATA_PATH, seed=seed)