# IMPORT LIBRARIES
import os
import errno
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential

# SET PARAMETERS
CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH, 'data')
ALL_IMAGES_PATH = os.path.join(DATA_PATH, 'train_images')
LABEL_CSV_FILE_LOCATION = os.path.join(DATA_PATH, 'train.csv')
SPLITTED_IMAGES_PATH = os.path.join(DATA_PATH, 'splitted_images')
TRAINING_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'training\\')
VALIDATION_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'validation\\')
validation_rate = 0.2
training_batch_size = 128
random.seed(33)


# CREATE USEFUL FUNCTION
def delete_create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        shutil.rmtree(path)
        os.mkdir(path)


# CREATE FOLDERS ACCORDING TO LABELS
def label_data(folder_path, label_csv_file_location):
    label_table = pd.read_csv(label_csv_file_location, sep=',')
    labels = np.unique(label_table['label'])
    for label in labels:
        delete_create_folder(os.path.join(folder_path, label))
    for row in label_table.itertuples(index=False):
        try:
            shutil.move(os.path.join(folder_path, row[0]), os.path.join(folder_path, str(row[1]) + '\\' + row[0]))
        finally:
            pass

label_data(ALL_IMAGES_PATH, os.path.join(DATA_PATH, 'train.csv'))

# SPLIT DATA IN TRAINING AND VALIDATION
def split_data(split_data_folder_path, source_data_path, val_rate=0.2):
    delete_create_folder(split_data_folder_path)
    delete_create_folder(os.path.join(split_data_folder_path, 'training'))
    delete_create_folder(os.path.join(split_data_folder_path, 'validation'))
    folders_list = os.listdir(source_data_path)
    for folder in folders_list:
        file_list = os.listdir(os.path.join(source_data_path, folder))
        validation_file_list = random.sample(file_list, round(val_rate * len(file_list)))
        delete_create_folder(os.path.join(split_data_folder_path, 'training\\' + folder))
        delete_create_folder(os.path.join(split_data_folder_path, 'validation\\' + folder))
        folder_path = os.path.join(source_data_path, folder)
        for file in file_list:
            if file in validation_file_list:
                shutil.copyfile(os.path.join(folder_path, file),
                                os.path.join(split_data_folder_path, 'validation\\' + folder + '\\' + file))
            else:
                shutil.copyfile(os.path.join(folder_path, file),
                                os.path.join(split_data_folder_path, 'training\\' + folder + '\\' + file))


split_data(SPLITTED_IMAGES_PATH, ALL_IMAGES_PATH, validation_rate)

# BALANCE UNBALANCED CLASSES BY OVERSAMPLING
def balance_classes(data_to_balance_path):
    folders_list = os.listdir(data_to_balance_path)
    samples_num_list = [len(os.listdir(os.path.join(data_to_balance_path, folder))) for folder in folders_list]
    max_samples_num = max(samples_num_list)
    for folder in folders_list:
        folder_path = os.path.join(data_to_balance_path, folder)
        file_list = os.listdir(folder_path)
        samples_num = len(file_list)
        samples_num_to_add = max_samples_num - samples_num
        added_files = 0
        while samples_num_to_add > 0:
            file_to_copy = random.choice(file_list)
            shutil.copyfile(os.path.join(folder_path, file_to_copy),
                            os.path.join(folder_path, str(added_files).zfill(5) + file_to_copy))
            samples_num_to_add -= 1
            added_files += 1

balance_classes(TRAINING_DATA_PATH)

# DATA AUGMENTATION AND RESCALING
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    'C:/DS_Projects/Images_Classification/Cassava_Leaf_Desease_Classification/data/splitted_images/training/',
    target_size=(300, 300),  # All images will be resized to 300x300
    batch_size=training_batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    'C:/DS_Projects/Images_Classification/Cassava_Leaf_Desease_Classification/data/splitted_images/validation/',
    target_size=(300, 300),  # All images will be resized to 300x300
    batch_size=32,
    class_mode='categorical')

# CALLBACKS CREATIONS
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("cassava_classif_model.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# MODEL BUILDING
model: Sequential = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# MODEL TRAINING
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    epochs=500)