# IMPORT LIBRARIES
import os
import errno
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# SET PARAMETERS
CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH, 'data')
ALL_IMAGES_PATH = os.path.join(DATA_PATH, 'train_images')
LABEL_CSV_FILE_LOCATION = os.path.join(DATA_PATH, 'train.csv')
SPLITTED_IMAGES_PATH = os.path.join(DATA_PATH, 'splitted_images')  # folder to create
TRAINING_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'training\\')  # folder to create
VALIDATION_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'validation\\')  # folder to create
validation_rate = 0.2
training_batch_size = 512
image_dimension = 150
random.seed(33)


# FUNCTION TO CREATE FOLDERS
# If a folder already exists, delete it and create it again
def delete_create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        shutil.rmtree(path)
        os.mkdir(path)


# FUNCTION TO DIVIDE DATA IN FOLDERS ACCORDING TO LABELS
def label_data(folder_path, label_csv_file_location):
    label_table = pd.read_csv(label_csv_file_location, sep=',')
    labels = np.unique(label_table['label'])
    for label in labels:
        delete_create_folder(os.path.join(folder_path, label))
    for row in label_table.itertuples(index=False):
        try:
            shutil.move(os.path.join(folder_path, row[0]),
                        os.path.join(folder_path, str(row[1]) + '\\' + row[0]))
        finally:
            pass


label_data(ALL_IMAGES_PATH, os.path.join(DATA_PATH, 'train.csv'))


# FUNCTION TO SPLIT DATA IN TRAINING AND VALIDATION SET
def split_data(split_data_folder_path, source_data_path, val_rate=0.2):
    delete_create_folder(split_data_folder_path)
    delete_create_folder(os.path.join(split_data_folder_path, 'training'))
    delete_create_folder(os.path.join(split_data_folder_path, 'validation'))
    folders_list = os.listdir(source_data_path)
    for folder in folders_list:
        file_list = os.listdir(os.path.join(source_data_path, folder))
        validation_file_list = random.sample(file_list,
                                             round(val_rate * len(file_list)))  # randomly select the validation set
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


# FUNCTION TO BALANCE UNBALANCED CLASSES BY OVERSAMPLING
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
        while samples_num_to_add > 0:  # add samples_num_to_add images in underrepresented classes
            file_to_copy = random.choice(file_list)
            shutil.copyfile(os.path.join(folder_path, file_to_copy),
                            os.path.join(folder_path,
                                         str(added_files).zfill(5) + file_to_copy))  # add prefix to file name
            samples_num_to_add -= 1
            added_files += 1


balance_classes(TRAINING_DATA_PATH)

# DATA AUGMENTATION AND RESCALING
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    'C:/DS_Projects/Images_Classification/Cassava_Leaf_Desease_Classification/data/splitted_images/training/',
    target_size=(image_dimension, image_dimension),
    batch_size=training_batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    'C:/DS_Projects/Images_Classification/Cassava_Leaf_Desease_Classification/data/splitted_images/validation/',
    target_size=(image_dimension, image_dimension),
    batch_size=32,
    class_mode='categorical')

# CALLBACKS CREATIONS
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'cassava_classif_model.h5'),
                                                   save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# OPTION 1: BUILD CONVOLUTIONAL MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_dimension, image_dimension, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# OPTION 2: BUILD TRANSFER LEARNING MODEL WITH RESNET
img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                                          input_shape=[image_dimension, image_dimension, 3])
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # do not change parameters of ResNer, at least in the first epochs

model = tf.keras.models.Sequential([
    img_adjust_layer,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# COMPILE MODEL
# Set dynamic learning rate
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# TRAIN MODEL
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    callbacks=[checkpoint_cb, early_stopping_cb])

# PRINT TRAINING-VALIDATION LOSS AND ACCURACY
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
