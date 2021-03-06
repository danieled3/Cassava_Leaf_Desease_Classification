"""Load data, perform data augmentation, train 2 models (classic convolutional model and ResNet) and evaluate
performances"""

# IMPORT LIBRARIES
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import my_utils

# SET PARAMETERS
CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH, 'data')
ALL_IMAGES_PATH = os.path.join(DATA_PATH, 'train_images')
LABEL_CSV_FILE_LOCATION = os.path.join(DATA_PATH, 'labels.csv')
SPLITTED_IMAGES_PATH = os.path.join(DATA_PATH, 'splitted_images')  # folder to create
TRAINING_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'training\\')  # folder to create
VALIDATION_DATA_PATH = os.path.join(SPLITTED_IMAGES_PATH, 'validation\\')  # folder to create

training_batch_size = 128  # >=512 exceed free system memory
validation_batch_size = 32
image_dimension = 150
patience = 15  # patience for early stopping

# DATA AUGMENTATION AND RESCALING
train_datagen = ImageDataGenerator(  # data augmentation
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # shear_range=0.2,      # I would obtain data too different from test set
    # zoom_range=0.2,
    # horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)  # rescaling

train_generator = train_datagen.flow_from_directory(
    'data/splitted_images/training/',
    target_size=(image_dimension, image_dimension),
    batch_size=training_batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    'data/splitted_images/validation/',
    target_size=(image_dimension, image_dimension),
    batch_size=validation_batch_size,
    class_mode='categorical')

# EARLY STOPPING CALLBACKS
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

# CHECKPOINT CALLBACK
checkpoint_conv_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'conv.h5'),
                                                        save_best_only=True)

# OPTION 1: BUILD CONVOLUTIONAL MODEL
model_conv = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_dimension, image_dimension, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# COMPILE MODEL
# Set dynamic learning rate
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)

model_conv.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
model_conv.summary()

# TRAIN MODEL
history_conv = model_conv.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    callbacks=[checkpoint_conv_cb, early_stopping_cb])

# CHECKPOINT CALLBACK
checkpoint_resnet_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'resnet.h5'),
                                                          save_best_only=True)

# OPTION 2: BUILD TRANSFER LEARNING MODEL WITH RESNET
img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                                          input_shape=[image_dimension, image_dimension, 3])
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # do not change parameters of ResNet, at least in the first epochs

model_resnet = tf.keras.models.Sequential([
    img_adjust_layer,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# COMPILE MODEL

model_resnet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
model_resnet.summary()

# TRAIN MODEL
history_resnet = model_resnet.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    callbacks=[checkpoint_resnet_cb, early_stopping_cb])

# PRINT TRAINING-VALIDATION LOSS AND ACCURACY
my_utils.print_history(history_conv,
                       title='Loss Trend for Convolutional model',
                       early_stopping_line=True, patience=patience)
my_utils.print_history(history_resnet,
                       title='Loss Trend for Transfer Learning model',
                       early_stopping_line=True, patience=patience)
