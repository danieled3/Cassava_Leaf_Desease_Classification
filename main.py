# IMPORT LIBRARIES
import os
import csv
import errno
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf


# CREATE FOLDERS ACCORDING TO LABELS
CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH, 'data')
ALL_IMAGES_PATH = os.path.join(DATA_PATH, 'train_images')
LABEL_CSV_FILE_LOCATION = os.path.join(DATA_PATH, 'train.csv')

def label_data(folder, label_csv_file_location):
    label_table = pd.read_csv(label_csv_file_location, sep=',')
    labels = np.unique(label_table['label'])
    for label in labels:
        try:
            os.mkdir(os.path.join(folder, str(label)))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            shutil.rmtree(os.path.join(folder, str(label)))
            os.mkdir(os.path.join(folder, str(label)))

    for row in label_table.itertuples(index=False):
        try:
            shutil.move(os.path.join(folder, row[0]),os.path.join(folder, str(row[1])+'\\'+row[0]))
        finally:
            pass

label_data(ALL_IMAGES_PATH, os.path.join(DATA_PATH, 'train.csv') )

# SPLIT DATA IN TRAINING AND VALIDATION
validation_rate = 0.2
last_char_val = np.array(['9','0'])
SPLITTED_IMAGES_PATH = os.path.join(DATA_PATH, 'splitted_images')

try:
    os.mkdir(SPLITTED_IMAGES_PATH)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    shutil.rmtree(SPLITTED_IMAGES_PATH)
    os.mkdir(SPLITTED_IMAGES_PATH)

folders_list = os.listdir(ALL_IMAGES_PATH)
for folder in folder_list:
    file_list = os.listdir(os.path.join(ALL_IMAGES_PATH,folder))
    validation_file_list =
file_list = os.listdir(SOURCE)
file_list_ne = [f for f in file_list if os.path.getsize(SOURCE + f) > 0]
testing_set = random.sample(file_list_ne, round((1 - SPLIT_SIZE) * len(file_list_ne)))
training_set = [f for f in file_list_ne if f not in testing_set]


CURRENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(PATH, 'data')
TRAIN_IMAGES_PATH = os.path.join(data_dir, 'train_images')





BATCH_SIZE = 32
IMG_SIZE = (160, 160)


import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

validation_horse_hames = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)

# IMAGES FROM CSV
def get_data(filename):
    with open(filename) as training_file:
        # Your code starts here
        file = csv.reader(training_file, delimiter=',', quotechar='|')
        labels = []
        images = []
        rows = 0
        for row in file:
            labels = labels + [row[0]]
            images = images + [row[1:len(row) + 1]]
            rows = rows + 1

        labels = np.array(labels)[1:len(labels) + 1]
        images = np.array(images)[1:len(images) + 1].reshape(rows - 1, 28, 28)
    # Your code ends here
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


# SPLIT DATA
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    file_list = os.listdir(SOURCE)
    file_list_ne = [f for f in file_list if os.path.getsize(SOURCE + f) > 0]
    testing_set = random.sample(file_list_ne, round((1 - SPLIT_SIZE) * len(file_list_ne)))
    training_set = [f for f in file_list_ne if f not in testing_set]

    if not os.path.exists(os.path.dirname(os.path.dirname(TRAINING))):
        os.mkdir(os.path.dirname(os.path.dirname(TRAINING)))
    if not os.path.exists(TRAINING):
        os.mkdir(TRAINING)
    if not os.path.exists(os.path.dirname(os.path.dirname(TESTING))):
        os.mkdir(os.path.dirname(os.path.dirname(TESTING)))
    if not os.path.exists(TESTING):
        os.mkdir(TESTING)

    for f in testing_set:
        copyfile(SOURCE + f, TESTING + f)
    for f in training_set:
        copyfile(SOURCE + f, TRAINING + f)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# PREPROCESS IMAGES

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation and rescaling

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')  #'categorical' for multiple classes

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary') #'categorical' for multiple classes

# CREATE CALLBACK
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# LOAD AND  TRANSFORM DATASET
mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# TRANSFER LEARNING
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
!wget - -no - check - certificate \
    https: // storage.googleapis.com / mledu - datasets / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
              - O / tmp / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)

model = Model( pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])



# CREATE CLASSIFICATION MODEL - MULTIPLE CLASSES
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# CREATE CLASSIFICATION MODEL - 2 CLASSES
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Add Dropout
    tf.keras.layers.Dropout(0.5),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid') # tf.keras.layers.Dense(5, activation='softmax') for multiple classes

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', #'categorical_crossentropy' for multiple classes or sparse categoricl crossentropy without to_categorical
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# TRAIN MODEL - TRAINING - NORMAL
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# TRAIN MODEL - VALIDATION - PREPROCESSING
history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

# TRAIN MODEL - VALIDATION - PREPROCESSING - from file

# Train the Model
history = model.fit_generator(
    train_datagen.flow(training_images, tf.keras.utils.to_categorical(training_labels), batch_size=64),
    validation_data=validation_datagen.flow(testing_images, tf.keras.utils.to_categorical(testing_labels)),
    steps_per_epoch=len(training_labels) // 64,
    epochs=20)

model.evaluate(testing_images, testing_labels, verbose=0)
# EVALUATE MODEL - NORMAL
test_loss = model.evaluate(test_images, test_labels)

# PRINT TRAINING-VALIDATION LOSS AND ACCURACY
import matplotlib.pyplot as plt
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

plt.show()

# SAVE MODEL
model.save("rps.h5")

# TEST IMAGE FROM WEB IN GOOGLE COLAB
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
# predicting images
    path = fn
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(fn)
print(classes)

