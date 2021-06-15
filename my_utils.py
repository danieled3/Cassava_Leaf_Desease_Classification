# IMPORT LIBRARIES
import os
import errno
import shutil
import random
import numpy as np
import pandas as pd

"""Create a folder. If the folder already exists, delete it and create it again

Arguments:
    path: string with the path of the new folder"""

def delete_create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        shutil.rmtree(path)
        os.mkdir(path)


"""Create a folder for each label and move in each of them the corresponding images

Arguments:
    folder_path(string): path of the folder where images are and where function will create a folder for each label
    label_csv_file_location(string): path of the csv file where every image name is associated to the 
                             corresponding label"""

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


"""Randomly divide images in training set and test set and move them in the folders "training" and "test"

Arguments:
    split_data_folder_path(string): path of the folder where the folders "training" and "test" will be created and where images will be moved to
    source_data_path(string): path where images are, divided in folders according to their labels
    val_rate(float): portion of images to use for validation set
    seed(integer): number to use as seed for random data splitting"""

def split_data(split_data_folder_path, source_data_path, val_rate=0.2, seed=123):
    delete_create_folder(split_data_folder_path)
    delete_create_folder(os.path.join(split_data_folder_path, 'training'))
    delete_create_folder(os.path.join(split_data_folder_path, 'validation'))
    folders_list = os.listdir(source_data_path)
    random.seed(seed)
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



"""Balance classes by oversampling 

Arguments:
    data_to_balance_path(string): path of the folder where data to be balanced are
    seed(integer): number to use as seed for random choice of images to copy"""

def balance_classes(data_to_balance_path, seed=123):
    folders_list = os.listdir(data_to_balance_path)
    samples_num_list = [len(os.listdir(os.path.join(data_to_balance_path, folder))) for folder in folders_list]
    max_samples_num = max(samples_num_list)
    random.seed(seed)
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
