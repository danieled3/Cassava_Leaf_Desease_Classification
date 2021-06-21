# Cassava Leaf Desease: Image Classification

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspects](#technical-aspects)
  * [Results](#result)
  * [Technologies](#technologies)
  * [To Do](#to-do)
  * [File List](#file-list)

  
## Overview <a name="overview" />
The aim of this project is to classify real images of leafs of Cassava into four disease categories or a fifth category indicating a healthy leaf.
I used data augmentation on a part of data of Kaggle Project [*Cassava Leaf Deasease Classification*](https://www.kaggle.com/c/cassava-leaf-disease-classification) to train
* a classical convolutional model with 2 convolutional layers
* a model build from Resnet model with transfer learning

and I analyzed performances in terms of accuracy and training/validation loss functions.

## Motivation <a name="motivation" />
I was curious to understand the potential of transfer learning in training models on real data.

## Technical Aspects <a name="technical-aspects" />
The main issues in this projects were the quantity and the extreme heterogeneity of Cassava images. So it was needed very high computation power to:
* process a huge amount of data
* perform data augmentation to avoid overfitting
* build and train complex and deep models

I tackled this problem by using Google Colab and by configuring Tensorflow to work with the GPU of my laptop. The best solution would have been to use a virtual machine in AWS Sagemaker or similar but it was too expensive.

## Results <a name="result" />
The trends of accuracy functions and loss functions for the convolutional model are the following:

<img src="https://user-images.githubusercontent.com/29163695/122765872-3bd03300-d2a1-11eb-89e0-cab12f31947c.png" height="350">
<img src="https://user-images.githubusercontent.com/29163695/122765901-42f74100-d2a1-11eb-9639-aa87a1f2b939.png" height="350">

Since the classes are 5, the images are very heterogeneous and the tested model has only 2 convolutional layers, an accuracy of 0.6 is a rather good result. After 14 epochs model overfits training data even if used data augmentation to generalize them. Performance may be improved by adding training data or by generating new images from training set in a different way. 

The trends of accuracy functions and loss functions for the model built from Resnet are the following:

<img src="https://user-images.githubusercontent.com/29163695/122765997-560a1100-d2a1-11eb-84d4-06ae093b42a6.png" height="350">
<img src="https://user-images.githubusercontent.com/29163695/122765948-4b4f7c00-d2a1-11eb-8474-4cb21fab8dfd.png" height="350">

The model built from Resnet assures almost the same accuracy of the convolutional model. Training loss is always higher than validation loss because training data, thanks to data augmentation, are more complex than validation data and our model has few trainable parameters. Adding some more layers and more nodes after Resnet layers may improve training and validation accuracy.

## Technologies <a name="technologies" />
I used *Tensorflow/Keras* for image preprocessing, data aumentation, ResnNet model implementation and model training. In order to increase computation power I also leverage *Google Colab*.

<img src="https://user-images.githubusercontent.com/29163695/122249778-6ac55e00-cec9-11eb-8e09-55fee48bc88f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">

## To Do <a name="to-do" />
* Train the parameters of Resnet after some epochs to further decrease loss function
* To optimize performances it would be useful to combine more models (ensemble learning). In this project we have an underfitting issue so we need a more complex model and a more effective data preprocessing
* Train deeper model by using cloud services (GCP, Azure, AWS Sagemaker)

## File List <a name="file-list" />
* **main.py** Data loading, data preprocessing, model training and model evaluation
* **my_utils.py** Useful functions to prepare data, balance classes and plot loss trend
* **data_preparation.py** Data preparation and balancing of unbalanced classes through oversampling
