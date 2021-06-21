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
I used data augmentation on data of Kaggle Project [*Cassava Leaf Deasease Classification*](https://www.kaggle.com/c/cassava-leaf-disease-classification) to train
* a classical convolutional model with 2 convolutional layers
* a model build from Resnet model with transfer learning

and I analyzed performances in terms of accuracy and training and validation loss functions.

## Motivation <a name="motivation" />
I was curious to understand the potential of transfer learning in training models on real data.

## Technical Aspect <a name="technical-aspects" />
The main issues in this projects were the quantity and the extreme heterogeneity of Cassava images. So it was needed very high computation power to:
* process a huge amount of data
* perform data augmentation to avoid overfitting
* build and train complex and deep models

I tackled this probelm by using Google Colab and by configuring Tensorflow to work with the GPU of my laptop. The best solution would have been to use a virtual machine in AWS Sagemaker or similar but it was too expensive.

## Result <a name="result" />
The confusion matrixes obtained from the predictions of the 4 analyzed models are the following:

<img src="https://user-images.githubusercontent.com/29163695/122113334-3fd3ff00-ce22-11eb-80e2-741cc13019e5.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112158-d0114480-ce20-11eb-85b8-47b4912d23ca.png" height="400">

<img src="https://user-images.githubusercontent.com/29163695/122112221-e0292400-ce20-11eb-8703-a550ec62404e.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112292-f0d99a00-ce20-11eb-9e06-88a16a05e469.png" height="400">

I noticed that:
1. The classification model provides a higher accuracy even if some predictions are heavily wrong (i.e. a lot of "5" in place of "1" or vice versa). The MAE is the highest because classes are independent and sorting information is not used.
2. Even if the regression models have the same complexity of the classification model, they do not provide so high accuracy. However, thanks to the optimization function used in the training phase, prediction errors are often lower.
3. Both classification and regression models, even if they are very simple, allow reaching the same precision of a human being. It is a proof of the potential of LSTM layers and convolutional layers in neural networks.

## Technologies <a name="technologies" />
I used *Tensorflow/Keras* for image preprocessing, data aumentation, ResnNet model implementation and model training. In order to increase computation power I also leverage *Google Colab*.

<img src="https://user-images.githubusercontent.com/29163695/122249778-6ac55e00-cec9-11eb-8e09-55fee48bc88f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">

## To Do <a name="to-do" />
* To optimize performances it would be useful to combine more models (ensemble learning). In this project we have an underfitting issue so we need a more complex model and a more effective data preprocessing.
* Train deeper model by using cloud services (GCP, Azure, AWS Sagemaker)

## File List <a name="file-list" />
* **main.py** Data loading, data preprocessing, model training and model evaluation
* **my_utils.py** Useful functions to prepare data, balance classes and plot loss trend
* **data_preparation.py** Data preparation and balancing of unbalanced classes through oversampling