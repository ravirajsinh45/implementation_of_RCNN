# implementation_of_RCNN
We implement RCNN algorithm for object detection from an Images.

# Inroduction
* R-CNN was proposed by Ross Girshick et al. in 2014 to deal with the problem of efficient object localization in object detection. 
* It changed the object detection field fundamentally. By leveraging selective search, CNN and SVM. 


# Summary of algorithm
![](https://media.geeksforgeeks.org/wp-content/uploads/20200219161502/RCNN1.png)

1. First of all selective search algorithm is applied to images and give us around 2000 region which might containing an object.
2. Wraped region is feed into CNN model(in paper, Alexnet is used) which return (1,4096) size feature vector.
3. This region feed into SVM classifier which give object class and confidence score.
4. Than after we have to train one bounding box regression model for generating tight rectangle boxes for object in images.


It Seems easy Right? Now i will explain what things i did for train my own custom RCNN model.


# RCNN for my custom dataset

I used my [Crop and Weed detection](https://www.kaggle.com/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) dataset which i collected and labeled it myself. I also uploaded it in [kaggle](https://www.kaggle.com/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes). This datset contains total 1300 images. I used 1000 images for training and 300 images for testing

## Data Preprocessing steps

* First I apply selective search algorithm and generated ~2000 region per images.
* Than I compare generated region with ground truth labels by mean of Intersection over union(iou).

  <img src="https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png" width="200" height="200">

* Which ever region has iou>0.5 is saved as positive example(it might object) and region which has iou<0.2 is saved as negative example.
* In [RCNN paper](https://arxiv.org/pdf/1311.2524.pdf) it had take iou<0.3 for negative example but by this i get worst result for my dataset so i changed it to 0.2.
* NOTE: I didn't included all negative examples which has iou<0.2, i only selected random images and double of number of postive examples.check this [code](https://github.com/ravirajsinh45/implementation_of_RCNN/blob/master/region_proposals.py) for better understanding. 
*  I saved just cordinnates of bounding boxes by using [data_processing_part_1.ipynb](https://github.com/ravirajsinh45/implementation_of_RCNN/blob/master/data_processing_part_1.ipynb).
* Saved all images using [rcnn-data-preprocessing-part-2](https://www.kaggle.com/ravirajsinh45/rcnn-data-preprocessing-part-2) notebook on kaggle.


## Training of model

* RCNN Model training is divided in three parts 
  1. CNN finetuning
  2. CNN + SVM training
  3. Bounding Box regression

  ### 1. CNN finetuning

  * I finetuned VGG16 model with my generated region proposals. But in paper they used Alexnet.
  * Input size of model is 224x224x3 and Output 3 classes (Crop, Weed and Background).
  * Model perform very well and give 95.88% accuracy on test images. 
  * I used kaggle platform to train my models for take advantge of free GPU.
  * Finetuning training notebook is [here](https://www.kaggle.com/ravirajsinh45/rcnn-training-part-1-finetuning).

  ### 2. CNN + SVM
  
  * I removed last two fully connected layers from finetuned model and used CNN model as feature extractor.
  * CNN model will returns (1,4096) size feature vector.
  * Than I trained SVM model using feature vectors.
  * SVM improves overall prediction of model.
  * SVM training is [here](https://www.kaggle.com/ravirajsinh45/rcnn-training-part-2-cnn-svm) on kaggle.

  ### 3. Bounding Box regresion
  
  * I didn't train bounding box regression model yet. but i will upload it whenever i train it.
  


## Performing detection

* I did't train BB regressor but i still perform detection and result are look nicer. take look in [notebook](https://www.kaggle.com/ravirajsinh45/rcnn-performing-detection).

  
  ### Good Predictions

  ### Bad predictions




