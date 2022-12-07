# AdvancedProject
Repository for final project in  Advanced Project 2022

All code that has to do with training and testing of the implemented neural network and baseline is in the NeuralNetweok folder. Frontend and API are built for the frontend webpage which can receive new picture and output beauty index and tell if an image is morphed or not.

### Data augmentations
Data augmentations is in a ipynb file called data_augmentation. The notebook is only meant to be run when data augmentation is needed and only needs to run once for each original image. It contains 27 augmentations done in three iterations for each block of original data.

### Baseline model
PretrainedAlexNet.ipynb contains converting the pretrained AlexNet model from SCUT-FBP5500-Database-Release to python 3 to be able to extract the predicted labels to use as a baseline. 

### Training and validation
Forward.ipynb contains pre-processing of labels, training and validation for discrete outputs of 5 classes. 
Forward_Continuous.ipynb contains the same but for extended output of 400 classes.

The basis of the code used for accuracy measures and training loop is built from the course material from week 4 in Deep Learning. 

### Python files
Python codes are in three seperate .py files. They work together with the frontend and return prediced labels from pretrained models, pretrained AlexNet from SCUT-FBP5500-Database-Release, our trained AlexNet predicted labels with discrete classes from 1-5 and our trained AlexNet predicted labels for semi continuous output. 

### Plots Folder
Holds all plots generated during the training process and hyperparameter tuning, this was more for documentation purposes.

### Data Folder
If you need access to the image data used in this project you need to contact the authors. 