# AdvancedProject
Repository for final project in  Advanced Project 2022

### Data augmentations
Data augmentations is in a ipynb file called data_augmentation. The notebook is only meant to be run when data augmentation is needed and only needs to run once for each original image. It contains 27 augmentations done in three iterations for each block of original data.

### Baseline model
PretrainedAlexNet.ipynb contains converting the pretrained AlexNet model from SCUT-FBP5500-Database-Release to python 3 to be able to extract the predicted labels to use as a baseline. 

### Training and validation
Forward.ipynb contains pre-processing of labels, training and validation for discrete outputs from 1-5. 
Forward_??.ipynb contains the same but for semi continuous outputs from 1-5. 

### Python files
Python codes are in three seperate .py files. They work together with the frontend and return prediced labels from pretrained models, pretrained AlexNet from SCUT-FBP5500-Database-Release, our trained AlexNet predicted labels with discrete classes from 1-5 and our trained AlexNet predicted labels for semi continuous output. 