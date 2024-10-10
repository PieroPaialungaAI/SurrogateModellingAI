# Surrogate Modelling for Time Of Flight Diffraction of Rough Defects 

Hello and welcome to the code of the paper "Surrogate Modelling for Time Of Flight Diffraction of Rough Defects". 


This GitHub repository collects all the code that has been used in the study made by Piero Paialunga and Joseph Corcoran at the __University Of Cincinnati__.

The structure of the code is the following: 

1. [__Importing and Preprocessing__](#1.Importing-and-Preprocessing) the data 
2. [Implementation of the Machine Learning models for the __full A-Scan reconstruction__ (CNN+RNN) and the main peak (RNN)](#2. Machine Learning model)
3. Matlab code description for the __defect generation__
4. Training of the __full A scan model__
5. Training of the __main peak prediction model__
6. __Testing of the models__ on the new generated defects
7. __Plotting and reporting__ of the results. 

## 1.Importing and Preprocessing

### 1.1 How to use it 
In order to import and preprocess your data run the following script on your terminal:
```
python preprocessing.py 'your_defect_path' 'your_ascan_path' 'your_preprocessed_data_path"
```
In case you are downloading the forPogo and `AScanFile.mat` A scan file, you can simply run. Note that the destination path should be named __preprocessed_data__.:
```
python preprocessing.py None None None 
```
### 1.2 Description
The first step of our code is the import and preprocess of the training data. 

The training data are generated using a software named [Pogo](http://www.pogo.software/). This software allows us to do a 
simulation on ultrasonic experiments. In particular, given the setup discussed in the Paper, we are able to generate for a given defect, the response A scan. 
All the profiles of the defects are included into a forPogo folder (???) and the defects A scan are collected into a single
`
.mat
`
file.

As both the profiles and the A Scans will be treated using Machine Learning, we wanted to prevent any NaN or vanishing gradient issues.
For this reason, bothe the defect profiles and the A scans have been multiplied by two different multiplying factors. 
The scaled A scan is also very close to 0 except for a small area. 
For this reason, the signal has been also trimmed between a starting and ending point. The starting point, ending point, multiplying factors and all the other constants are included into the 
`
constants.py
`
file. 
The multiplying, trimming and reading operations are all included into the `
utils.py
` file.

Lastly, the `preprocessing.py` file does the operation of:

* Reading the defects and A scans from the defect and A Scan paths (or the default ones), 
* Applying the scale factors 
* Trimming the A Scans
* Exporting them into `.csv` files. 

In particular, you will get a:

* `right_profile.csv` that is the preprocessed right profile of the defects file 
* `left_profile.csv` that is the preprocessed left profile of the defects (used for plotting) file
* `target_scans.csv` that is the preprocessed A scans file.

## 2. Machine Learning models

### 2.1 How to use it 

Two pretrained versions of the full scan model and the main peak model are reported in the __models__ folder. 
After the import and preprocessing steps, you can run the training using the script:
```python training.py```


### 2.2 Description
Two different models are implemented: 

1. The `full_scan_model.h5` takes as an input the defect and it outputs is the full normalized A scan model. This model combines the computational power of a Convolutional Neural Network (CNN) and the one of the Recurrent Neural Network (RNN-GRU). The inputs of these two models are the raw 240 long defect and a reshaped version of the same defect. The reshape converts the 240 long defect into 10 bits of 24 values. The dataset for this model is prepared using the `data_loader.py` script in the `dataset` module. As we can see from the image below the output of the model is 1800 points long (that's the length of the target A scan)

2. The `main_peak_model.h5` takes as an input the defect and it outputs a single value, which is the main peak and the normalization factor of the A scan. This model is a Recurrent Neural Network LSTM one and the input is the raw defect. The dataset for this model is prepared using the `data_loader.py` script in the `dataset` module. As we can see from the image below the output of the model is a single value. 










