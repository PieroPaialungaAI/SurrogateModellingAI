# Surrogate Modelling for Time Of Flight Diffraction of Rough Defects 

Hello and welcome to the code of the paper "Surrogate Modelling for Time Of Flight Diffraction of Rough Defects". 


This GitHub repository collects all the code that has been used in the study made by Piero Paialunga and Joseph Corcoran at the __University Of Cincinnati__.

The structure of the code is the following: 

1. [__Importing and Preprocessing__](##1.Importing and Preprocessing) the data 
2. Implementation of the Machine Learning model for the __full A-Scan reconstruction__ (CNN+RNN)
3. Implementation of the Machine Learning model for the __main peak prediction__ (RNN)
4. Matlab code description for the __defect generation__
5. Training of the __full A scan model__
6. Training of the __main peak prediction model__
7. __Testing of the models__ on the new generated defects
8. __Plotting and reporting__ of the results. 

##1.Importing and Preprocessing
In order to import and preprocess your data run the following script on your terminal:
```
python preprocessing.py 'your_defect_path' 'your_ascan_path'
```
In case you are downloading the forPogo and `AScanFile.mat` A scan file, you can simply run:
```
python preprocessing.py None None
```
###1.2 Description
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

* `right_profile_defects.csv` that is the preprocessed right profile of the defects file 
* `left_profile_defects.csv` that is the preprocessed left profile of the defects (used for plotting) file
* `target_scans.csv` that is the preprocessed A scans file.




