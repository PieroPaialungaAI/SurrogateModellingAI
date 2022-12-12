import scipy.io
import numpy as np
import pandas as pd 
import glob 
from constants import *
from os import listdir
from os.path import isfile, join

def a_scans_prepro(a_scans):
    return SCALE_FACTOR_SIGNAL*a_scans[:,int(TRIMMING_POINT):int(TRIMMING_POINT+LENGTH_SIGNAL)]

def defects_prepro(defects):
    return SCALE_FACTOR_DEFECT*np.array([x-x[-1] for x in defects])

def ordering_defects(defects_path):
    onlyfiles = [f for f in listdir(defects_path) if isfile(join(defects_path, f))]
    new_ind = []
    for file in onlyfiles:
        new_ind.append(int(file.split('_')[-1].split('.')[0]))
    onlyfiles = np.array(onlyfiles)[np.argsort(new_ind)]
    path_tot = []
    for f in onlyfiles:
        path_tot.append(defects_path+'/'+f)
    return path_tot
