import scipy.io
import numpy as np
import pandas as pd 
import glob 
from constants import *
from os import listdir
from os.path import isfile, join
import os 


def trimming_points_value(angle):
    return ANGLE_TRIMMING[angle]
 
def a_scans_prepro(a_scans,trimming_point):
    return SCALE_FACTOR_SIGNAL*a_scans[:,int(trimming_point):int(trimming_point+LENGTH_SIGNAL)]

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

def extract_folders(file_loc):
    folder = file_loc
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    data_folders = []
    for sub_folder in sub_folders:
        if len(sub_folder.split('.'))==1:
            data_folders.append(sub_folder)
    final_data_folders = []
    for d in data_folders:
        angle = float(d.split('_')[0])
        a_scan_loc= folder+'/'+d+'/'+TIME_SERIES_STRING
        defects_loc = folder+'/'+d+'/'+DEFECT_STRING
        dict_data = {'a_scan_loc':a_scan_loc,'defect_loc':defects_loc,'angle':angle}
        final_data_folders.append(dict_data)
    return final_data_folders