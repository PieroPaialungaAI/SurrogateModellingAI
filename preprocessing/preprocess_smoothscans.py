import scipy.io
import numpy as np
import pandas as pd 
import glob 
from utils import *
from constants import *
import argparse
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

if __name__=='__main__':
    folders = FILE_PATH+'/SmoothDefects'
    onlyfiles = [f for f in listdir(folders) if isfile(join(folders, f))]
    print('Saving preprocessed smooth scan')
    smooth_files = []
    angles = []
    for f in onlyfiles:
        try:
            complete_file_path = folders+'/'+f
            angle = int(f.split('_')[0])
            angles.append(angle)
            print('Saving angles ='+str(angle))
            smooth_ascan = preprocess_smooth_scan(complete_file_path)
            smooth_files.append(smooth_ascan)
            np.save(folders+'/'+str(angle)+'SmoothDefectPreprocessed/'+'_smooth_preprocessed.npy',smooth_ascan)
        except:
            continue
    plt.figure(figsize=(10,10))  
    smooth_files = np.array(smooth_files)[np.argsort(angles)]
    angles = np.array(angles)[np.argsort(angles)]
    k=1
    for i in range(len(smooth_files)):
        plt.subplot(len(smooth_files),1,k)
        plt.plot(smooth_files[i])
        plt.title(angles[i])
        k=k+1
    plt.tight_layout()
    plt.show()
    

    