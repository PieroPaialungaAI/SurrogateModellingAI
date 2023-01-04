import scipy.io
import numpy as np
import pandas as pd 
import glob 
from utils import *
from constants import *
import argparse
import os


""" This script reads the defects from a given folder and the resulting A scan from a given file. 
Once this is done, it prepares them to the Machine Learning step by doing the preprocessing techniques """ 


class DataPreProcessor:
    
    def __init__(self,a_scan_loc,defect_loc):
        self.a_scan_loc = a_scan_loc
        self.defect_loc = defect_loc

    
    def preprocess_ascans(self):
        Y = scipy.io.loadmat(self.a_scan_loc)[SCAN_STRING]
        Y = a_scans_prepro(Y)
        return Y
    
    def preprocess_defects(self):
        list_of_defects = ordering_defects(self.defect_loc)
        X_top = np.array([scipy.io.loadmat(p)[DEFECT_STRING_TOP][0] for p in list_of_defects])
        X_top = defects_prepro(X_top)
        X_bottom = np.array([scipy.io.loadmat(p)[DEFECT_STRING_BOTTOM][0] for p in list_of_defects])
        X_bottom = defects_prepro(X_bottom)
        return {'X_top':X_top,'X_bottom':X_bottom}

def full_preprocess(f,dataloc):
    print('Using default A scan and defect path \n')
    a_scan_loc, defect_loc, angle = f['a_scan_loc'],f['defect_loc'],f['angle']
    data_preprocessor = DataPreProcessor(a_scan_loc,defect_loc)
    print('Preprocessing Defects...\n')
    X =  data_preprocessor.preprocess_defects()
    X_top, X_bottom = X['X_top'],X['X_bottom']
    print('Preprocessing A Scans...\n')
    Y = data_preprocessor.preprocess_ascans()
    print('Preprocessing is successfull, exporting the data...\n')
    X_top, X_bottom = pd.DataFrame(X_top).loc[0:len(Y)], pd.DataFrame(X_bottom).loc[0:len(Y)]
    Y = pd.DataFrame(Y)
    print('Exporting on default folder')
    dataloc = dataloc+'/'+str(int(angle))+'_data'
    if not os.path.exists(dataloc):
        os.mkdir(dataloc)
    Y.to_csv(dataloc+'/'+'target_scans.csv')
    X_top.to_csv(dataloc+'/'+'right_profile_defects.csv')
    X_bottom.to_csv(dataloc+'/'+'left_profile_defects.csv')
    
if __name__ == "__main__":
    e_fold = extract_folders(FILE_PATH)
    for fold in e_fold:
        full_preprocess(fold,DATA_LOC)

    
    
    
    
    
    
    
