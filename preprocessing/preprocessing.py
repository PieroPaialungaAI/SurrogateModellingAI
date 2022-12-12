import scipy.io
import numpy as np
import pandas as pd 
import glob 
from utils import *
from constants import *
import argparse



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

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reading input folders')
    parser.add_argument("a_scan_loc", type=str)
    parser.add_argument("defect_loc", type=str)
    parser.add_argument("preprocessed_file_path", type=str)

    args = parser.parse_args()
    data_preprocessor = DataPreProcessor(args.a_scan_loc,args.defect_loc)
    try:
        X = data_preprocessor.preprocess_defects()
    except: 
        print('Using default A scan and defect path \n')
        data_preprocessor = DataPreProcessor(A_SCAN_PATH,DEFECT_PATH)
        print('Preprocessing Defects...\n')
        X =  data_preprocessor.preprocess_defects()
    X_top, X_bottom = X['X_top'],X['X_bottom']
    print('Preprocessing A Scans...\n')
    Y = data_preprocessor.preprocess_ascans()
    print('Preprocessing is successfull, exporting the data...\n')
    X_top, X_bottom = pd.DataFrame(X_top).loc[0:len(Y)], pd.DataFrame(X_bottom).loc[0:len(Y)]
    Y = pd.DataFrame(Y)
    #Originally "various_r_scans.csv"
    try:
        Y.to_csv(args.preprocessed_file_path+'/'+'target_scans.csv')
        #Originally "varios_r_defects_right.csv"
        X_top.to_csv(args.preprocessed_file_path+'/'+'right_profile_defects.csv')
        #Originally "varios_r_defects_left.csv"
        X_bottom.to_csv(args.preprocessed_file_path+'/'+'left_profile_defects.csv')
    except:
        print('Exporting on default folder')
        Y.to_csv(DATA_LOC+'/'+'target_scans.csv')
        #Originally "varios_r_defects_right.csv"
        X_top.to_csv(DATA_LOC+'/'+'right_profile_defects.csv')
        #Originally "varios_r_defects_left.csv"
        X_bottom.to_csv(DATA_LOC+'/'+'left_profile_defects.csv')
    
    
    
    
    
    
    
