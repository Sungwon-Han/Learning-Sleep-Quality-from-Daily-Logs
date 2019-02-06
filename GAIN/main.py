import argparse

from preprocess import *

import pandas as pd
import numpy as np
import random
import pickle

from GAIN import GAIN

def Argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epouh',type=int,default=20000,help='setting epouh of GAIN')
    parser.add_argument('--file_path',type=str,default=file_path,help='setting file path')
 
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = Argument()

    ##Setting
    input_data_path= input(">>> Input File path : ")
    input_save_path = input(">>> Input Save path + File name.csv : ")
    epouh= input(">>> Input Epouh  : ")

    #epouh = 20000
    file_path = input_data_path

    ##loading data
    sleep_activity_data_csv = pd.read_csv(file_path)
    
    ##user_Id
    user_Id = list(set(sleep_activity_data_csv["userId"]))  

    ##feature name
    feature_name = list(sleep_activity_data_csv.columns)[1:] 

    ##Calculating Max and Min
    max_ = sleep_activity_data[sleep_activity_data_csv.columns[1:]].max()  
    min_ = sleep_activity_data[sleep_activity_data_csv.columns[1:]].min()  

    ##FillNa with 0.8
    sleep_activity_data = sleep_activity_data_csv.fillna(0.8)

    ##Standard
    sleep_activity_data_standard = (sleep_activity_data[sleep_activity_data.columns[1:]]-min_)/(max_-min_) 
    sleep_activity_data_standard =  pd.concat([user_Id_date,sleep_activity_data_standard],axis=1) 

    ##Storing user data to dictionary
    dict_user_data = Dict_user_data(sleep_activity_data_standard)
    
    ##Transforming user_data into window format and record the number of window in each user
    dict_window,window_number = Dict_window(dict_user_data,user_Id_date)  
    
    ##transfer dict_window into window 
    window = Window(dict_window,user_Id_date) 
    
    ##setting mask matrix and can set the number of missong data
    mask_dataframe = Mask_dataframe(sleep_activity_data_csv)
    mask_window = Mask_window(mask_dataframe)
 
    ##Building GAIN model
    generate_window = GAIN(window,mask_window,epouh)

    
    ##Transform generate window into impute data    
    impute_data_standard,impute_data = Delete_window(window_number,generate_window,user_Id,feature_name,max_,min_)
    
    ##Saving
    impute_data_standard.to_csv(input_save_path)
    
    '''
    impute_data is a dataframe withod min-max normalization
    impute_data_standard is a dataframe(min-max normalization)
    
    '''