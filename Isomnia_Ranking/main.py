import argparse
import json

import pandas as pd
from preprocess import *
from LSTM_DH2 import LSTM_DH2
from Cascade_Forest import Cascade_Forest
from rank_measure import *

def Argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window＿size',type=int,default=8,help='setting window size')
    parser.add_argument('--sleep_efficiency_location',type=int,default=4,help='setting the location of main factor')
    parser.add_argument('--thr',type=float,default=0,help='setting threshold')
    parser.add_argument('--train_No',type=int,default=40,help='setting the number of training data')
    parser.add_argument('--test_No',type=int,default=6,help='setting the number of testing data')
 
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    
    window＿size = 8
    test_No = 1

    json_path = '....../parameter.json'

    with open(json_path,'r') as file_object:
        parameter = json.load(file_object)

    file_path = parameter['Isomnia_Ranking']['file_path']
    thr = parameter['Isomnia_Ranking']['threshold']
    sleep_efficiency_location = parameter['Isomnia_Ranking']['main_effect_location']
    
    input_data_csv = pd.read_csv(file_path)
    user_Id = list(set(input_data_csv["userId"]))
    
     
    ##Data Pre-Processing

    sleep_activity_nap = max_min_normalization(input_data_csv)
    
    dict_user_data = Dict_user_data(sleep_activity_nap)
    
    dict_user_window,dict_user_sleep_efficency = Dict_user_window_sf(user_Id,dict_user_data,window＿size,sleep_efficiency_location)

    dict_user_window_diff,dict_user_sf_diff = Dict_user_window_sf_diff(user_Id,dict_user_window,dict_user_sleep_efficency,thr)

    train_No = dict_user_window_diff[1].shape[0]-test_No
    
    dict_user_X_train,dict_user_Y_train,dict_user_X_test,dict_user_Y_test = Dict_X_Y_seperate(user_Id,dict_user_window_diff,dict_user_sf_diff,train_No)

    X_train,Y_train = X_Y_train(user_Id,dict_user_X_train,dict_user_Y_train)

    ##Building LSTM_DH_G model
    latent_train_vector ,dict_user_X_latent_vector= LSTM_DH2(X_train,Y_train,user_Id,dict_user_X_test)

    dict_user_X_predict = Cascade_Forest(test_No,user_Id,latent_train_vector,Y_train,dict_user_X_latent_vector)

    ##Transforming fitting format
    list_day_dict_predict = Standard(test_No,user_Id,dict_user_X_predict)

    ##Ranking predicting result in test data
    for day in range(test_No):
        predict_rank = Greedy_TSP(user_Id,list_day_dict_predict[day]) 
        print(" Rank : ",predict_rank)
