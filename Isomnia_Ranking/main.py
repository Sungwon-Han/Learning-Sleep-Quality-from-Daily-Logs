import argparse

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

    args = Argument()

    input_data_path= input(">>> Input File path : ")
    sleep_efficiency_location = input(">>>Main Effect Feature : ")-1
    experiment_time = input(">>>Experiment time : ")
    input_data_csv = pd.read_csv(input_data_path)
    user_Id = list(set(input_data_csv["userId"]))
    
    window＿size = 8
    #sleep_efficiency_location = 4
    thr = 0.006
    #train_No = 40
    test_No = 6
    train_No = experiment_time-test_No
    
    
    ##Data Pre-Processing


    dict_user_data = Dict_user_data(user_Id,input_data_csv)
    
    dict_user_window,dict_user_sleep_efficency = Dict_user_window_sf(user_Id,dict_user_data,window＿size,sleep_efficiency_location)

    dict_user_window_diff,dict_user_sf_diff = Dict_user_window_sf_diff(user_Id,dict_user_window,dict_user_sleep_efficency,thr)

    dict_user_X_train,dict_user_Y_train,dict_user_X_test,dict_user_Y_test = Dict_X_Y_seperate(user_Id,dict_user_window_diff,dict_user_sf_diff,train_No)

    X_train,Y_train = X_Y_train(user_Id,dict_user_X_train,dict_user_Y_train)

    ##Building LSTM_DH_G model
    latent_train_vector ,dict_user_X_latent_vector= LSTM_DH(X_train,Y_train,user_Id,dict_user_X_test)

    dict_user_X_predict = Cascade_Forest(test_No,user_Id,latent_train_vector,Y_train,dict_user_X_latent_vector)

    ##Transforming fitting format
    list_day_dict_predict = Standard(test_No,user_Id,dict_user_X_predict)
    list_day_dict_true = Standard(test_No,user_Id,dict_user_Y_test)  

    ##Ranking predicting result in test data
    for day in range(test_No):
        predict_rank = Greedy_TSP(user_Id,list_day_dict_predict[day]) 
        print(" Rank : ",predict_rank)