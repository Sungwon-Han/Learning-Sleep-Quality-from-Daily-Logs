import pandas as pd

import pandas as pd
import numpy as np
import random
import pickle


def Correct(sleep_activity_data_csv):
    for i in range(sleep_activity_data_csv.shape[0]):
        for j in range(sleep_activity_data_csv.shape[1]):
            if sleep_activity_data_csv.iloc[i,j] == -1:
                sleep_activity_data_csv.iloc[i,j] = np.nan
    return sleep_activity_data_csv

def Dict_user_data(user_data):
    user_Id = list(set(user_data["userId"]))

    dict_user_data = {}
    for i in range(len(user_Id)):
        user_1 = user_data[user_data["userId"]==user_Id[i]]
        user_1_data = np.array(user_1[user_1.columns[3:]])
        dict_user_data[user_Id[i]] = user_1_data
    return dict_user_data

def Dict_window(dict_data,user_Id_date):

    user_Id = list(set(user_Id_date["userId"]))

    dict_window = {}
    window_number = []
    for Id in user_Id:
        dict_user_data_1 = dict_data[Id]
        window_1 = []
        for i in range(dict_user_data_1.shape[0]-8+1):
            window_1.append([dict_user_data_1[i+j] for j in range(8)])
        window_1 = np.array(window_1)
        window_number.append(window_1.shape[0])
        dict_window[Id] = window_1
    return dict_window,window_number

def Window(dict_window,user_Id_date):

    user_Id = list(set(user_Id_date["userId"]))
    window = []
    for i in range(len(user_Id)):
        for j in range(dict_window[user_Id[i]].shape[0]):
            window.append(dict_window[user_Id[i]][j])
    window = np.array(window)
    return window

def Mask_dataframe(sleep_activity_data_csv):

    def binary_(x):
        if x is False:
            x = 1
        else:
            x = 0
        return x
    
    user_Id_date = sleep_activity_data_csv[list(sleep_activity_data_csv.columns)[:3]]
    mask_dataframe = sleep_activity_data_csv[list(sleep_activity_data_csv.columns)[3:]].isnull().applymap(lambda x: binary_(x))
    mask_dataframe = pd.concat([user_Id_date,mask_dataframe],axis=1)
    
    return mask_dataframe

def Mask_window(user_Id_date,mask_dataframe):
    
    
    dict_mask_data = Dict_user_data(mask_dataframe)
    dict_mask_window,window_mask_number = Dict_window(dict_mask_data,user_Id_date)
    mask_window = Window(dict_mask_window,user_Id_date)
    
    return mask_window

def Delete_window(window_number,window,user_Id,feature_name,max_,min_,user_Id_date):
    window_add_number = np.cumsum(window_number)

    dict_impute_window = {}
    for i in range(len(user_Id)):
        if i==0:
            dict_impute_window[user_Id[i]] = window[:window_add_number[i]]
        else:
            dict_impute_window[user_Id[i]] = window[window_add_number[i-1]:window_add_number[i]]
            
    dict_impute_data = {}
    for i in range(len(user_Id)):
        user_window = dict_impute_window[user_Id[i]]
        user_data = []
        for j in range(user_window.shape[0]):
            if j != user_window.shape[0]-1:
                user_data.append(user_window[j,0])
            else:
                for k in range(user_window[j].shape[0]):
                    user_data.append(user_window[j][k])
                #user_data.append(user_window[j])
        dict_impute_data[user_Id[i]] = np.array(user_data)
    print(dict_impute_data[1].shape)
    impute_data = []
    for i in range(len(user_Id)):
        for j in range(dict_impute_data[user_Id[i]].shape[0]):
            impute_data.append(dict_impute_data[user_Id[i]][j])
    
    impute_data = np.array(impute_data)

    impute_data_Nu = pd.DataFrame(np.array(impute_data),columns=feature_name)
    impute_data_standard = pd.concat([user_Id_date,impute_data_Nu],axis=1) #%
    impute_data = min_+(max_-min_)*impute_data_Nu
    impute_data = pd.concat([user_Id_date,impute_data],axis=1)
    return impute_data_standard,impute_data
