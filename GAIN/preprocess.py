import pandas as pd
import numpy as np
import pickle


def Dict_user_data(user_Id,user_data):

    dict_user_data = {}
    for i in range(len(user_Id)):
        user_1 = user_data[user_data["userId"]==user_Id[i]]
        user_1_data = np.array(user_1[user_1.columns[1:]])
        dict_user_data[user_Id[i]] = user_1_data
    return dict_user_data

def Dict_window(user_Id,dict_data):


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

def Window(user_Id,dict_window):

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
    
    user_Id_date = sleep_activity_data_csv[list(sleep_activity_data_csv.columns)[:1]]
    mask_dataframe = sleep_activity_data_csv[list(sleep_activity_data_csv.columns)[1:]].isnull().applymap(lambda x: binary_(x))
    mask_dataframe = pd.concat([user_Id_date,mask_dataframe],axis=1)
    
    return mask_dataframe

def Mask_window(mask_dataframe):
    
    
    dict_mask_data = Dict_user_data(mask_dataframe)
    dict_mask_window,window_mask_number = Dict_window(dict_mask_data,user_Id_date)
    mask_window = Window(dict_mask_window,user_Id_date)
    
    return mask_window

def Delete_window(window_number,window,user_Id,feature_name,max_,min_):
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

    impute_data = []
    for i in range(len(user_Id)):
        for j in range(dict_impute_data[user_Id[i]].shape[0]):
            impute_data.append(dict_impute_data[user_Id[i]][j])
    
    impute_data_Nu = pd.DataFrame(np.array(impute_data),columns=list(feature_name))
    userId = sleep_activity_data_standard[['userId']]

    impute_data_standard = pd.concat([userId,impute_data_Nu],axis=1) #%
    impute_data = min_+(max_-min_)*impute_data_Nu
    impute_data = pd.concat([userId,impute_data],axis=1)
    
    return impute_data_standard,impute_data