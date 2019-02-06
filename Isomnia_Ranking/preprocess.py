import pandas as pd
import numpy as np
import pyprind



def max_min_normalization(sleep_activity_nap):
    min_ = sleep_activity_nap.min()[3:]
    max_ = sleep_activity_nap.max()[3:]
    sleep_activity_nap_feature = list(sleep_activity_nap.columns)[3:]
    userId = sleep_activity_nap[["userId","month","date"]]
    sleep_activity_nap = (sleep_activity_nap[sleep_activity_nap_feature]-min_)/(max_-min_)
    sleep_activity_nap = pd.concat([userId,sleep_activity_nap],axis=1)
    return sleep_activity_nap

def Dict_user_data_original(user_Id,month,date,sleep_activity_nap):
    dict_user_data = {}
    for i in pyprind.prog_bar(range(len(user_Id))):
        user_data = []
        for j in range(len(month)):
            for k in range(len(date[j])):
                user_data_date = sleep_activity_nap[(sleep_activity_nap["userId"]==user_Id[i])&(sleep_activity_nap["month"]==month[j])&(sleep_activity_nap["date"]==date[j][k])]
                user_data_constrain = np.array(user_data_date[user_data_date["sleep_end_time"]==max(user_data_date["sleep_end_time"])])[0,3:]
                user_data.append(user_data_constrain)
        user_data = np.array(user_data)
        dict_user_data[user_Id[i]] = user_data
    return dict_user_data

def Dict_user_data(user_Id,sleep_activity_nap):
    dict_user_data = {}
    for i in pyprind.prog_bar(range(len(user_Id))):
        #user_data = []
        #for j in range(len(month)):
            #for k in range(len(date[j])):
        user_data = sleep_activity_nap[(sleep_activity_nap["userId"]==user_Id[i])]
                #user_data_constrain = np.array(user_data_date[user_data_date["sleep_end_time"]==max(user_data_date["sleep_end_time"])])[0,3:]
                #user_data.append(user_data_constrain)
        user_data = np.array(user_data)[:,1:]
        dict_user_data[user_Id[i]] = user_data
    return dict_user_data

def Dict_user_window_sf(user_Id,dict_user_data,window＿size,sleep_efficiency_location):
    dict_user_window,dict_user_sleep_efficency = {},{}
    for i in range(len(user_Id)):
        user_data = dict_user_data[user_Id[i]]
        user_window = []
        user_sleep_efficiency = []
        for j in range(user_data.shape[0]-window＿size):
            user_window.append(user_data[j:j+window＿size,:])
            user_sleep_efficiency.append(user_data[j+window＿size,sleep_efficiency_location])
        dict_user_window[user_Id[i]] = np.array(user_window)
        dict_user_sleep_efficency[user_Id[i]] = np.array(user_sleep_efficiency)
    return dict_user_window,dict_user_sleep_efficency

def Dict_user_window_sf_diff(user_Id,dict_user_window,dict_user_sleep_efficency,thr):
    dict_user_window_diff,dict_user_sf_diff = {},{}
    for i in range(len(user_Id)):
        user_window_diff ,user_sf_diff= [],[]
        for j in range(dict_user_window[user_Id[i]].shape[0]):
            user_sf_diff.append([])
            user_window_diff.append([])
            for k in range(len(user_Id)):
                if user_Id[i] != user_Id[k]:
                    user_window_diff[j].append(dict_user_window[user_Id[i]][j]-dict_user_window[user_Id[k]][j])
                    if dict_user_sleep_efficency[user_Id[i]][j]-dict_user_sleep_efficency[user_Id[k]][j] > thr:
                        user_sf_diff[j].append(0)
                    else:
                        user_sf_diff[j].append(1)

        user_window_diff = np.array(user_window_diff)
        user_sf_diff = np.array(user_sf_diff)
        
        dict_user_window_diff[user_Id[i]] = user_window_diff
        dict_user_sf_diff[user_Id[i]] = user_sf_diff
    return dict_user_window_diff,dict_user_sf_diff

def Dict_X_Y_seperate(user_Id,dict_user_window_diff,dict_user_sf_diff,train_No):

    dict_user_X_train,dict_user_X_test,dict_user_Y_train,dict_user_Y_test = {},{},{},{}
    for i in range(len(user_Id)):
        dict_user_X_train[user_Id[i]] = dict_user_window_diff[user_Id[i]][:train_No,:]
        dict_user_X_test[user_Id[i]] = dict_user_window_diff[user_Id[i]][train_No:,:]
        dict_user_Y_train[user_Id[i]] = dict_user_sf_diff[user_Id[i]][:train_No,:]
        dict_user_Y_test[user_Id[i]] = dict_user_sf_diff[user_Id[i]][train_No:,:]
    return dict_user_X_train,dict_user_Y_train,dict_user_X_test,dict_user_Y_test

def X_Y_train(user_Id,dict_user_X_train,dict_user_Y_train):
    X_train ,Y_train= [],[]
    for i in range(len(user_Id)):
        X_train.append(dict_user_X_train[user_Id[i]])
        Y_train.append(dict_user_Y_train[user_Id[i]])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    train_size = X_train.shape[0]*X_train.shape[1]*X_train.shape[2]
    X_train = X_train.reshape((train_size,X_train.shape[3],X_train.shape[4]))
    Y_train = Y_train.reshape((train_size,1))
    return X_train,Y_train