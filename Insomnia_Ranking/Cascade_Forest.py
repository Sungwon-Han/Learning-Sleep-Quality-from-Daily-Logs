import numpy as np

import sys
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest as gcForest
from sklearn.metrics import accuracy_score

import multiprocessing as mp

from functools import wraps



def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 1
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    #ca_config["estimators"].append(
            #{"n_folds": 5, "type": "XGBClassifier", "n_estimators": 8  , "max_depth": 5,
            #"objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1,'num_class':2} )
    #ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 8, "max_depth": None, "n_jobs": -1})
    #ca_config["estimators"].append({"n_folds":5, "type": "ExtraTreesClassifier", "n_estimators": 8, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def Cascade_Forest(test_No,user_Id,latent_train_vector,Y_train,dict_user_X_latent_vector):
    config = get_toy_config()
    gcf = gcForest(config) 
    gcf.fit_transform(latent_train_vector,Y_train.reshape((-1,)))
        
    dict_user_X_predict = {}
    for i in range(len(user_Id)):
        user_X_predict = []
        for j in range(test_No):
            user_X_predict.append(gcf.predict_proba(dict_user_X_latent_vector[user_Id[i]][j])[:,1])
        user_X_predict = np.array(user_X_predict)
        dict_user_X_predict[user_Id[i]] = user_X_predict    
    return dict_user_X_predict  


