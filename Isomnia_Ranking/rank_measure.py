import numpy as np

def Rank(day,test_No,user_Id,dict_user_X_predict_Y):
    rank = []
    for i in range(test_No):
        rank.append([])
        for j in range(len(user_Id)):
            rank[i].append(dict_user_X_predict_Y[user_Id[j]][i].shape[0]-np.sum(dict_user_X_predict_Y[user_Id[j]][i])+1)
    return rank[day]

def Rank_sort(True_data,Predicted_data):

    from scipy.stats import rankdata

    b = sorted([(True_data[i],Predicted_data[i]) for i in range(len(True_data))])
    rank = [b[i][1] for i in range(len(b))]
    #rank = rankdata(b_1, method='dense')
    return rank

def Greedy_TSP(location,dict_distance):
    all_K = []
    all_score = []
    for init_location in location:
        K,indivisual_score = One_point_Greedy_TSP(init_location,location,dict_distance) 
        reverse_score = Reverse_score(K,dict_distance)
        all_K.append(K)
        all_score.append(sum(indivisual_score)+reverse_score)
    all_score_rank = []
    for i in range(len(all_score)):
        all_score_rank.append((all_score[i],all_K[i]))
    rank = min(all_score_rank)[1]   
    return rank


def One_point_Greedy_TSP(init_location,location,dict_distance):

    S = init_location
    K = []
    K.append(S)
    score = []
    while len(K) < len(dict_distance):
        I = []
        for j in range(len(location)):
            if location[j] not in K:
                I.append(location[j])
        dict_I_distance = {}
        indivisual_score = []
        for k in range(len(I)):
            dict_I_distance[I[k]] = dict_distance[S][I[k]]
            indivisual_score.append(dict_distance[S][I[k]])
        min_score = min(indivisual_score)
        score.append(min_score)

        S = min(dict_I_distance,key=dict_I_distance.get)
        K.append(S)
    return K,score
    
def Reverse_score(rank,dict_distance):
    indivisual_reverse_score = []
    for i in range(len(rank)-1):
        indivisual_reverse_score.append(dict_distance[rank[i]][rank[i+1]])
    reverse_score = sum(indivisual_reverse_score)
    return reverse_score

#dict_user_X_preodict dict user-6-41

def Standard(test_No,user_Id,dict_user_X_preodict):
    list_day_dict_X_predict = []
    for i in range(test_No):
        day_dict_X_predict = {}
        for j in range(len(user_Id)):
            day_dict_X_predict[user_Id[j]] = dict_user_X_preodict[user_Id[j]][i]
        list_day_dict_X_predict.append(day_dict_X_predict)
    list_day_dict_X_preidct_all_user = []
    for i in range(test_No):
        day_dict_X_predict_standard = {}
        for j in range(len(user_Id)):
            X_predict = []
            for k in range(len(user_Id)):
                if j == k  :
                    X_predict.append(0)
                    if k != len(user_Id)-1:
                        X_predict.append(list_day_dict_X_predict[i][user_Id[j]][k])

                else:
                    if k != len(user_Id)-1:
                        X_predict.append(list_day_dict_X_predict[i][user_Id[j]][k])
                    #if j==len(user_Id)-1 and k == len(user_Id)-2:
                       # X_predict.append(0) 

            X_predict = np.array(X_predict)
            day_dict_X_predict_standard[user_Id[j]] = X_predict
        list_day_dict_X_preidct_all_user.append(day_dict_X_predict_standard)

    list_day_dict_X_preidct_standard = []
    for i in range(test_No):
        dict_X_predict_standard = {}
        for j in range(len(user_Id)): 
            dict_X_predict_standard[user_Id[j]] = {}
            for k in range(len(user_Id)):                                        
                dict_X_predict_standard[user_Id[j]][user_Id[k]] = list_day_dict_X_preidct_all_user[i][user_Id[j]][k]
        dict_X_predict_standard[user_Id[j]] = dict_X_predict_standard[user_Id[j]]
        list_day_dict_X_preidct_standard.append(dict_X_predict_standard)                

    return list_day_dict_X_preidct_standard
        
    

    








