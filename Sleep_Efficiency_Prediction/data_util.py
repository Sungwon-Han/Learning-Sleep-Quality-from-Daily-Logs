import numpy as np
from sklearn.preprocessing import MinMaxScaler

def impute_BLANK(save = True):
    _, index, value = load_dataset("RAW")
    zeros = np.zeros([1, 14])
    data = []
    for row in value[:, 1:]:
        row = np.array(row).reshape(-1, len(row))
        if np.any(row[0, :] == -1, axis = 0):
            minus_index = np.argwhere(row == -1)
            row[0, minus_index] = zeros[0, minus_index]   
            
        data.append(row)
    data = np.concatenate([index, np.array(data).reshape(-1, 14)], axis = 1)
    
    if save:
        np.savetxt("../data/sample_data_sleeps_BLANK.csv", data, header=",".join(load_header()), delimiter=",", comments='', fmt='%s')          
    return data

def impute_AVERAGE(save = True):
    _, index, value = load_dataset("RAW")
    prev_user_id = -1
    data = []
    for row in value:
        if prev_user_id != int(row[0]):
            prev_user_id = int(row[0])
            temp = []
     
        row = np.array(row).reshape(-1, len(row))[:, 1:]    
        if len(temp) == 0:
            mean = np.zeros([1, 14])
        else:
            mean = np.mean(np.array(temp), axis = 0)
            
        if np.any(row[0, :] == -1, axis = 0):
            minus_index = np.argwhere(row == -1)
            row[0, minus_index] = mean[0, minus_index]    
        temp.append(row)
        data.append(row)
      
    data = np.concatenate([index, np.array(data).reshape(-1, 14)], axis = 1)
    
    if save:
        np.savetxt("../data/sample_data_sleeps_AVERAGE.csv", data, header=",".join(load_header()), delimiter=",", comments='', fmt='%s')      
    return data

def impute_GAIN():
    data, _, _ = load_dataset("GAIN")
    return data

def load_dataset(mode):
    if mode == "BLANK":
        data = np.genfromtxt("../data/sample_data_sleeps_BLANK.csv", delimiter=',')[1:]
    elif mode == "AVERAGE":
        data = np.genfromtxt("../data/sample_data_sleeps_AVERAGE.csv", delimiter=',')[1:]      
    elif mode == "GAIN":
        data = np.genfromtxt("../data/sample_data_sleeps_Imp-GAIN.csv", delimiter=',')[1:]
    else:
        data = np.genfromtxt("../data/sample_data_sleeps_ORIGINAL.csv", delimiter=',')[1:]
       
    index = data[:, 0:3]
    value = np.concatenate([index[:, 0:1], data[:, 3:17]], axis = 1)        
    return data, index, value

def load_header():
    data = np.genfromtxt("../data/sample_data_sleeps_ORIGINAL.csv", delimiter=',', names = True)[1:]
    return list(data.dtype.names)


def make_window_list(index, value, column_size = 8):
    value = np.concatenate([index, value], axis = 1)
    userX = []
    for userid in np.unique(index):
        user_data = value[np.where(value[:, 0] == userid)][:, 1:]
        userX.append(user_data)

    Sliding_X = []
    for window in userX:
        i = 0
        while i + column_size <= window.shape[0]:
            Sliding_X.append(window[i:i+column_size, :])
            i = i + 1   
    
    return Sliding_X

def get_train_index(raw_data):
    window_list = make_window_list(raw_data[:, 0:1], raw_data[:, 3:])
    window_list.reverse()
    saved_train_index = []
    saved_test_index = []
    index = 0
    testset_count = 0
    for window in window_list:
        if testset_count < 7 and window[-1, 3] != -1: 
            testset_count += 1
            saved_test_index.append(index)
        else:
            saved_train_index.append(index)

        index += 1
        if index % 35 == 0:
            testset_count = 0
    return saved_train_index, saved_test_index

def get_metadata(input_data, loss_list, saved_train_index, saved_test_index):
    input_data_reverse = np.flip(input_data.copy(), 0)
    scaler = MinMaxScaler()
    metadata = np.genfromtxt("../data/sample_meta-data_sleeps.csv", delimiter=',')[1:, 1:]
    user_num = metadata.shape[0]
    
    scaled_loss_list = scaler.fit_transform(np.array(loss_list).reshape(-1, 1)).T   
    tile_loss_list = np.tile(scaled_loss_list, [user_num * 35, 1])
    
    scaled_meta_list = scaler.fit_transform(metadata)
    metadata_X = []
    for i in range(user_num):
        for j in range(35):
            metadata_X.append(scaled_meta_list[i, :])

    metadata_X = np.flip(np.concatenate([np.array(metadata_X), tile_loss_list, input_data_reverse[:, :7, :].reshape(-1, 98)], axis = 1), 0)
    metadata_train = metadata_X[saved_train_index]
    metadata_test = metadata_X[saved_test_index]
    return metadata_train, metadata_test

def preprocess(data, saved_train_index, saved_test_index):
    X = data[:, 3:17]
    minval = np.min(X, axis = 0)[None, :]
    diff = X.max(axis = 0) - X.min(axis = 0)[None, :]

    sleep_efficiency = X[:, 3]
    preprocessed_X = (X - minval) / diff
    preprocessed_X[:, 3] = sleep_efficiency
    window_list = np.array(make_window_list(data[:, 0:1], preprocessed_X)).tolist()
    window_list.reverse()
    training = np.array(window_list)[saved_train_index]
    test = np.array(window_list)[saved_test_index]    
    return window_list, training, test
