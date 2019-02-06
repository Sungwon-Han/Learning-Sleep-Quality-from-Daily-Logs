import tensorflow as tf
import numpy as np
import pyprind

def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

def v2_tool(X_train):
    
    one_1 = np.array([1 for i in range(X_train.shape[0]*X_train.shape[1]*(X_train.shape[2]-1))]).reshape((X_train.shape[0],X_train.shape[1],(X_train.shape[2]-1)))
    one_2 = np.array([1 for i in range(X_train.shape[0]*X_train.shape[1]*(X_train.shape[2]))]).reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2]))

    TS = np.array([[ [X_train[i,j,3] for k in range(X_train.shape[2]-1)]for j in range(X_train.shape[1])] for i in range(X_train.shape[0])])    
    TO = []
    for i in range(X_train.shape[0]):
        TO.append([])
        for j in range(X_train.shape[1]):
            TO[i].append([])
            for k in range(X_train.shape[2]):
                if k != 3 :
                    TO[i][j].append(X_train[i,j,k])
    TO = np.array(TO)

    return one_1,one_2,TS,TO


def v2_test_tool( user_Id ,dict_user_X_test):
    dict_user_v2_tool = {}
    for i in range(len(user_Id)):
        user_v2_tool = []
        for j in range(dict_user_X_test[user_Id[i]].shape[0]):
            user_one_1,user_one_2,user_TS,user_TO = v2_tool(dict_user_X_test[user_Id[i]][j])
            user_v2_tool.append([user_one_1,user_one_2,user_TS,user_TO])

        dict_user_v2_tool[user_Id[i]] = user_v2_tool
    return dict_user_v2_tool



def LSTM_DH2(X_train,Y_train,user_Id,dict_user_X_test):
    tf.set_random_seed(1)
    np.random.seed(1)

    # Hyper Parameters
    BATCH_SIZE = 64
    TIME_STEP = 8          # rnn time step / image height
    INPUT_SIZE = 14        # rnn input size / image width
    LR = 0.01               # learning rate
    hidden_state_dim = 27

    one_1,one_2,TS,TO = v2_tool(X_train)

            

    # tensorflow placeholders
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP , INPUT_SIZE])     
    tf_y = tf.placeholder(tf.float32, [None, 1])  

    tf_one_1 = tf.placeholder(tf.float32,[None,TIME_STEP,INPUT_SIZE-1])
    tf_TS = tf.placeholder(tf.float32,[None,TIME_STEP,INPUT_SIZE-1])  
    tf_TO = tf.placeholder(tf.float32,[None,TIME_STEP,INPUT_SIZE-1])
    tf_one_2 = tf.placeholder(tf.float32,[None,TIME_STEP,INPUT_SIZE])                           

    # RNN
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_state_dim)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,                   # cell you have chosen
        tf_x,                      # input
        initial_state=None,         # the initial hidden state
        dtype=tf.float32,           # must given if set initial_state = None
        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    )

    D_hat = tf.concat([tf_x,tf_one_1],2)
    D_hat_h = D_hat*outputs

    W1 = tf.Variable(tf.random_normal([TIME_STEP,INPUT_SIZE-1]),dtype=tf.float32)
    W2 = tf.Variable(tf.random_normal([TIME_STEP,INPUT_SIZE-1]),dtype=tf.float32)

    WT = W1*tf_TS+W2*tf_TO
    W_hat = tf.concat([tf_one_2,WT],2)
    WD_hat_H = W_hat*D_hat_h

    sum_WD_hat_H = tf.reduce_sum(WD_hat_H,1,name="latent_vector")
    output = tf.layers.dense(sum_WD_hat_H, 1)            
    output = tf.sigmoid(output)


    #Weights = tf.Variable(tf.random_normal([TIME_STEP,hidden_state_dim]),dtype=tf.float32)
    #outputs2 = Weights*outputs
    #outputs3 = tf.reduce_sum(outputs2,axis=1,name="latent_vector")
    #output = tf.layers.dense(outputs3, 1)             # output based on the last output step
    #output = tf.sigmoid(output)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=tf_y)

    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    
    sess = tf.Session()

    

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    for step in pyprind.prog_bar(range(1000)):    # training

        mb_idx = sample_idx(X_train.shape[0], BATCH_SIZE)
        X_mb = X_train[mb_idx,:]
        Y_mb = Y_train[mb_idx]
        one_1_mb = one_1[mb_idx,:]
        one_2_mb = one_2[mb_idx,:]
        TS_mb = TS[mb_idx,:]
        TO_mb = TO[mb_idx,:]

        _, loss_ = sess.run([train_op, loss], {tf_x: X_mb, tf_y: Y_mb,tf_one_1:one_1_mb,tf_one_2:one_2_mb,tf_TS:TS_mb,tf_TO:TO_mb})
    
    latent_vector =  sess.graph.get_operation_by_name("latent_vector").outputs[0]
    latent_train_vector = sess.run(latent_vector,{tf_x: X_train,tf_one_1:one_1,tf_one_2:one_2,tf_TS:TS,tf_TO:TO})   #%
    
    v2_test = v2_test_tool(user_Id,dict_user_X_test)

    dict_user_X_latent_vector = {}
    for i in range(len(user_Id)):
        user_X_latent_vector = []
        for j in range(dict_user_X_test[user_Id[i]].shape[0]):

            _one_1 = v2_test[user_Id[i]][j][0]
            _one_2 = v2_test[user_Id[i]][j][1]
            _TS = v2_test[user_Id[i]][j][2]
            _TO = v2_test[user_Id[i]][j][3]

            user_X_latent_vector.append(sess.run(latent_vector,{tf_x: dict_user_X_test[user_Id[i]][j],tf_one_1:_one_1,tf_one_2:_one_2,tf_TS:_TS,tf_TO:_TO}))
        user_X_latent_vector = np.array(user_X_latent_vector)
        dict_user_X_latent_vector[user_Id[i]] = user_X_latent_vector   



    return latent_train_vector,dict_user_X_latent_vector
