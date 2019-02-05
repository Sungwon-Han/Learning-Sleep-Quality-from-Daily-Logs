import numpy as np
import tensorflow as tf

def make_feed_dict(X, metaX, Y, keep_prob, batch_range = None):
    if batch_range == None:
        return {'X1:0': X[0],
                'X2:0': X[1],
                'X3:0': X[2],
                'X4:0': X[3],
                'X5:0': X[4],
                'X6:0': X[5],
                'X7:0': X[6],
                'metaX:0' : metaX,
                'Y:0': Y,
                'keep_prob_ph:0': keep_prob}
    
    return {'X1:0': X[0][batch_range, :],
            'X2:0': X[1][batch_range, :],
            'X3:0': X[2][batch_range, :],
            'X4:0': X[3][batch_range, :],
            'X5:0': X[4][batch_range, :],
            'X6:0': X[5][batch_range, :],
            'X7:0': X[6][batch_range, :],
            'metaX:0' : metaX[batch_range],
            'Y:0': Y[batch_range],
            'keep_prob_ph:0': keep_prob}

def input_to_value(step, input_vec, hidden_size):
    with tf.variable_scope("InputToVal{}".format(step)):
        input_w = tf.Variable(tf.truncated_normal([hidden_size[step], hidden_size[7]], stddev=0.1)) 
        input_b = tf.Variable(tf.constant(0., shape=[hidden_size[7]]))
        input_r = tf.nn.tanh(tf.matmul(input_vec, input_w) + input_b)
        return input_r

def phase2_trainorload(model_name, trainX, trainmetaX, trainY, testX, testmetaX, testY, 
                       batch_size, learning_rate, hidden_size, metadata_size, metadata_hidden_size, num_epochs, keep_prob, 
                       printlog = True, load = False):
    
    tf.reset_default_graph()
    MODEL_PATH = './model'    
    input_list = []
    for step in range(7):
        input_list.append(tf.placeholder(tf.float32, [None, hidden_size[step]], name='X{}'.format(step + 1)))
    
    metadata = tf.placeholder(tf.float32, [None, metadata_size], name='metaX')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
    Y = tf.placeholder(tf.float32, [None], name='Y')
    
    reduced_value_list = []
    for step in range(7):
        reduced_value_list.append(input_to_value(step, input_list[step], hidden_size))
    
    # Fully connected layer
    W1 = tf.Variable(tf.truncated_normal([metadata_size, metadata_hidden_size[0]], stddev=0.1)) 
    b1 = tf.Variable(tf.constant(0., shape=[metadata_hidden_size[0]]))
    r1 = tf.nn.relu(tf.matmul(metadata, W1) + b1)
    
    # Dropout
    drop1 = tf.nn.dropout(r1, keep_prob_ph)
    
    W2 = tf.Variable(tf.truncated_normal([metadata_hidden_size[0], metadata_hidden_size[1]], stddev=0.1)) 
    b2 = tf.Variable(tf.constant(0., shape=[metadata_hidden_size[1]]))
    r2 = tf.nn.relu(tf.matmul(drop1, W2) + b2)  
    
    # Dropout
    drop2 = tf.nn.dropout(r2, keep_prob_ph)

    W3 = tf.Variable(tf.truncated_normal([metadata_hidden_size[1], hidden_size[7]], stddev=0.1)) 
    b3 = tf.Variable(tf.constant(0., shape=[hidden_size[7]]))
    query = tf.nn.tanh(tf.matmul(drop2, W3) + b3)
        
    # Attention
    X_transposed = tf.transpose(tf.stack(reduced_value_list), [1, 0, 2])  
    vu = tf.matmul(tf.nn.l2_normalize(X_transposed, dim = [1, 2]), tf.expand_dims(query, -1))
    alphas = tf.nn.softmax(tf.squeeze(vu), name='alphas')
    output = tf.reduce_sum(X_transposed * tf.expand_dims(alphas, -1), 1)
    
    outW = tf.Variable(tf.truncated_normal([hidden_size[7], 1], stddev=0.1))
    outb = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.squeeze(tf.nn.relu(tf.matmul(output, outW) + outb))

    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.square(y_hat - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()    
    if printlog and load == False:
        log = open('log/log_{}.csv'.format(model_name), 'w')   
        
    total_batch = int(trainX[0].shape[0] / batch_size)
    with tf.Session() as sess:
        sess.run(init)
        if load == True:
            saver.restore(sess, "./{}/{}".format(MODEL_PATH, model_name))
        else:
            for epoch in range(num_epochs):
                loss_train = 0
                for i in range(total_batch):
                    batch_range = range(i * batch_size, (i + 1) * batch_size)
                    loss_tr, _ = sess.run([loss, optimizer],
                                          feed_dict = make_feed_dict(trainX, trainmetaX, trainY, keep_prob, batch_range))
                    loss_train += loss_tr
                loss_train /= total_batch

                if printlog and epoch % 10 == 0:
                    log.write("{} epoch: {}\t training loss: {:.6f}\n".format(model_name, epoch, loss_train))

            saver.save(sess, './{}/{}'.format(MODEL_PATH, model_name))
  
        test_loss, alphas, y_hat = sess.run([loss, alphas, y_hat], 
                                            feed_dict= make_feed_dict(testX, testmetaX, testY, 1.0))
               
        if printlog and load == False:
            log.write("{} final test_loss : {}".format(model_name, test_loss))
            log.close()
            
        return test_loss, alphas, y_hat    