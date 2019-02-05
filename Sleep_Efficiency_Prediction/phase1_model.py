import numpy as np
import tensorflow as tf

def attention(inputs, sequence_length):
    hidden_size = inputs.shape[2].value

    query_w = tf.Variable(tf.random_normal([sequence_length * hidden_size, hidden_size], stddev=0.1))
    query_b = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
    query = tf.tanh(tf.tensordot(tf.reshape(inputs, [-1, sequence_length * hidden_size]), query_w, axes=1) + query_b)
    resized_inputs = tf.nn.l2_normalize(inputs, dim = [1, 2])
    
    vu = tf.squeeze(tf.matmul(resized_inputs, tf.expand_dims(query, -1)))
    alphas = tf.nn.softmax(vu, name='alphas')
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    return output, alphas


def phase1_trainorload(model_name, training, test, sequence_length, batch_size, learning_rate, hidden_size, num_epochs, keep_prob, 
                       printlog = True, load = False):
    tf.reset_default_graph()
    MODEL_PATH = './model'
        
    X = tf.placeholder(tf.float32, [None, sequence_length, 14], name='X')
    Y = tf.placeholder(tf.float32, [None], name='Y')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

    # LSTM layer
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

    # Attention layer
    attention_output, alphas = attention(rnn_outputs, sequence_length)

    # Dropout
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

    W2 = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.1)) 
    b2 = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.relu(tf.matmul(drop, W2) + b2)
    y_hat = tf.squeeze(y_hat)

    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.square(y_hat - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)   
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    total_batch = int(training.shape[0] / batch_size)
    if printlog and load == False:
        log = open('log/log_{}.csv'.format(model_name), 'w')
        
    with tf.Session() as sess:
        sess.run(init)
        if load == True:
            saver.restore(sess, "./{}/{}".format(MODEL_PATH, model_name))
        else:
            for epoch in range(num_epochs):
                loss_train = 0
                for i in range(total_batch): 
                    batch = training[i * batch_size : (i + 1) * batch_size]
                    batch_xs = batch[:, 7 - sequence_length:7, :]
                    batch_ys = batch[:, 7, 3]
                    loss_tr, _ = sess.run([loss, optimizer], 
                                          feed_dict={X: batch_xs, Y: batch_ys, keep_prob_ph: 0.8})
                    loss_train += loss_tr
                loss_train /= total_batch

                if printlog and epoch % 10 == 0:
                    log.write("{} epoch: {}\t training loss: {:.6f}\n".format(model_name, epoch, loss_train))

            saver.save(sess, "./{}/{}".format(MODEL_PATH, model_name))
            
        test_loss, alphas, y_hat, test_phase2_h = sess.run([loss, alphas, y_hat, attention_output],
                                                           feed_dict={X: test[:,7 - sequence_length:7,:],
                                                                      Y: test[:,7,3],
                                                                      keep_prob_ph: 1.0})

        training_phase2_h = sess.run(attention_output,
                                     feed_dict={X: training[:,7 - sequence_length:7,:],
                                                Y: training[:,7,3],
                                                keep_prob_ph: 1.0})
        
        if printlog and load == False:
            log.write("{} final test loss: {}".format(model_name, test_loss))
            log.close()
            
        return test_loss, alphas, y_hat, training_phase2_h, test_phase2_h