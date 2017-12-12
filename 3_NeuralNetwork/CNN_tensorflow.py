import math
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops

def load_dataset():
    train_dataset = h5py.File('signs_train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('signs_test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_one_hot(Y,labels):
    m = Y.shape[1]
    temp = np.zeros((labels,m))
    temp[Y,np.arange(m)]=1
    return temp

def initial_batches(X,Y,batch_size,seed):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # shuffled the examples
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))

    cal_batches = math.ceil(m/batch_size)   # avoid batch_size is greater than m
    cal_batches = int(cal_batches)
    for i in range(cal_batches-1):
        mini_batch_X = shuffled_X[i*batch_size:(i+1)*batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[i*batch_size:(i+1)*batch_size,:]
        mini_batch = mini_batch_X,mini_batch_Y
        mini_batches.append(mini_batch)
    mini_batch_X = shuffled_X[(cal_batches-1)*batch_size:cal_batches*batch_size,:,:,:]
    mini_batch_Y = shuffled_Y[(cal_batches-1)*batch_size:cal_batches*batch_size,:]
    mini_batch = mini_batch_X,mini_batch_Y
    mini_batches.append(mini_batch)
    return mini_batches, cal_batches

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    #use None to let m examples to initialize it later
    X = tf.placeholder(shape=[None, n_H0, n_W0, n_C0], dtype=tf.float32, name="X")
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32, name="Y")
    
    return X, Y

def initialize_parameters():

    tf.set_random_seed(0)                             
    
    W1 = tf.get_variable(name="W1", shape=[5,5,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable(name="W2", shape=[3,3,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def forward_propagation(X, parameters):

    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 300, activation_fn=tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer(seed = 0)) 
    Z3 = tf.contrib.layers.fully_connected(Z3, 60, activation_fn=tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    Z3 = tf.contrib.layers.fully_connected(Z3, 30, activation_fn=tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    Z3 = tf.contrib.layers.fully_connected(Z3, 6, activation_fn=None, weights_initializer = tf.contrib.layers.xavier_initializer(seed = 0))
                                              #don't use sofmax activation, we'll use it later.
    return Z3

def compute_cost(Z3, Y, parameters, lambd):

    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)

    L = len(parameters)
    L2_cost = 0
    for i in range(L):
        L2_cost += lambd * tf.nn.l2_loss(parameters["W"+str(i+1)])
    
    total_cost = cost + L2_cost
    total_cost = tf.reduce_mean(total_cost)
    
    return total_cost

def model(X_train, Y_train, X_test, Y_test, learning_rate,
          num_epochs, minibatch_size, lambd, print_cost):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    seed = 1                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y,parameters,lambd)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            seed = seed + 1
            minibatches,num_minibatches = initial_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch % 1 == 0:
                predict_op = tf.argmax(Z3, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
                print ("Cost after epoch %i: %.3f , Train : %.3f, Test : %.3f" % (epoch, minibatch_cost, train_accuracy, test_accuracy))

        return train_accuracy, test_accuracy, parameters

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_one_hot(Y_train_orig, 6).T
Y_test = convert_one_hot(Y_test_orig, 6).T

_, _, parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001, num_epochs = 20000, minibatch_size = 128, lambd = 0, print_cost = True)