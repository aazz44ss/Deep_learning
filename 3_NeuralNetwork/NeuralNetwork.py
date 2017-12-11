import numpy as np
import h5py
import math
from numpy import genfromtxt


"""
1. implement convert_one_hot 
2. implement softmax

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
"""

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

X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

"""
X_train = genfromtxt("regularization_train_x.txt")
Y_train = genfromtxt("regularization_train_y.txt",dtype=np.int)
Y_train.resize(1,Y_train.shape[0])
X_test = genfromtxt("regularization_test_x.txt")
Y_test = genfromtxt("regularization_test_y.txt",dtype=np.int)
Y_test.resize(1,Y_test.shape[0])
"""


def relu(Z):
    A = Z*(Z>0)
    return A,Z
def relu_prime(Z):
    temp = 1*(Z>0)
    return temp
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A,Z
def sigmoid_prime(Z):
    a,_ = sigmoid(Z)
    a = a*(1-a)
    return a
def softmax(Z):
    temp = np.exp(Z)
    temp_sum = np.sum(temp,axis=0,keepdims=True)
    A = temp/temp_sum
    return A,Z
def softmax_prime(Z):  #wrong softmax_prime, should be diag(a)-a*a.T, however it's difficult to code if there are m examples
    a = softmax(Z)
    return a-a*a
def cost_prime(AL,Y):
    dAL = - (np.divide(Y, AL+0.00001))  
    return dAL




def initialize_parameters(layers_dims):
    """
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    """
    L = len(layers_dims)
    parameters = {}
    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])*((2/layers_dims[i-1])**(1/2))#((2/layers_dims[i-1])**(1/2)) initialize W properly can make Z (Z=WA+b) at same scale in every layer.
        parameters["b"+str(i)] = np.zeros((layers_dims[i],1))
    return parameters

def initialize_adam(parameters):

    L = len(parameters)//2
    v = {}
    s = {}
    for i in range(L):
        v["dW"+str(i+1)] = np.zeros(parameters["W"+str(i+1)].shape)
        v["db"+str(i+1)] = np.zeros(parameters["b"+str(i+1)].shape)
        s["dW"+str(i+1)] = np.zeros(parameters["W"+str(i+1)].shape)
        s["db"+str(i+1)] = np.zeros(parameters["b"+str(i+1)].shape)
    return v,s

def convert_one_hot(Y,labels):
    m = Y.shape[1]
    temp = np.zeros((labels,m))
    temp[Y,np.arange(m)]=1
    return temp


def linear_forward(A_prev, W, b):

    Z = np.dot(W,A_prev)+b
    linear_cache = A_prev,W,b

    return Z,linear_cache

def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev,W,b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    cache = linear_cache, activation_cache

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    L = len(parameters)//2
    A = X
    for i in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev, parameters["W"+str(i)], parameters["b"+str(i)], "relu")
        caches.append(cache)

    A_prev = A
    AL,cache = linear_activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], "softmax")
    caches.append(cache)

    return AL,caches

def L_model_forward_dropout(X, parameters,dropout_keep_prob):

    caches = []
    cache_dropouts = []
    L = len(parameters)//2
    A = X
    for i in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev, parameters["W"+str(i)], parameters["b"+str(i)], "relu")
        caches.append(cache)

        #dropout terms
        D = np.random.rand(A.shape[0],A.shape[1])
        D = (D < dropout_keep_prob)
        A = A*D/dropout_keep_prob
        cache_dropouts.append(D)
        
    A_prev = A
    AL,cache = linear_activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], "softmax")
    caches.append(cache)

    return AL,caches,cache_dropouts

def compute_cost(AL, Y, parameters, lambd):

    m = Y.shape[1]
    L = len(parameters)//2

    cost = np.sum(-1*(Y*np.log(AL+0.00001)))/m    #  AL offset to avoid somtimes AL close to 1 or 0, and have log(0) error
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    L2_regularization_cost=0
    for i in range(L):
        L2_regularization_cost += np.sum(np.square(parameters["W"+str(i+1)]))
    L2_regularization_cost = np.squeeze(L2_regularization_cost)*lambd/2/m    
    
    total_cost = cost + L2_regularization_cost

    return total_cost

def cost_softmax_backward(AL,Y,cache,lambd):    # will be the same as cost_sigmoid back propagation
    linear_cache, activation_cache = cache
    dZ = AL-Y
    dA_prev,dW,db = linear_backward(dZ,linear_cache,lambd)
    return dA_prev,dW,db

def linear_backward(dZ, linear_cache, lambd):

    A_prev,W,b = linear_cache
    m = A_prev.shape[1]
    
    dW = (np.dot(dZ,A_prev.T)+lambd*W)/m     # d(L2_regularization_cost)/dW term
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation=="relu":
        dZ = dA*relu_prime(Z)
    elif activation=="softmax":
        dZ = dA*softmax_prime(Z)
    elif activation=="sigmoid":
        dZ = dA*sigmoid_prime(Z)
    
    dA_prev,dW,db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev,dW,db

def L_model_backward(AL, Y, caches, lambd):

    grads = {}
    L = len(caches)

#    dAL = cost_prime(AL,Y)
#    dA_prev, dW, db = linear_activation_backward(dAL, caches[L-1], "softmax", lambd)
    dA_prev, dW, db = cost_softmax_backward(AL,Y,caches[L-1],lambd)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(1,L)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu", lambd)
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

    return grads

def L_model_backward_dropout(AL, Y, caches, cache_dropouts, dropout_keep_prob, lambd):

    grads = {}
    L = len(caches)

    dA_prev, dW, db = cost_softmax_backward(AL,Y,caches[L-1],lambd)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(1,L)):
        dA_prev = dA_prev*cache_dropouts[i-1]/dropout_keep_prob #dropout terms
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu", lambd)
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

    return grads

def update_parameters(parameters, grads, v, s, beta1, beta2, learning_rate):

    L = len(parameters) // 2 
    epsilon=0.00001
    for i in range(L):
        v["dW"+str(i+1)] = beta1*v["dW"+str(i+1)] + (1-beta1)*grads["dW"+str(i+1)]
        v["db"+str(i+1)] = beta1*v["db"+str(i+1)] + (1-beta1)*grads["db"+str(i+1)]
        s["dW"+str(i+1)] = beta2*s["dW"+str(i+1)]+(1-beta2)*np.square(grads["dW"+str(i+1)])
        s["db"+str(i+1)] = beta2*s["db"+str(i+1)]+(1-beta2)*np.square(grads["db"+str(i+1)])
        parameters["W"+str(i+1)] = parameters["W"+str(i+1)] - learning_rate*v["dW"+str(i+1)]/(np.sqrt(s["dW"+str(i+1)])+epsilon)*(1-beta2+epsilon) # epsilon avoid divide 0 error
        parameters["b"+str(i+1)] = parameters["b"+str(i+1)] - learning_rate*v["db"+str(i+1)]/(np.sqrt(s["db"+str(i+1)])+epsilon)*(1-beta2+epsilon) # (1-beta2+epsilon) can scale back and achieve close adam if we set beta2=1

    return parameters

def backward_check(parameters, X, Y, epsilon,lambd):
    AL,caches = L_model_forward(X,parameters)
    gradients = L_model_backward(AL,Y,caches,lambd)
    for key, values in parameters.items():
        for i in range(len(values)):
            numerator = 0
            denominator = 0
            for j in range(len(values[i])):

                parameters[key][i][j] += epsilon
                AL, _ = L_model_forward(X,parameters)
                cost_plus = compute_cost(AL, Y, parameters, lambd)
                parameters[key][i][j] -= 2*epsilon
                AL, _ = L_model_forward(X,parameters)
                cost_minus = compute_cost(AL, Y, parameters, lambd)
                parameters[key][i][j] += epsilon

                grad_approx = (cost_plus - cost_minus)/2/epsilon

                numerator += np.linalg.norm(gradients["d"+key][i][j]-grad_approx)            
                denominator += np.linalg.norm(gradients["d"+key][i][j])
                denominator += np.linalg.norm(grad_approx)

        difference = numerator/denominator
        if difference < 2*epsilon:
            print("d"+str(key)+":OK")
        else:
            print("d"+str(key)+":wrong")

def predict(X, parameters):

    AL, cache = L_model_forward(X,parameters)
    predictions = 1*(AL>0.5) #AL is sigmoid function, output bwtween 0 and 1

    return predictions

def initial_batches(X,Y,batch_size):
    m = X.shape[1]
    mini_batches = []

    # shuffled the examples
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    cal_batches = math.ceil(m/batch_size)   # avoid batch_size is greater than m
    for i in range(cal_batches-1):
        mini_batch_X = shuffled_X[:,i*batch_size:(i+1)*batch_size]
        mini_batch_Y = shuffled_Y[:,i*batch_size:(i+1)*batch_size]
        mini_batch = mini_batch_X,mini_batch_Y
        mini_batches.append(mini_batch)
    mini_batch_X = shuffled_X[:,(cal_batches-1)*batch_size:cal_batches*batch_size]
    mini_batch_Y = shuffled_Y[:,(cal_batches-1)*batch_size:cal_batches*batch_size]
    mini_batch = mini_batch_X,mini_batch_Y
    mini_batches.append(mini_batch)
    return mini_batches

def compute_accuracy(prediction,Y):
    m = Y.shape[1]
    accuracy = 0.
    for i in range(m):
        if np.array_equal(prediction.T[i],Y.T[i]):
            accuracy += 1
    return accuracy/m

def L_layer_model(X, Y, layers_dims, learning_rate, num_epochs, batch_size, beta1, beta2, lambd, dropout_keep_prob, print_cost):

    np.random.seed(1)
    parameters = initialize_parameters(layers_dims)
    v,s = initialize_adam(parameters)
    m = Y.shape[1]
#    backward_check(parameters, X, Y, epsilon = 1e-7, lambd=lambd)  #check backward propagation is correct or not

    if dropout_keep_prob == 1.0:
        for i in range(num_epochs):
            mini_batches = initial_batches(X,Y,batch_size)
            for mini_batch in mini_batches:
                mini_batch_X,mini_batch_Y = mini_batch
                AL,caches = L_model_forward(mini_batch_X,parameters)
                grads = L_model_backward(AL, mini_batch_Y, caches, lambd)
                parameters = update_parameters(parameters, grads, v, s, beta1, beta2, learning_rate)
            if print_cost and i%10==0:
                cost = compute_cost(AL,mini_batch_Y,parameters,lambd)
                prediction = predict(X,parameters)
                print("cost after %6d epochs:%.3f ,accuracy: %.2f%%" %(i,cost,compute_accuracy(prediction,Y)*100))
    else:
        for i in range(num_epochs):
            mini_batches = initial_batches(X,Y,batch_size)
            for mini_batch in mini_batches:
                mini_batch_X,mini_batch_Y = mini_batch
                AL,caches,cache_dropouts = L_model_forward_dropout(mini_batch_X,parameters,dropout_keep_prob)
                grads = L_model_backward_dropout(AL,mini_batch_Y,caches,cache_dropouts,dropout_keep_prob, lambd)
                parameters = update_parameters(parameters, grads, v, s, beta1, beta2, learning_rate)
            if print_cost and i%10==0:
                cost = compute_cost(AL,mini_batch_Y,parameters,lambd)
                prediction = predict(X,parameters)
                print("cost after %6d epochs:%.3f ,accuracy: %.2f%%" %(i,cost,compute_accuracy(prediction,Y)*100))

    return parameters

Y_train = convert_one_hot(Y_train,6)
Y_test = convert_one_hot(Y_test,6)

"""
dropout_keep_prob: dropout keep probability (close when set to 1)
lambd: regularization (close when set to 0)
beta1: momentum (close when set to 0)
beta2: RMSprop (close when set to 1)
"""

layers_dims = [X_train.shape[0],40,25,12,6,6,Y_train.shape[0]]
parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.05, num_epochs=10000, batch_size = 32, beta1=0.9, beta2=0.999, lambd=0.8, dropout_keep_prob=1, print_cost=True)

prediction = predict(X_test,parameters)
print("test accuracy:  %.2f%%" %(compute_accuracy(prediction,Y_test)*100))