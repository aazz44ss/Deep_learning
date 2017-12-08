import numpy as np
import math
from numpy import genfromtxt

"""
針對mini_batch每次更新parameter只針對該次mini_batch更新的缺點做改進
W = W + alpha*dW   (X)
------------------------
v = beta*v+(1-beta)*dW，
W = W + alpha*v
每次更新W都會含有beta比例的前幾次mini_batch的影響，只有(1-beta)比例這次mini_batch影響
"""

def relu(Z):
    A = Z*(Z>0)
    return A,Z
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A,Z
def relu_prime(Z):
    temp = 1*(Z>0)
    return temp
def sigmoid_prime(Z):
    a = 1/(1+np.exp(-Z))
    a = a*(1-a)
    return a
def cost_prime(AL,Y):
    temp = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  #may occur divide 0 error
    return temp


"""
X = genfromtxt('titanic_train_x.txt')
X = X.T   #X would be look like (number of eigenvalues, number of datas)
Y = genfromtxt('titanic_train_y.txt')
Y.resize(1,Y.shape[0]) (Yes No, number of datas)
X_test = genfromtxt('titanic_test.txt')
X_test = X_test.T
"""

X_train = genfromtxt("regularization_train_x.txt")
Y_train = genfromtxt("regularization_train_y.txt")
Y_train.resize(1,Y_train.shape[0])
X_test = genfromtxt("regularization_test_x.txt")
Y_test = genfromtxt("regularization_test_y.txt")
Y_test.resize(1,Y_test.shape[0])

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

def initialize_velocity(parameters):

    L = len(parameters)//2
    v = {}
    for i in range(L):
        v["dW"+str(i+1)] = np.zeros(parameters["W"+str(i+1)].shape)
        v["db"+str(i+1)] = np.zeros(parameters["b"+str(i+1)].shape)
    
    return v

def linear_forward(A_prev, W, b):

    Z = np.dot(W,A_prev)+b
    linear_cache = A_prev,W,b

    return Z,linear_cache

def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev,W,b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

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
    AL,cache = linear_activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
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
    AL,cache = linear_activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    return AL,caches,cache_dropouts

def compute_cost(AL, Y, parameters, lambd):

    m = Y.shape[1]
    L = len(parameters)//2

    cost = np.sum(-1*(Y*np.log(AL+0.0001)+(1-Y)*np.log(1-(AL-0.0001))))/m    #  AL offset to avoid somtimes AL close to 1 or 0, and have log(0) error
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    L2_regularization_cost=0
    for i in range(L):
        L2_regularization_cost += np.sum(np.square(parameters["W"+str(i+1)]))
    L2_regularization_cost = np.squeeze(L2_regularization_cost)*lambd/2/m    
    
    total_cost = cost + L2_regularization_cost

    return total_cost

def cost_sigmoid_backward(AL,Y,cache):
    linear_cache, activation_cache = cache
    dZ = AL-Y
    dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

def linear_backward(dZ, linear_cache):

    A_prev,W,b = linear_cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation=="relu":
        dZ = dA*relu_prime(Z)
    elif activation=="sigmoid":
        dZ = dA*sigmoid_prime(Z)
    
    dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)

#    dAL = cost_prime(AL,Y)
#    dA_prev, dW, db = linear_activation_backward(dAL, caches[L-1], "sigmoid")
    dA_prev, dW, db = cost_sigmoid_backward(AL,Y,caches[L-1])  # put cost and sigmoid backward together can avoid divide 0 error occured in cost backward
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(1,L)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu")
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

    return grads

def L_model_backward_dropout(AL, Y, caches, cache_dropouts, dropout_keep_prob):

    grads = {}
    L = len(caches)

    dA_prev, dW, db = cost_sigmoid_backward(AL,Y,caches[L-1])  # put cost and sigmoid backward together can avoid divide 0 error occured in cost backward
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(1,L)):
        dA_prev = dA_prev*cache_dropouts[i-1]/dropout_keep_prob #dropout terms
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu")
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

    return grads

def update_parameters(parameters, grads, v, beta1, learning_rate,lambd_divide_m):

    L = len(parameters) // 2 

    for i in range(L):
        v["dW"+str(i+1)] = beta1*v["dW"+str(i+1)] + (1-beta1)*(grads["dW"+str(i+1)] + lambd_divide_m*parameters["W"+str(i+1)]) # d(L2_regularization_cost)/dW term
        v["db"+str(i+1)] = beta1*v["db"+str(i+1)] + (1-beta1)*grads["db"+str(i+1)]
        parameters["W"+str(i+1)] = parameters["W"+str(i+1)] - learning_rate*v["dW"+str(i+1)] 
        parameters["b"+str(i+1)] = parameters["b"+str(i+1)] - learning_rate*v["db"+str(i+1)]

    return parameters

def backward_check_compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = np.sum(-1*(Y*np.log(AL)+(1-Y)*np.log(1-(AL))))/m    

    return cost

def backward_check(parameters, X, Y, epsilon):
    AL,caches = L_model_forward(X,parameters)
    gradients = L_model_backward(AL,Y,caches)
    for key, values in parameters.items():
        for i in range(len(values)):
            numerator = 0
            denominator = 0
            for j in range(len(values[i])):

                parameters[key][i][j] += epsilon
                AL, _ = L_model_forward(X,parameters)
                cost_plus = backward_check_compute_cost(AL, Y)
                parameters[key][i][j] -= 2*epsilon
                AL, _ = L_model_forward(X,parameters)
                cost_minus = backward_check_compute_cost(AL, Y)
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
    predictions = 1*(AL > 0.5) #AL is sigmoid function, output bwtween 0 and 1

    return predictions

def initial_batches(X,Y,batch_size):
    m = X.shape[1]
    mini_batches = []
    cal_batches = math.ceil(m/batch_size)   # avoid batch_size is greater than m
    for i in range(cal_batches-1):
        mini_batch_X = X[:,i*batch_size:(i+1)*batch_size]
        mini_batch_Y = Y[:,i*batch_size:(i+1)*batch_size]
        mini_batch = mini_batch_X,mini_batch_Y
        mini_batches.append(mini_batch)
    mini_batch_X = X[:,(cal_batches-1)*batch_size:cal_batches*batch_size]
    mini_batch_Y = Y[:,(cal_batches-1)*batch_size:cal_batches*batch_size]
    mini_batch = mini_batch_X,mini_batch_Y
    mini_batches.append(mini_batch)
    return mini_batches

def L_layer_model(X, Y, layers_dims, learning_rate, num_epochs, batch_size, beta1, lambd, dropout_keep_prob, print_cost):

    np.random.seed(1)
    parameters = initialize_parameters(layers_dims)
    v = initialize_velocity(parameters)
    m = Y.shape[1]

#    backward_check(parameters, X, Y, epsilon = 1e-7)  #check backward propagation is correct or not

    if dropout_keep_prob == 1.0:
        for i in range(num_epochs):
            mini_batches = initial_batches(X,Y,batch_size)
            for mini_batch in mini_batches:
                mini_batch_X,mini_batch_Y = mini_batch
                AL,caches = L_model_forward(mini_batch_X,parameters)
                grads = L_model_backward(AL,mini_batch_Y,caches)
                parameters = update_parameters(parameters, grads, v, beta1, learning_rate,lambd/mini_batch_Y.shape[1])
            if print_cost and i%1000==0:
                cost = compute_cost(AL,mini_batch_Y,parameters,lambd)
                prediction = predict(X,parameters)
                print("cost after %6d epochs:%.3f ,accuracy: %.2f%%" %(i,cost,(100 - np.mean(np.abs(prediction - Y)) * 100)))
    else:
        for i in range(num_epochs):
            mini_batches = initial_batches(X,Y,batch_size)
            for mini_batch in mini_batches:
                mini_batch_X,mini_batch_Y = mini_batch
                AL,caches,cache_dropouts = L_model_forward_dropout(mini_batch_X,parameters,dropout_keep_prob)
                grads = L_model_backward_dropout(AL,mini_batch_Y,caches,cache_dropouts,dropout_keep_prob)
                parameters = update_parameters(parameters, grads, v, beta1, learning_rate,lambd/mini_batch_Y.shape[1])
            if print_cost and i%1000==0:
                cost = compute_cost(AL,mini_batch_Y,parameters,lambd)
                prediction = predict(X,parameters)
                print("cost after %6d epochs:%.3f ,accuracy: %.2f%%" %(i,cost,(100 - np.mean(np.abs(prediction - Y)) * 100)))

    return parameters


layers_dims = [X_train.shape[0],40,30,20,10,Y_train.shape[0]]
parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.01, num_epochs=10000, batch_size = 999, beta1=0.9, lambd=0.8, dropout_keep_prob=1, print_cost=True)


prediction = predict(X_train,parameters)
print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction - Y_train)) * 100))
prediction = predict(X_test,parameters)
print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction - Y_test)) * 100))