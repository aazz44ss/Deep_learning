import numpy as np
from numpy import genfromtxt


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


def layer_size(X,Y,hidden_layer):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    Returns:
    n_x -- the size of the input layer, number of eigenvalues
    n_h -- the size of the hidden layer, number of layer_1
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_h = hidden_layer
    n_y = Y.shape[0] # size of output layer

    return n_x, n_h, n_y

def initial_parameters(n_x,n_h,n_y):

    W1 = np.random.randn(n_h,n_x)*0.01  #If not initialize randomly, it's only linear fit repeat n_h times.
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X,parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = 1/(1+np.exp(-Z2))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):

    m = Y.shape[1]
    cost = np.sum(-1*(Y*np.log(A2)+(1-Y)*np.log(1-A2)))/m
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    return cost


def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis = 1,keepdims = True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis = 1,keepdims = True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5) #A2 is sigmoid function, a number bwtween 0 and 1

    return predictions

def nn_model(X, Y, n_h, num_iterations, learning_rate, print_cost):

    n_x,n_h,n_y = layer_size(X,Y,n_h)
    parameters = initial_parameters(n_x,n_h,n_y)
    
    for i in range(num_iterations):

        A2,cache = forward_propagation(X,parameters)

        if print_cost and i%1000==0 :
            cost = compute_cost(A2,Y,parameters)
            print("cost after %6d iterations:%.3f" %(i,cost))

        grads = backward_propagation(parameters,cache,X,Y)

        parameters = update_parameters(parameters,grads,learning_rate)


    return parameters

parameters = nn_model(X_train,Y_train, n_h=2, num_iterations=10000, learning_rate=0.01, print_cost=True)
prediction = predict(parameters,X_train)
print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction - Y_train)) * 100))
prediction = predict(parameters,X_test)

