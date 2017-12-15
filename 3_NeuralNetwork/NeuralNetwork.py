import numpy as np
import h5py
import math
from numpy import genfromtxt


"""
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

def relu(Z):
    A = Z*(Z>0)
    return A,Z
def relu_prime(Z):
    temp = 1*(Z>0)
    return temp
def relu_backward(dA,Z):
    relu_prime = 1*(Z>0)
    dA_prev = dA*relu_prime
    return dA_prev
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

def initialize_parameters(layers_dims):
    """
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    """
    np.random.seed(0)
    L = len(layers_dims)
    parameters = {}
    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])*((2/layers_dims[i-1])**(1/2))#((2/layers_dims[i-1])**(1/2)) initialize W properly can make Z (Z=WA+b) at same scale in every layer.
        parameters["b"+str(i)] = np.zeros((layers_dims[i],1))
    return parameters

def conv_initialize_parameters(conv_layers_dims):
    """
    1st layer: Conv -> Relu -> maxPool
    Conv filter     W1 :(f_H, f_W, n_C_prev, n_C)
    Conv bias       b1 :(1, 1, 1, n_C)
    Conv stride     Sc1
    Conv pad        pad   (0 for same)
    maxPool window  k1
    maxPool stride  Sp1
    .....
    """
    np.random.seed(0)
    L = len(conv_layers_dims)//2
    conv_parameters = {}
    for i in range(L):
        conv_parameters["W"+str(i+1)] = np.random.randn(conv_layers_dims["W"+str(i+1)][0],conv_layers_dims["W"+str(i+1)][1],conv_layers_dims["W"+str(i+1)][2],conv_layers_dims["W"+str(i+1)][3])*((2./conv_layers_dims["W"+str(i+1)][0]/conv_layers_dims["W"+str(i+1)][1]/conv_layers_dims["W"+str(i+1)][2])**(0.5))
        conv_parameters["b"+str(i+1)] = np.zeros((1,1,1,conv_layers_dims["b"+str(i+1)][3]))
    return conv_parameters

def initialize_adam(conv_parameters, parameters):

    v = {}
    s = {}

    L = len(conv_parameters)//2
    for i in range(L):
        v["dcon_W"+str(i+1)] = np.zeros(conv_parameters["W"+str(i+1)].shape)
        v["dcon_b"+str(i+1)] = np.zeros(conv_parameters["b"+str(i+1)].shape)
        s["dcon_W"+str(i+1)] = np.zeros(conv_parameters["W"+str(i+1)].shape)
        s["dcon_b"+str(i+1)] = np.zeros(conv_parameters["b"+str(i+1)].shape)

    L = len(parameters)//2
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

def compute_cost(AL, Y, conv_parameters, parameters, lambd):

    m = Y.shape[1]
    
    cost = np.sum(-1*(Y*np.log(AL+0.00001)))/m    #  AL offset to avoid somtimes AL close to 1 or 0, and have log(0) error
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    L2_regularization_cost=0

    L = len(parameters)//2
    for i in range(L):
        L2_regularization_cost += np.sum(np.square(parameters["W"+str(i+1)]))
    L2_regularization_cost = np.squeeze(L2_regularization_cost)*lambd/2/m    
    
    L = len(conv_parameters)//2
    for i in range(L):
        L2_regularization_cost += np.sum(np.square(conv_parameters["W"+str(i+1)]))
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

def update_parameters(conv_parameters, conv_grads, parameters, grads, v, s, t, beta1, beta2, learning_rate):

    epsilon=0.00001

    L = len(conv_parameters)//2
    v_corrected = {}
    s_corrected = {}
    for i in range(L):
        v["dcon_W"+str(i+1)] = beta1*v["dcon_W"+str(i+1)] + (1-beta1)*conv_grads["dW"+str(i+1)]
        v["dcon_b"+str(i+1)] = beta1*v["dcon_b"+str(i+1)] + (1-beta1)*conv_grads["db"+str(i+1)]
        v_corrected["dcon_W" + str(i+1)] = v["dcon_W" + str(i+1)]/(1-beta1**(t))
        v_corrected["dcon_b" + str(i+1)] = v["dcon_b" + str(i+1)]/(1-beta1**(t))
        s["dcon_W"+str(i+1)] = beta2*s["dcon_W"+str(i+1)]+(1-beta2)*np.square(conv_grads["dW"+str(i+1)])
        s["dcon_b"+str(i+1)] = beta2*s["dcon_b"+str(i+1)]+(1-beta2)*np.square(conv_grads["db"+str(i+1)])
        s_corrected["dcon_W" + str(i+1)] = s["dcon_W" + str(i+1)]/(1-beta2**(t))
        s_corrected["dcon_b" + str(i+1)] = s["dcon_b" + str(i+1)]/(1-beta2**(t))
        conv_parameters["W"+str(i+1)] = conv_parameters["W"+str(i+1)] - learning_rate*v_corrected["dcon_W"+str(i+1)]/(np.sqrt(s_corrected["dcon_W"+str(i+1)])+epsilon)
        conv_parameters["b"+str(i+1)] = conv_parameters["b"+str(i+1)] - learning_rate*v_corrected["dcon_b"+str(i+1)]/(np.sqrt(s_corrected["dcon_b"+str(i+1)])+epsilon)
    
    L = len(parameters) // 2
    for i in range(L):
        v["dW"+str(i+1)] = beta1*v["dW"+str(i+1)] + (1-beta1)*grads["dW"+str(i+1)]
        v["db"+str(i+1)] = beta1*v["db"+str(i+1)] + (1-beta1)*grads["db"+str(i+1)]
        v_corrected["dW" + str(i+1)] = v["dW" + str(i+1)]/(1-beta1**(t))
        v_corrected["db" + str(i+1)] = v["db" + str(i+1)]/(1-beta1**(t))
        s["dW"+str(i+1)] = beta2*s["dW"+str(i+1)]+(1-beta2)*np.square(grads["dW"+str(i+1)])
        s["db"+str(i+1)] = beta2*s["db"+str(i+1)]+(1-beta2)*np.square(grads["db"+str(i+1)])
        s_corrected["dW" + str(i+1)] = s["dW" + str(i+1)]/(1-beta2**(t))
        s_corrected["db" + str(i+1)] = s["db" + str(i+1)]/(1-beta2**(t))
        parameters["W"+str(i+1)] = parameters["W"+str(i+1)] - learning_rate*v_corrected["dW"+str(i+1)]/(np.sqrt(s_corrected["dW"+str(i+1)])+epsilon) # epsilon avoid divide 0 error
        parameters["b"+str(i+1)] = parameters["b"+str(i+1)] - learning_rate*v_corrected["db"+str(i+1)]/(np.sqrt(s_corrected["db"+str(i+1)])+epsilon) # (1-beta2+epsilon) can scale back and achieve close adam if we set beta2=1
    return conv_parameters, parameters

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

def zero_pad(X, pad):   # avoid image shrink after convolution or pooling
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    return X_pad

def conv_single_step(a_slice_prev, W, b):

    temp = a_slice_prev*W
    Z = np.sum(temp)+float(b)    # float(b) makes b from 3D to 1D
    return Z

def conv_forward(A_prev, W, b, stride, pad):

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape

    if pad == "same":
        pad = int((A_prev.shape[1]*(stride-1)+W.shape[0]-1)/2)

    n_H = int((n_H_prev-f+2*pad)/stride)+1
    n_W = int((n_W_prev-f+2*pad)/stride)+1

    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev,pad)

    hparameters = {"stride":stride,
                   "pad":pad}

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = conv_single_step(A_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:], W[:,:,:,c], b[:,:,:,c])
                    Z[i, h, w, c] = a_prev_slice
    conv_cache = (A_prev, W, b, hparameters)
    return Z, conv_cache

def pool_forward(A_prev, stride, f, mode):

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    hparameters = {"stride":stride,
                   "f":f}

    n_H = int((n_H_prev-f)/stride)+1
    n_W = int((n_W_prev-f)/stride)+1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i,h,w,c] = np.average(a_prev_slice)

    pool_cache = (A_prev, hparameters)
    return A, pool_cache

def conv_backward(dZ, conv_cache,lambd):

    (A_prev, W, b, hparameters) = conv_cache
    (m, n_H, n_W, n_C) = dZ.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_pad_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    dW[:,:,:,c] += a_prev_pad_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += dZ[i,h,w,c] * W[:,:,:,c]

    dA_prev[:, :, :, :] = dA_prev_pad[:, pad:dA_prev_pad.shape[1]-pad, pad:dA_prev_pad.shape[2]-pad, :]  # clear pad
    dW /= m
    db /= m
    dW += lambd*W/m
    return dA_prev, dW, db

def create_mask_from_window(x):

    mask = (x==np.max(x))
    return mask

def distribute_value(dz, shape):

    n_H, n_W = shape
    average = dz/n_H/n_W
    a = np.ones(shape)*average
    return a

def pool_backward(dA, pool_cache, mode):

    A_prev, hparameters = pool_cache
    f = hparameters["f"]
    stride = hparameters["stride"]

    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w,c]
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, a_prev_slice.shape)
    return dA_prev

def conv_forward_propagation(X, conv_parameters):


    W1 = conv_parameters["W1"]
    b1 = conv_parameters["b1"]
    W2 = conv_parameters["W2"]
    b2 = conv_parameters["b2"]

    conv_caches = []

    Z1, conv_cache = conv_forward(X, W1, b1, stride=1, pad="same")
    A1,_ = relu(Z1)
    P1, pool_cache = pool_forward(A1, stride=4, f=4, mode="max")
    cache = conv_cache, pool_cache, Z1
    conv_caches.append(cache)

    Z2, conv_cache = conv_forward(P1, W2, b2, stride=1, pad="same")
    A2,_ = relu(Z2)
    P2, pool_cache = pool_forward(A2, stride=2, f=2, mode="max")
    cache = conv_cache, pool_cache, Z2
    conv_caches.append(cache)

    return P2, conv_caches

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

def L_model_backward(AL, Y, caches, lambd):

    grads = {}
    L = len(caches)

    dA_prev, dW, db = cost_softmax_backward(AL,Y,caches[L-1],lambd)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for i in reversed(range(1,L)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu", lambd)
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

    return dA_prev, grads

def conv_backward_propagation(dAL, conv_caches,lambd):


    conv_grads = {}

    conv_cache, pool_cache, Z = conv_caches[1]
    dA2 = pool_backward(dAL, pool_cache, mode="max")
    dZ2 = relu_backward(dA2,Z)
    dP1, dW, db = conv_backward(dZ2, conv_cache,lambd)

    conv_grads["dW2"] = dW
    conv_grads["db2"] = db

    conv_cache, pool_cache, Z = conv_caches[0]
    dA1 = pool_backward(dP1, pool_cache, mode="max")
    dZ1 = relu_backward(dA1,Z)
    dP0, dW, db = conv_backward(dZ1, conv_cache,lambd)

    conv_grads["dW1"] = dW
    conv_grads["db1"] = db

    return conv_grads

def predict(X, conv_parameters, parameters):

    m = X.shape[0]
    P, conv_caches = conv_forward_propagation(X, conv_parameters)
    P_flatten = P.reshape(X.shape[0], -1).T
    AL, caches = L_model_forward(P_flatten, parameters)
    
    predictions = AL
    for i in range(m):
        predictions.T[i] = 1*(AL.T[i] == np.max(AL.T[i])) 

    return predictions

def compute_accuracy(prediction,Y):
    m = Y.shape[1]
    accuracy = 0.
    for i in range(m):
        if np.array_equal(prediction.T[i],Y.T[i]):
            accuracy += 1
    return accuracy/m

def conv_backward_check(conv_parameters, parameters, X, Y, epsilon, lambd):
    
    P, conv_caches = conv_forward_propagation(X, conv_parameters)
    P_flatten = P.reshape(P.shape[0], -1).T # flatten P and convert from (m,Eigenvalues) to (Eigenvalues,m)
    Y = Y.T
    AL, caches = L_model_forward(P_flatten, parameters)
    dP_flatten, gradients = L_model_backward(AL, Y, caches, lambd)
    dP = dP_flatten.T.reshape(P.shape)
    conv_grads = conv_backward_propagation(dP, conv_caches, lambd)
    
    for key, values in conv_parameters.items():
        for i in range(len(values)):
            for j in range(len(values[i])):
                for k in range(len(values[i][j])):
                    numerator = 0
                    denominator = 0
                    for l in range(len(values[i][j][k])):
                        conv_parameters[key][i][j][k][l] += epsilon
                        P, conv_caches = conv_forward_propagation(X, conv_parameters)
                        P_flatten = P.reshape(P.shape[0], -1).T
                        AL, _ = L_model_forward(P_flatten,parameters)
                        cost_plus = compute_cost(AL, Y, conv_parameters, parameters, lambd)
                        
                        conv_parameters[key][i][j][k][l] -= 2*epsilon
                        P, conv_caches = conv_forward_propagation(X, conv_parameters)
                        P_flatten = P.reshape(P.shape[0], -1).T
                        AL, _ = L_model_forward(P_flatten,parameters)
                        cost_minus = compute_cost(AL, Y, conv_parameters, parameters, lambd)
                        
                        conv_parameters[key][i][j][k][l] += epsilon
                        
                        grad_approx = (cost_plus - cost_minus)/2/epsilon
                        
                        numerator += np.linalg.norm(conv_grads["d"+key][i][j][k][l]-grad_approx)
                        denominator += np.linalg.norm(conv_grads["d"+key][i][j][k][l])
                        denominator += np.linalg.norm(grad_approx)
                        print("approx: %3f  real:%3f" %(conv_grads["d"+key][i][j][k][l],grad_approx))
    
        difference = numerator/denominator
        if difference < 2*epsilon:
            print("dconv_"+str(key)+":OK")
        else:
            print("dconv_"+str(key)+":wrong")
    
    for key, values in parameters.items():
        for i in range(len(values)):
            numerator = 0
            denominator = 0
            for j in range(len(values[i])):
                
                parameters[key][i][j] += epsilon
                AL, _ = L_model_forward(P_flatten,parameters)
                cost_plus = compute_cost(AL, Y, conv_parameters, parameters, lambd)
                parameters[key][i][j] -= 2*epsilon
                AL, _ = L_model_forward(P_flatten,parameters)
                cost_minus = compute_cost(AL, Y, conv_parameters, parameters, lambd)
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

def CNN_model(X_test, Y_test, X, Y, conv_layers_dims, layers_dims, learning_rate, num_epochs, batch_size, beta1, beta2, lambd, dropout_keep_prob, print_cost):
    
    conv_parameters = conv_initialize_parameters(conv_layers_dims)
    parameters = initialize_parameters(layers_dims)
    v,s = initialize_adam(conv_parameters, parameters)

    t = 0 # adam counter
    seed = 0
    for i in range(num_epochs):
        seed = seed + 1
        mini_batches,num_of_batches = initial_batches(X_train, Y_train, batch_size, seed)
        mini_batch_cost = 0
        for mini_batch in mini_batches:
            mini_batch_X,mini_batch_Y = mini_batch
            P, conv_caches = conv_forward_propagation(mini_batch_X, conv_parameters)
            P_flatten = P.reshape(P.shape[0], -1).T # flatten P and convert from (m,Eigenvalues) to (Eigenvalues,m)
            mini_batch_Y = mini_batch_Y.T
            AL, caches = L_model_forward(P_flatten, parameters)
            if print_cost == True and i % 1 ==0:
                temp_cost = compute_cost(AL, mini_batch_Y, conv_parameters, parameters, lambd)
                mini_batch_cost += temp_cost/num_of_batches
            dP_flatten, grads = L_model_backward(AL, mini_batch_Y, caches, lambd)
            dP = dP_flatten.T.reshape(P.shape)
            conv_grads = conv_backward_propagation(dP, conv_caches,lambd)
            t = t + 1 #adam couner
            conv_parameters, parameters = update_parameters(conv_parameters, conv_grads, parameters, grads, v, s, t, beta1, beta2, learning_rate)
        if print_cost == True and i % 1 ==0:
            #print("train cost after %6d epochs:%.3f" %(i,mini_batch_cost))
            prediction = predict(X_test,conv_parameters,parameters)
            print("train cost after %6d epochs:%.3f ,test_accuracy: %.2f%%" %(i,mini_batch_cost,compute_accuracy(prediction,Y_test.T)*100))
    return conv_parameters, parameters

conv_layers_dims = {"W1":[5,5,3,8],
                    "b1":[1,1,1,8],
                    "W2":[3,3,8,16],
                    "b2":[1,1,1,16]}
"""
Conv    1: in: 64x64x3   out: 64x64x8
ReLu    1: in: 64x64x8   out: 64x64x8
maxPool 1: in: 64x64x8   out: 16x16x8
Conv    2: in: 16x16x8   out:16x16x16
ReLu    2: in:16x16x16   out:16x16x16
maxPool 2: in:16x16x16  out:   8x8x16
FC+Relu 1: in:   1024   out:    300
FC+Relu 2: in:    300   out:     60
FC+Relu 3: in:     60   out:     30
FC      4: in:     30   out:      6
softmax  : in       6   out:      6
"""
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_one_hot(Y_train_orig, 6).T
Y_test = convert_one_hot(Y_test_orig, 6).T

layers_dims = [1024,300,60,30, Y_train.shape[1]]

conv_parameters, parameters = CNN_model(X_test, Y_test, X_train, Y_train, conv_layers_dims, layers_dims, learning_rate=0.001, num_epochs=100, batch_size = 256, beta1=0.9, beta2=0.999, lambd=0, dropout_keep_prob=1, print_cost=True)
prediction = predict(X_test,conv_parameters,parameters)
print("cost after %6d epochs:%.3f ,accuracy: %.2f%%" %(i,cost,compute_accuracy(prediction,Y_test)*100))