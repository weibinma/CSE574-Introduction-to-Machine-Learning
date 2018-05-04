import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

feature_list = []


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigm = 1 / (1 + np.exp(-1 * z))
    return sigm  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    ##################################################
    # method 1
    ##################################################
    # zero = np.zeros(train_data.shape[1])
    # trans = train_data.T
    # # print(tans.shape)
    # for i in range(trans.shape[0]):
    #     if np.all(trans[i] == trans[i, 0], 0):
    #         zero[i] = 1
    # zero = (zero != 1)
    #
    # train_data = train_data.T[zero, :].T
    # validation_data = validation_data.T[zero, :].T
    # test_data = test_data.T[zero, :].T
    ##################################################
    # method 2
    ##################################################
    index_list = []
    N = train_data.shape[1]
    for i in range(N):
        if np.all(train_data[:, i] == train_data[0, i], axis=0):
            index_list.append(i)
    # print(index_list)
    train_data = np.delete(train_data, index_list, axis=1)
    validation_data = np.delete(validation_data, index_list, axis=1)
    test_data = np.delete(test_data, index_list, axis=1)

    ####################################################
    #check result
    # for i in range(784):
    #     feature_list.append(i)
    # for i in range(len(index_list)):
    #     feature_list.remove(index_list[i])
    # print(len(feature_list))
    # print(train_data.shape, validation_data.shape, test_data.shape)
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    # format output like 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # training data 50000 * features， assume features = 784



    N = training_data.shape[0]
    format_output = np.zeros((N, n_class))  # 50000 * 10
    for i in range(N):
        format_output[i][int(training_label[i])] = 1

    # comput the output of each layer
    bias1 = np.ones(N)  # bias1 = 784 * 1
    input1 = np.vstack((training_data.T, bias1))  # bias1 = 785 * 50000
    raw1 = np.dot(w1, input1)
    output1 = sigmoid(raw1)  # output = 50 * 785 * 785 * 50000 = 50 * 50000

    bias2 = np.ones(N)
    input2 = np.vstack((output1, bias2))  # input2 = 51 * 50000
    raw2 = np.dot(w2, input2)
    output2 = sigmoid(raw2)  # output = 10 * 51 * 51 * 50000 = 10 * 50000
    ##################################################

    # compute error, obj_val
    ln_output = np.log(output2)  # 10 * 50000
    ln_one_minus_output = np.log(1 - output2)  # 10 * 50000
    J = -1 / N * np.sum(
        np.multiply(format_output, ln_output.T) + np.multiply((1 - format_output), ln_one_minus_output.T))

    ##################################################

    # compute gradient
    delta = output2 - format_output.T  # 10 * 50000 (oj - yl)
    grad_w2 = np.dot(delta, input2.T)  # 10 * 51
    # delta2 =oj * (1 - oj)
    delta_w = np.dot(w2.T, delta)  # (51 * 10)*(10 * 50000) = 51 * 50000
    coefficient = np.multiply(input2, (1 - input2))  # 51 * 50000
    delta3 = np.multiply(coefficient, delta_w)  # mutiply(51 * 50000) * (51 *50000 )= 51 * 50000

    grad_w1 = np.dot(delta3, input1.T)  # (51 * 50000) * (50000 * 785) = 51 * 785
    grad_w1 = grad_w1[0:n_hidden, :]  # 50 * 785
    # print('jhhj')
    ##################################################

    # Regularization in Neural Network
    obj_val = J + (
        lambdaval / (2 * N) * (np.sum(np.multiply(w1, w1)) + np.sum(np.multiply(w2, w2))))

    grad_w1 = (grad_w1 + lambdaval * w1) / N
    grad_w2 = (grad_w2 + lambdaval * w2) / N

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    bias1 = np.ones(data.shape[0])
    input1 = np.vstack((data.T, bias1))  # bias1 = 785 * 10000
    output1 = sigmoid(np.dot(w1, input1))  # 50 * 10000

    bias2 = np.ones(data.shape[0])
    input2 = np.vstack((output1, bias2))  # 51 * 10000
    output = sigmoid(np.dot(w2, input2))  # 10 * 51 * (51 * 10000) = 10 *10000

    output_T = output.T  # 10000 * 10
    labels = np.zeros(output_T.shape[0])  # 10000 * 1
    # print("output shape:",output_T.shape)
    for i in range(data.shape[0]):
        labels[i] = np.argmax(output_T[i])

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# print(feature_list)
print(len(feature_list))
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.
start_time = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
##########################
# write params to pickle
# fd = open('params.pickle', 'wb')
# otherparams = {'list': feature_list, 'n_hidden': n_hidden, 'w1': w1, 'w2': w2, 'lambdaval': lambdaval}
# pickle.dump(otherparams, fd)
# fd.close()
##########################
# Test the computed parameters
# print(w1,'\n','#######',w2)
predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
print('\n')
print("--- %s seconds ---" % (time.time() - start_time))
predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
