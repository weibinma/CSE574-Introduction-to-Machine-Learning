'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = np.sqrt(6) / np.sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigm = 1 / (1 + np.exp(-1 * z))
    return sigm # your code here
# Replace this with your nnObjFunction implementation
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
# Replace this with your nnPredict implementation
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

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.
start_time = time.time()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

print('\n')
print("--- %s seconds ---" % (time.time() - start_time))

predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
