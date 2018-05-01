
# coding: utf-8

# In[16]:


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd
from scipy.stats import multivariate_normal
import time

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    X = pd.DataFrame(X)
    means = []
    y_labels = np.unique(y)
    for i in y_labels:
        means.append(np.mean(X[y==i],axis=0))
    np.asarray(means) # mean for each class according to X1,X2  -> 5x2 matrix
    means = np.transpose(means) #2x5 matrix 
    
    covs = []
    for i in y_labels:
        covs.append(np.cov(X[y==i].T))        
    covmat = sum(covs) / len(covs)
    return means,covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ytest = ytest[:, 0]
    prob = np.zeros((len(np.unique(y)), Xtest.shape[0]))
    for i, m in enumerate(means.T):
        prob[i, :] = multivariate_normal.pdf(Xtest, m, covmat)
    ypred = np.argmax(prob, axis=0) + 1  #index begin from 0,which need to add 1 so as to match the class 1,2,3,4,5
    acc = (ypred == ytest).sum() / ytest.shape[0]
    return acc,ypred

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    X = pd.DataFrame(X)
    means = []
    y_labels = np.unique(y)
    for i in y_labels:
        means.append(np.mean(X[y==i],axis=0))
    np.asarray(means) # mean for each class according to X1,X2  -> 5x2 matrix
    means = np.transpose(means) #2x5 matrix
    #covmats - A list of k d x d learnt covariance matrices for each of the k classes
    covmats = []
    for i in y_labels:
        covmats.append(np.cov(X[y==i].T)) 
    return means,covmats

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ytest = ytest[:, 0]  ##y[:, 0] #np.ravel(y) make dimension of y become 1 dimension
    prob = np.zeros((len(np.unique(y)), Xtest.shape[0]))
    for i, (m, c) in enumerate(zip(means.T, covmats)):
        prob[i, :] = multivariate_normal.pdf(Xtest, m, c)
    ypred = np.argmax(prob, axis=0) + 1
    acc = (ypred == ytest).sum() / ytest.shape[0]

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    w = (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    mse = np.dot((ytest - np.dot(Xtest,w)).T, (ytest - np.dot(Xtest,w))) / N
    return mse

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    lambda_I = lambd * np.eye(X.shape[1])
    w = (np.linalg.inv(X.T.dot(X) + lambda_I)).dot(X.T).dot(y) 
    return w

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = np.reshape(w,(w.size, 1))
    #compute error
    error = (np.dot((y - np.dot(X, w)).T, (y - np.dot(X, w)))) / 2 + (lambd * np.dot(w.T, w)) / 2
    
    #compute error_grad
    error_grad = -(X.T.dot(y - X.dot(w))) + lambd * w  #derivate error 
    error_grad = error_grad.flatten()
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = np.shape(x)[0]
    Xp = np.empty([N, p + 1])
    for i in range(N):
        for j in range(p + 1):
            Xp[i][j] = np.power(x[i], j)  #value for x^p 
    return Xp


# Problem 1

# In[17]:


###########
##Problem1
###########
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
fig.savefig('LDA & QDA')


# From this two plot above, I both obtained high accuracy for LDA and QDA. LDA is more ideal than QDA since each class possesses the same covariance, so it is assumpted to be a linear classifier. But for QDA, each class has different covariance matrix, so QDA is a quadratic classifier. Thus, there are a little different in the two boundaries. For LDA, the covariance is only linear, since the classes share the same covirance matrix, which means each class has the same shape for grouping, so we can consider the median line between the center of two classes as coundary which is linear. For QDA, since each class has different covariance matrix. So, after reflecting, their boundaries are quadratic so as to be more flexible, practical and experimental.

# Problem 2

# In[18]:


###########
##Problem2
###########
start_time = time.time()

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

end_time = time.time()
time_linear = end_time - start_time
print('MSE without intercept : '+str(mle[0][0]))
print('MSE with intercept : '+str(mle_i[0][0]))
print('Time Consuming :', time_linear)


# From the final result, we can find that the MSE with intercept is smaller than the MSE without intercept. This is because the line learnt from training sample must pass through origin, which means is limited to rotate. On the other hand, when adding the intercept, the line is much more flexible and is much more practical and persuasive. And the accuracy is higher, about 29 times, than without adding intercept. Thus, adding the bias term is better.

# Problem 3

# In[19]:


###########
##Problem3
###########
#add intercept
start_time = time.time()

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')
    
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

k = 101
lambdas = np.linspace(0, 1, num=k) # the value of lambda is very small, so I set it in range(0, 0.01)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#find optimal value for lambda
y_min_ridge = min(mses3)
x_min_ridge = lambdas[mses3.argmin()]
#comparing weight between OLE and Ridge Regression
w_OLE_total = np.sum(w_i)
w_ridg_total = np.sum(w_l)
w_OLE_var = np.var(w_i)
w_ridg_var = np.var(w_l)

end_time = time.time()
time_ridge = end_time - start_time
print("The optimal value for lambda is : ", x_min_ridge, ", where the MSE is : ", y_min_ridge[0])
print("The total sum of weight learnt using OLE is : ", w_OLE_total)
print("The total sum of weight learnt using Ridge Regression is : ", w_ridg_total)
print("The variance of weight learnt using OLE is : ", w_OLE_var)
print("The variance of weight learnt using Ridge Regression is : ", w_ridg_var)
print('Time Consuming :', time_ridge)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
fig.savefig('MSE for train data and test data')

plt.show()


# As known, ridge regression is used for dealing with overfitting. From the first two plots, I find that the MSE for the Test Data shows a 'V' shape, since when labda = 0, it can't control the weights for overfitting, but as lambda increases, the lambda term in objective function become real values which will minimize the squared loss. So, as this point, the error in test data will begin to decrease. However, as lambda continues increasing, the test error raises again, this is because if the lambda is high, it will restrict the growing of weight in learing process and then generates underfitting in turn in test data.
# So the optimal value for lambda is equal to the minimized value of lambda in plot. By computing, the optimal lambda is equal to 0.06 where the MSE value is equal to 2851.33021344
# 
# From the third plot, I find weights learnt using ridge regression is more stable and linear while the weights learnt using OLE is much more wavy and higher. Moreover, the variance for OLE is much higher than the variance for ridge regression, The variance of weight learnt using OLE is :237806821.049 and The variance of weight learnt using Ridge Regression is only :2546.2691635. Since the ridge regression uses regularization and the lambda part in ridge regression can help control the value of weights. Thus, the value of weights obtained is in a much lower magnitude and more stable after ridge regression.

# Problem 4

# In[20]:


###########
##Problem4
###########
#add intercept
start_time = time.time()

# X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
# Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

    
#find optimal value for lambda
y_min_grad = min(mses4)
x_min_grad = lambdas[mses4.argmin()]

end_time = time.time()
time_grad = end_time - start_time
print("The optimal value for lambda is : ", x_min_grad, ", where the MSE is : ", y_min_grad[0])
print('Time Consuming :', time_grad)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

fig.savefig('gradient descent and regular ridge regression in MSE')
plt.show()


# As given information, gradient descent is used to minimize the loss function. From the plots above, I find that the plot for minimizing the lose function with gradient descent is similar with the plot for minimizing the regularized squared loss with ridge regression. They both have similar weight values and variance. Both of the MSE show a 'V' shape as lambd increases as well. But, there is a obvious difference between this two methods. When the matrix is very laege and if we use regular method to minimize the loss function for ridge regression, the covariance matrix will become unstable due to the inverse term in the formula. So, at this point, we can use gradient descent method which has similar effect for minimizing loss function to replace regular method.

# Problem 5

# In[21]:


###########
##Problem5
###########
start_time = time.time()

pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    
#find the optimal value for p in terms of training error
y_min1 = min(mses5_train[:,0])
x_min1 = mses5_train[:,0].argmin()
print("The optimal value for p in training data is : ", x_min1, ", where the MSE is : ", y_min1)
    
#find the optimal value for p in terms of test error
y_min_nonlinear = min(mses5[:,0])
x_min_nonlinear = mses5[:,0].argmin()
print("The optimal value for p in test data is : ", x_min_nonlinear, ", where the MSE is : ", y_min_nonlinear)

end_time = time.time()
time_nonlinear = end_time - start_time
print('Time Consuming :', time_nonlinear)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))

fig.savefig('MSE for nonlinear regression')
plt.show()


# By observing this two plots, I find that when I set lamda = 0 on training data, as p value increases, both of the MSE of ridge regression and gradient descent methods will decrease, and the No Regularization will decrease more than the Regularization. Moreover, we can find the error will decrease sharply when p value is in range (0,1). This is because when p = 0, it means using a horizontal line as the regression line, and p = 1 is the same as the linear ridge regression. So, during this range in (0,1), the error will decrease quickly due to the function of linear ridge regression. 
# 
# Another observation in test data, when I set lambda = optimal lambda(0.06), the smallest of error happens in p = 1 for gradient descent method and then increases sharply after p = 1. This is because when p value is more than 1, the attribute x will be converted into higher polinomial terms which will increase the complexity of weight function so it's easier to fit the original data more, it then generates overfitting and the error raises quickly.

# Problem 6

# In[22]:


###########
##Problem6
###########
col_0 = [mle_i[0][0], y_min_ridge[0], y_min_grad[0], y_min_nonlinear]
col_1 = [time_linear, time_ridge, time_grad, time_nonlinear]
summary = pd.DataFrame({'MSE' : col_0, 'Taking_Time' : col_1}, 
                       index = ['LinearRegression', 'RidgeRegression', 'Gradient Descent', 'NonLinearRegression'])

print('MSE with intercept in Linear Regression : ', mle_i[0][0])
print('MSE in Ridge Regression : ', y_min_ridge[0])
print('MSE in Gradient Descent for Ridge Regression : ', y_min_grad[0])
print('MSE in Non-Linear Regression : ', y_min_nonlinear)
print()
print('Time Consuming in Linear Regression :', time_linear)
print('Time Consuming in Ridge Regression :', time_ridge)
print('Time Consuming in Gradient Descent for Ridge Regression :', time_grad)
print('Time Consuming in Non-Linear Regression :', time_nonlinear)
print()
print(summary)


# After implementing this four methods, I would like to make final recommendations for anyone using regression for predicting diabetes level using the input feature in two sides. 
# 
# First side is in MSE, from the dataframe I plot above, we can easily find that the MSE value in Ridge Regression and Gradient Descent for Ridge Regression is much lower than it in Linear Regression and Non Linear Regression. This is because we put lambda into formula so as to control the weights. But, remembered, the ridge regression loss function contains a inverse term, when the data become large, this inverse term will become unstable due to the unstable covirance matrix. And as we learned, there might be a singular matrix (irreversible) term in the formula. In this situation, the ridge regression will become poor feasible, in turn the gradient descent method will become best choice, since it doesn't contain any inverse term in the formula.
# 
# Second side is in time consuming, I find the Linear Regression takes lowest time consuming while the gradient descent method takes highest. So, if we only want to consider time saving side, the Linear Regression will be the best choice.
