import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Create Training Data
def load_extra_datasets():  
    N = 200
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles (mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2,  shuffle=True, random_state=None)
    return  gaussian_quantiles

def sigmoid(x):
    return 1/(1+np.exp(-x))

def prediction(probability):
    if(probability<=0.5):
        return 0;
    else:
        return 1;

gaussian_quantiles= load_extra_datasets()
x, y = gaussian_quantiles

x, y = x.T, y.reshape(1, y.shape[0])

plt.figure(1)
plt.scatter(x[0, :], x[1, :], c=y[0,:], s=40, cmap=plt.cm.Spectral)

#print x.shape
#print y.shape
#print x[:,0:5].T
#print y



# Initialize
inputLayerSize = x.shape[0]
hiddenLayerSize = 4
outputLayerSize = y.shape[0]

weightVector1 = np.random.randn(hiddenLayerSize,inputLayerSize)
weightVector2 = np.random.randn(outputLayerSize,hiddenLayerSize)

biasVector1 = np.zeros(shape = (hiddenLayerSize,1))
biasVector2 = np.zeros(shape = (1,outputLayerSize))
#print np.dot(weightVector1,x).shape

for i in range(0,10000):

    #Forward Computation
    sigma1 = np.dot(weightVector1,x)+biasVector1
    activation1 = np.tanh(sigma1)

    sigma2 = np.dot(weightVector2,activation1)+biasVector2
    activation2 = sigmoid(sigma2)

    #Loss
    m = x.shape[1] 
    lossVector = np.multiply(y,np.log(activation2)) + np.multiply((1-y),np.log(1-activation2))
    costTotal = -np.sum(lossVector)/m
    #print activation1.shape
    if (i%1000 == 0):
        print ("Cost_",i,": ",costTotal)
    #Derivative Computations
    dsigma2 = activation2 - y
    dweightVector2 = (1 /m) * np.dot(dsigma2, activation1.T)
    dbiasVector2 = (1 / m) * np.sum(dsigma2, axis=1, keepdims=True)
    dsigma1 = np.multiply(np.dot(weightVector2.T, dsigma2), 1 - np.power(activation1, 2))
    dweightVector1 = (1 / m) * np.dot(dsigma1, x.T)
    dbiasVector1 = (1 / m) * np.sum(dsigma1, axis=1, keepdims=True)

    #Update Weights
    learningRate = 0.1
    weightVector1 = weightVector1 - learningRate*dweightVector1
    biasVector1 = biasVector1 - learningRate*dbiasVector1
    weightVector2 = weightVector2 - learningRate*dweightVector2
    biasVector2 = biasVector2 - learningRate*dbiasVector2

print ("Training Complete\n")

#Do prediction on trained weights
sigma1 = np.dot(weightVector1,x)+biasVector1
activation1 = np.tanh(sigma1)

sigma2 = np.dot(weightVector2,activation1)+biasVector2
activation2 = sigmoid(sigma2)

predictionVector = activation2>0.5
plt.figure(2)
plt.scatter(x[0, :], x[1, :], c=predictionVector[0,:], s=40, cmap=plt.cm.Spectral)
plt.show()

