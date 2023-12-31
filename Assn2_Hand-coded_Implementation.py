"""
Name: Anshul Pattoo
Student Number: 20124223
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import random
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class BPN:
  
  #Computes the neural network's outputs for a given input pattern.
  def computeOutputs(self, inputPattern):
    hiddenLayerActivations = []
    hiddenLayerOuts = []
    outputLayerActivations = []
    networkOutputs = []
    #'h' indicates the set of weights for the corresponding hidden index.
    hiddenLayerOuts = self.outputFunc(np.dot(self.weightsIH, inputPattern))
 
    #Complete activation and output computations for each output node.
    networkOutputs = self.outputFunc(np.dot(self.weightsHO, hiddenLayerOuts))
    return (hiddenLayerOuts, networkOutputs)

  #Computes the dot product between aggregate error term (delta) and X.
  def aggErrTimesInputMatr(self, aggErrTerm, inputMatr):
    deltaX = np.dot(np.array(inputMatr).reshape((-1,1)), aggErrTerm.reshape((1,-1)))
    return deltaX
 
 #Computes derivative for given output matrix.
  def outputFuncDerivative(self, outputMatr):
    outputMatr = np.array(outputMatr)
    return outputMatr * (1 - outputMatr)

  #Computes individual output for given activation value.
  def outputFunc(self, activationVal):
    return 1/(1 + np.exp(-activationVal))

  #Returns a set of random floating point values in arrays. This is used for initializing the weights.
  def initializeArrVals(self, outsideArrayLen, insideArrayLen):
    outerList = []
    innerList = []
    for i in range(outsideArrayLen):
      for j in range(insideArrayLen):
        innerList.append(np.random.uniform(-0.5, 0.5))
      outerList.append(innerList)
      innerList = []
    return outerList
  
  #Calculates the aggregate error term between the hidden layer and output later.
  def calcAggregateErrorHO(self, actualOutputMatr, predOutputMatr):
    dMinusY = actualOutputMatr - predOutputMatr
    yDotOneMinusY = self.outputFuncDerivative(predOutputMatr)
    return dMinusY * yDotOneMinusY
 
  #Calculates the aggregate error term between the input layer and hidden layer.
  def calcAggregateErrorIH(self, actualOutputMatr, predOutputMatr, hiddenOutputs):
    sumTermMatrix = []
    #Length of the sum matrix is equivalent to that of a single set of weights for an output node: 100.
    sum = np.zeros(100)
    for j in range(self.numOutNodes):
      sumTermMatrix = (actualOutputMatr[j] - predOutputMatr[j]) * predOutputMatr[j] * (1 - predOutputMatr[j]) * self.weightsHO[j]
      sum = np.add(sum, np.array(sumTermMatrix))
      sumTermMatrix = []
    hiddenOutputs = np.array(hiddenOutputs)
    
    return np.dot(sum, np.dot(hiddenOutputs, 1 - hiddenOutputs))

  #Determines the activation value for a node j in the output layer.
  def activationHO(self, hiddenOutputs, outIndex):
    #Returns dot product between a 100 hidden node matrix and the corresponding set of weights for an output node.
    return hiddenOutputs * self.weightsHO[outIndex]

  #Determines the activation value for a node h in the hidden layer.
  def activationIH(self, inputPattern, hiddenIndex):
    #Returns dot product between a 785 input node matrix and the corresponding set of weights for a hidden node.
    return np.dot(inputPattern, self.weightsIH[hiddenIndex])

  #Calculates and displays the performance metrics.
  def performanceMetrics(self, desiredOutputs, predOutputs):
    actLabels = []
    predLabels = []
    for i in range(len(desiredOutputs)):
      actLabels.append(np.argmax(desiredOutputs[i]))
      predLabels.append(np.argmax(predOutputs[i]))

    #Accuracy, precision, and recall.
    accuracy = sklearn.metrics.accuracy_score(actLabels, predLabels,  normalize = True)
    precision = sklearn.metrics.precision_score(actLabels, predLabels, average = 'macro')
    recall = sklearn.metrics.recall_score(actLabels, predLabels, average = 'micro')

    #Confusion matrix.
    data = {'yActual': actLabels, 'yPredicted': predLabels}
    df = pd.DataFrame(data, columns=['yActual','yPredicted'])
    confusionMatrix = pd.crosstab(df['yActual'], df['yPredicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusionMatrix)

    print("The prediction accuracy is " + "{:.2f}".format(accuracy * 100) + "% .")
    print("The overall precision is " + "{:.2f}".format(precision) + ".") 
    print("The recall is " + "{:.2f}".format(recall) + ".")

  #Tests data.
  def test(self, inputPatterns, desiredOutputs):
    (inputs, desiredOutputs) = self.preprocessData(inputPatterns, desiredOutputs)
    predOutputs = []
    for i in range(len(inputPatterns)):
      networkOutputs = self.computeOutputs(inputs[i])[1]
      #Transform predicted output to one-hot encoding. We assume that there is only one instance of this maximum value.
      predOutputs.append(networkOutputs)
   
    #Performance metrics calculated and printed.
    self.performanceMetrics(desiredOutputs, predOutputs)

  #Trains data.
  def train(self, inputPatterns, outputs):
    #Conduct data pre-processing. Inputs here are 784 x 1 matrices; outputs are one-hot encodings.
    (inputPatterns, outputs) = self.preprocessData(inputPatterns, outputs)
    deltaWtMinusOneIH = deltaWtMinusOneHO = 0
    networkOutputs = []
    predOutputs = []
    
    for epoch in range(self.epochs):
      #For each of the input patterns, compute the following. 60,000 is the number of inputs — this value is hard-coded.
      for i in range(60000):
        #Compute hidden layer outputs and network outputs.
        (hiddenLayerOuts, networkOutputs) = self.computeOutputs(inputPatterns[i])
      
        #Modify weights between hidden and output nodes.       
        aggErrorTermHO = self.calcAggregateErrorHO(outputs[i], networkOutputs)
        #Note that "delta" represents the aggregate error term, and "Xh" represents the set of hidden layer outputs.
        deltaXh = self.aggErrTimesInputMatr(aggErrorTermHO, hiddenLayerOuts)
        cDeltaXh = np.dot(self.learningRate, deltaXh)
        deltaWtHO = cDeltaXh + (self.momentum * deltaWtMinusOneHO)
        
        #Weights used in next iteration: W (t+1).
        self.weightsHO = self.weightsHO + deltaWtHO.T
        
        #deltaWtMinusOne for following iteration.
        deltaWtMinusOneHO = deltaWtHO
        #Modify weights between input and hidden nodes. Same process as above followed.
        aggErrorTermIH = self.calcAggregateErrorIH(outputs[i], networkOutputs, hiddenLayerOuts)
        cDeltaXi = self.learningRate * np.array(self.aggErrTimesInputMatr(aggErrorTermIH, inputPatterns[i]))
        deltaWtIH = cDeltaXi + (self.momentum * deltaWtMinusOneIH)
        
        self.weightsIH = self.weightsIH + deltaWtIH.T
 
  #Preprocesses the data. Returns a tuple containing modified input and output data.
  def preprocessData(self, inputs, outputs_raw):
    #Normalize the data to the range [0, 1].
    inputs = inputs.astype("float32")
    inputs /= 255
    #Flattens each 28x28 input matrix to an array of 785x1 (this includes an input node containing 1).
    flattenedInputs = []
    for i in range(len(inputs)):
      flattenedInputs.append(inputs[i].flatten())
      flattenedInputs[i] = np.append(flattenedInputs[i], 1)
    
    inputs = np.array(flattenedInputs)
    #Produce one-hot encodings for the output matrix.
    outputs = np.zeros((outputs_raw.shape[0],10))
   
    for i in range(outputs_raw.shape[0]):
      outputs[i][outputs_raw[i]] = 1
    
    return (inputs, outputs)
    
  def __init__(self, numInNodes, numHiddenNodes, numOutNodes, momentum = 0.1, learningRate = 0.01, epochs = 1):
    #Choose a set of random weights between 0 and 1.
    #Input-hidden weights: Each hidden node has a count of weights equal to the number of input nodes.
    self.weightsIH = np.array(self.initializeArrVals(numHiddenNodes, numInNodes))
    #Hidden-output weights: Each output node has a count of weights equal to the number of hidden nodes.
    self.weightsHO =np.array(self.initializeArrVals(numOutNodes, numHiddenNodes))
    self.numHiddenNodes = numHiddenNodes
    self.numOutNodes = numOutNodes
    self.learningRate = learningRate
    self.momentum = momentum
    self.epochs = epochs
 
#Data is loaded here.
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data("/Users/anshulpattoo/Desktop/mnist.npz")
BPNObj = BPN(785, 100, 10)
BPNObj.train(trainX, trainY)

#Performance metrics get printed to console for a given test run. 
#They are printed from the performanceMetrics() method.
BPNObj.test(trainX, trainY)
