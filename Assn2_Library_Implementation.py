"""
Name: Anshul Pattoo
Student Number: 20124223
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import time
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class BPN: 

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
    predOut = self.classifier.predict(inputs)
    self.performanceMetrics(desiredOutputs, predOut)
  
  #Trains data.
  def train(self, inputPatterns, outputs):
    #Conduct data pre-processing
    (inputs, outputs) = self.preprocessData(inputPatterns, outputs)

    #Initializing the MLPClassifier. 
    self.classifier = MLPClassifier(hidden_layer_sizes=(self.hidNodes, ), max_iter = self.epochs, momentum = self.momentum,
    learning_rate_init = self.learningRate, activation = 'relu', solver = 'adam', random_state = 1)

    #Fitting the training data to the network
    self.classifier.fit(inputs, outputs)
    
  #Preprocesses data.
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
  
  def __init__(self, numInNodes, numOutNodes, numHiddenNodes = 100, momentum = 0.1, learningRate = 0.01, epochs = 1):
    self.learningRate = learningRate
    self.momentum = momentum
    self.hidNodes = numHiddenNodes
    self.epochs = 1
  

#Data is loaded here.
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data("/Users/anshulpattoo/Desktop/mnist.npz")
BPNObj = BPN(785, 10)
BPNObj.train(trainX, trainY)

#Performance metrics get printed to console for a given test run. 
#They are printed from the performanceMetrics() method.
BPNObj.test(testX, testY)


