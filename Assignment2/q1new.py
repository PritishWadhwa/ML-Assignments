import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import os
import gzip


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')

        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        pass

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X * (X >= 0)

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1*(X >= 0)

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X) * (1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-self.tanh(X)*self.tanh(X)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        exp = np.exp(X)
        return exp/(np.sum(exp, axis=1, keepdims=True))

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer

        """

        weight = np.zeros(shape)
        return weight

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.rand(shape[0], shape[1])*0.01
        return weight

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        weight = np.random.normal(size=shape, scale=0.01)
        return weight

    def fit(self, X, y, x_test=None, y_test=None):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py

        y = self.expand_labels(y)

        m, n_0 = X.shape
        n_l = y.shape[1]

        parameters = self.initialization()
        self.parameters = parameters

        train_loss_history = []
        train_accuracy_history = []
        test_loss_history = []
        test_accuracy_history = []

        for epoch in tqdm(range(self.num_epochs), desc="Progress Total : ", position=0, leave=True):

            n_batches = m//self.batch_size
            X_batches = [X[self.batch_size*i:self.batch_size *
                           (i+1), :] for i in range(0, n_batches)]
            y_batches = [y[self.batch_size*i:self.batch_size *
                           (i+1), :] for i in range(0, n_batches)]

            train_batch_loss = []
            test_batch_loss = []
            train_batch_accuracy = []
            test_batch_accuracy = []

            for curr_x, curr_y in tqdm(zip(X_batches, y_batches), desc="Progress Epoch: " + str(epoch+1) + "/" + str(self.num_epochs), position=0, leave=True, total=len(X_batches)):
                A, activations, preactivations = self.forward_prop(
                    curr_x, parameters)

                train_cost = self.cross_entropy_loss(A, curr_y)
                train_batch_loss.append(train_cost)
#                 print(A)
                self.backward_prop(curr_x, curr_y, preactivations, activations)
#                 train_batch_accuracy.append(self.score(curr_x,np.argmax(curr_y,axis = 1)))
                if(x_test is not None):
                    proba = self.predict_proba(x_test)
#                     print(proba.shape)
                    test_loss = self.cross_entropy_loss(
                        proba, self.expand_labels(y_test))
                    test_batch_loss.append(test_loss)
#                     test_batch_accuracy.append(self.score(x_test, y_test))

#             print("Training Accuracy : ", np.array(train_batch_accuracy).mean())
#             print("Validation Accuracy : ", np.array(test_batch_accuracy).mean())
            print("Validation loss : ", np.array(test_batch_loss).mean())
            print("Training Loss : ", np.array(train_batch_loss).mean())
            print("Score : ", self.score(x_test, y_test))
            train_loss_history.append(np.array(train_batch_loss).mean())
#             train_accuracy_history.append( np.array(train_batch_accuracy).mean())
            test_loss_history.append(np.array(test_batch_loss).mean())
#             test_accuracy_history.append(  np.array(test_batch_accuracy).mean())

        self.train_loss_history = train_loss_history
        self.train_accuracy_history = train_accuracy_history
        self.test_loss_history = test_loss_history
        self.test_accuracy_history = test_accuracy_history

        self.parameters = parameters

        return self

    def initialization(self):
        parameters = {}
        layers = self.layer_sizes

        for i in range(0, len(layers)-1):
            if(self.weight_init == 'zero'):
                curr_layer = self.zero_init((layers[i], layers[i+1]))

            elif(self.weight_init == 'random'):
                curr_layer = self.random_init((layers[i], layers[i+1]))

            else:
                curr_layer = self.normal_init((layers[i], layers[i+1]))

            parameters["W" + str(i+1)] = curr_layer
            parameters["b" + str(i+1)] = np.zeros((1, layers[i+1]))

        self.parameters = parameters

        return parameters

    def forward_prop(self, X, parameters):
        """
        Implements one forward propagation of the deep neural network.

        Parameters 
        ----------
        X : Training set to be forward propagated
        parameters : model parameters 

        Returns
        -------
        A_l : Activations of the final layer
        forward_cache : list contraining all the linear_cache and activation_cache of all the layers

        """

        A = X
        L = len(parameters)//2

        activations = {}
        preactivations = {}

        for i in range(0, L-1):
            A_prev = A
#             print(A_prev.shape)

            Z = np.dot(A_prev, parameters["W" + str(i+1)]
                       ) + parameters["b" + str(i+1)]

            if(self.activation == "relu"):
                A = self.relu(Z)

            elif (self.activation == "tanh"):
                A = self.tanh(Z)

            elif (self.activation == "linear"):
                A = self.linear(Z)

            elif (self.activation == "sigmoid"):
                A = self.sigmoid(Z)

            preactivations["Z" + str(i+1)] = Z
            activations["A" + str(i+1)] = A
            A_prev = A

        Z_l = np.dot(A_prev, parameters["W" + str(L)]
                     ) + parameters["b" + str(L)]
        A_l = self.softmax(Z_l)
        preactivations["Z" + str(L)] = Z_l
        activations["A" + str(L)] = A_l

        return A_l, activations, preactivations

    def backward_prop(self, X, Y, preactivations, activations):
        """
        Implements backward propagation of the complete model.

        Parameters
        ----------
        y : The ground truth labels of the curr training set
        A_l : activations of the final layer of the model
        cache : tuple containing the linear_cache and activation_cache

        Returns
        -------
        gradients : Dictionary containing the gradient vectors for each layer of the model

        """

        derivatives = {}
        L = len(activations)
        activations["A0"] = X

        A = activations["A" + str(L)]
        dZ = A - Y

        dW = np.dot(activations["A" + str(L-1)].T, dZ)/len(X)
        db = np.sum(dZ, axis=0, keepdims=True) / len(X)

        dAPrev = np.dot(dZ, self.parameters["W" + str(L)].T)

        derivatives["dW" + str(L)] = dW
        derivatives["db" + str(L)] = db

        for l in range(L - 1, 0, -1):
            if(self.activation == "relu"):
                dact = self.relu_grad(preactivations["Z" + str(l)])

            elif (self.activation == "tanh"):
                dact = self.tanh_grad(preactivations["Z" + str(l)])

            elif (self.activation == "linear"):
                dact = self.linear_grad(preactivations["Z" + str(l)])

            elif (self.activation == "sigmoid"):
                dact = self.sigmoid_grad(preactivations["Z" + str(l)])

            dZ = dAPrev * dact
            dW = (1/len(X)) * np.dot(activations["A" + str(l - 1)].T, dZ)
            db = (1/len(X)) * np.sum(dZ, axis=0, keepdims=True)
            if l > 1:
                dAPrev = np.dot(dZ, self.parameters["W" + str(l)].T)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        for i in range(0, L):
            self.parameters["W" + str(i+1)] = self.parameters["W" + str(i+1)] - \
                self.learning_rate*derivatives["dW" + str(i+1)]
            self.parameters["b" + str(i+1)] = self.parameters["b" + str(i+1)] - \
                self.learning_rate*derivatives["db" + str(i+1)]
        return

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        proba, _, k = self.forward_prop(X, self.parameters)
        return proba

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        proba = self.predict_proba(X)
#         print(proba.shape)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        y_pred = self.predict(X)
        acc = (y_pred == y)
        return acc.sum()/len(y)

    def cross_entropy_loss(self, A, y):
        n = len(y)
        logp = - np.log(A[np.arange(n), y.argmax(axis=1)])
        loss = np.sum(logp)/n
        return loss

    def expand_labels(self, y):
        m = len(y)
        c = np.max(y)
        new_y = np.zeros((m, c+1))
        for i in range(m):
            l = y[i]
            new_y[i, l] = 1

        return new_y


# train_df = pd.read_csv('dataset/mnist/mnist_train.csv')
# test_df = pd.read_csv('dataset/mnist/mnist_test.csv')
allX, ally = load_mnist('Weights/Ques2/', kind='train')
allX_2, ally_2 = load_mnist('Weights/Ques2/', kind='t10k')
X = allX
X = np.concatenate((X, allX_2), axis=0)
y = ally
y = np.concatenate((y, ally_2), axis=0)


def train_test_split(df, trainSize=0.8, testSize=0.2, random_state=1):
    validSize = 1 - trainSize - testSize
    indices = np.arange(df.shape[0])
    np.random.seed(random_state)
    np.random.shuffle(indices)
    trainData = df.iloc[indices[:int(
        trainSize*df.shape[0])]].reset_index(drop=True)
    validData = df.iloc[indices[int(
        trainSize*df.shape[0]):int((trainSize+validSize)*df.shape[0])]].reset_index(drop=True)
    testData = df.iloc[indices[int(
        (trainSize+validSize)*df.shape[0]):]].reset_index(drop=True)
    if validSize == 0:
        return trainData, testData
    else:
        return trainData, validData, testData


X = pd.DataFrame(X)
y = pd.DataFrame(y)
allData = pd.concat([X, y], axis=1)
print(allData.shape)
# allData.head()
trainData, validData, testData = train_test_split(
    allData, trainSize=0.7, testSize=0.2, random_state=42)
trainX = trainData.iloc[:, :-1]
trainY = trainData.iloc[:, -1]
validX = validData.iloc[:, :-1]
validY = validData.iloc[:, -1]
testX = testData.iloc[:, :-1]
testY = testData.iloc[:, -1]
trainX = np.array(trainX)
trainY = np.array(trainY)
validX = np.array(validX)
validY = np.array(validY)
testX = np.array(testX)
testY = np.array(testY)

# dataset = train_df.to_numpy()
# testset = test_df.to_numpy()

# X_train = dataset[:, 1:]/255
# X_test = testset[:, 1:]/255

X_train = trainX/255
X_test = testX/255
standardscalar = StandardScaler()
X_train = standardscalar.fit_transform(X_train)
X_test = standardscalar.transform(X_test)

# y_train = dataset[:, 0]
# y_test = testset[:, 0]

y_train = trainY
y_test = testY

nn = MyNeuralNetwork(6, [784, 256, 128, 64, 32, 10], 'relu',
                     0.08, 'random', len(X_train)//20, 150)


nn.fit(X_train, y_train, X_test, y_test)

nn.score(X_test, y_test)


plt.plot([x for x in range(1, len(nn.train_loss_history) + 1, 1)],
         nn.train_loss_history, label="Average Training  Loss ")
plt.plot([x for x in range(1, len(nn.test_loss_history) + 1, 1)],
         nn.test_loss_history, label="Average Validation  Loss ")
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()
