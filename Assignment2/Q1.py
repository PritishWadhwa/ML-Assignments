from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import pickle
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


class Layer():
    def __init__(self):
        self.weights = []
        self.bias = []
        self.dw = []
        self.db = []
        self.A = []
        self.Z = []
        self.shape = ()


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
        pass

        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation
        self.layers = self.initialize_network()
        start_l = Layer()
        self.layers.insert(0, start_l)  # Dummy layer
        self.Avgcost_epoch = []
        self.data_processing()

    def data_processing(self):
        # train_x, train_y = loadlocal_mnist("/content/drive/MyDrive/ass3_data/train-images.idx3-ubyte", "/content/drive/MyDrive/ass3_data/train-labels.idx1-ubyte")
        # test_x, test_y = loadlocal_mnist("/content/drive/MyDrive/ass3_data/t10k-images.idx3-ubyte", "/content/drive/MyDrive/ass3_data/t10k-labels.idx1-ubyte")
        train_x, train_y = loadlocal_mnist(
            "Weights/Ques2/extracted/train-images-idx3-ubyte", "Weights/Ques2/extracted/train-labels-idx1-ubyte")
        test_x, test_y = loadlocal_mnist(
            "Weights/Ques2/extracted/t10k-images-idx3-ubyte", "Weights/Ques2/extracted/t10k-labels-idx1-ubyte")
        # self.train_x = preprocessing.normalize(train_x)
        # self.test_x = preprocessing.normalize(test_x)
        self.train_x = train_x
        self.test_x = test_x
        enc = OneHotEncoder(sparse=False, categories='auto')
        self.train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
        self.test_y = enc.transform(test_y.reshape(len(test_y), -1))

    def initialize_network(self):
        network = []
        for i in range(1, self.n_layers):
            network.append(self.initialize_layer(
                self.layer_sizes[i], self.layer_sizes[i-1], self.weight_init))
        return network

    def initialize_layer(self, n_output, n_input, initialization):
        """
        single layer initializer
        n_input : input dimension
        initialization : possible inputs: zero, random, normal
        n_output : output dimension
        each layer is a dictionary with weight and bias
        """
        layer = Layer()
        shape = (n_output, n_input)
        print("Layershape : ", shape)
        if initialization == "zero":
            layer.weights = self.zero_init(shape)
        if initialization == "random":
            layer.weights = self.random_init(shape)
        if initialization == "normal":
            layer.weights = self.normal_init(shape)
        # np.random.seed(1)
        # layer.weights=np.random.randn(shape[0], shape[1]) / np.sqrt(shape[1])
        layer.bias = np.zeros([n_output, 1])
        return layer

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
        # Z=X.T
        r = np.maximum(0, X)
        return r

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
        # Z=X.T
        return (self.relu(X) > 0)*1

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
        # Z=X.T
        s = 1/(1+np.exp(-X))
        return s

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
        # Z=X.T
        return self.sigmoid(X) * (1 - self.sigmoid(X))

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
        return 1

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
        # Z=X.T
        t = np.tanh(X)
        return t

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
        # Z=X.T
        return 1-(np.power(self.tanh(X), 2))

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
        # print("INput Shape : ",X.shape)
        # Z=X.T
        expX = np.exp(X - np.max(X))
        s = expX / (expX.sum(axis=0, keepdims=True))
        # print("Output Shape : ",s.T.shape)
        return s

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
        s = X.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        np.random.seed(1)
        weights = np.zeros(shape[0], shape[1])
        return weights

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        np.random.seed(1)
        weights = np.random.randn(shape[0], shape[1])*0.01
        return weights

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        np.random.seed(1)
        weights = 0.01*np.random.normal(size=[shape[0], shape[1]])  # change

        return weights

    def transfer(self, X):
        """
        Transfer function application
        """
        activation = self.activation
        if activation == "sigmoid":
            return self.sigmoid(X)
        elif activation == "relu":
            return self.relu(X)
        elif activation == "linear":
            return self.linear(X)
        elif activation == "tanh":
            return self.tanh(X)
        elif activation == "softmax":
            return self.softmax(X)

    def lastLayer_Softmax(self, A):
        Z = np.dot(self.layers[-1].weights, A)+self.layers[-1].bias
        A = self.softmax(Z)
        self.layers[-1].A = A
        self.layers[-1].Z = Z
        return A

    def forward_prop(self, input_r):
        """
        input_r : input vector
        Forward propogation which:-
        --> calculates activation output
        -->applies transfer function to it
        -->makes output for current layer as input for next layer
        """
        A = input_r.T
        for i in range(1, len(self.layers)-1):
            Z = np.dot(self.layers[i].weights, A)+self.layers[i].bias
            A = self.transfer(Z)
            self.layers[i].A = A
            self.layers[i].Z = Z
        A = self.lastLayer_Softmax(A)
        return A

    def transfer_grad(self, Z, dAPrev):
        activation = self.activation
        if activation == "sigmoid":
            dZ = dAPrev * self.sigmoid_grad(Z)
        elif activation == "relu":
            dZ = dAPrev * self.relu_grad(Z)
        elif activation == "linear":
            dZ = dAPrev * self.linear_grad(Z)
        elif activation == "tanh":
            dZ = dAPrev * self.tanh_grad(Z)
        else:
            dZ = dAPrev * self.softmax_grad(Z)
        return dZ

    def lastLayer_backProp(self, A, Y, n):
        dZ = A - Y.T
        dW = np.dot(dZ, self.layers[-2].A.T) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n
        # for last layer
        dAPrev = self.layers[-1].weights.T.dot(dZ)
        self.layers[-1].dw = dW
        self.layers[-1].db = db
        return dZ, dW, db, dAPrev

    def backward_prop(self, X, Y):
        """
        backward pass 
          X : input data
          Y : true values
        """
        n = X.shape[0]
        self.layers[0].A = X.T
        A = self.layers[-1].A  # PREDICTED VALUES
        dZ, dW, db, dAPrev = self.lastLayer_backProp(A, Y, n)
        for layer_no in range(len(self.layers)-2, 0, -1):
            dZ = self.transfer_grad(self.layers[layer_no].Z, dAPrev)
            dW = 1. / n * np.dot(dZ, self.layers[layer_no-1].A.T)
            db = 1. / n * np.sum(dZ, axis=1, keepdims=True)
            if layer_no > 1:
                dAPrev = np.dot(self.layers[layer_no].weights.T, dZ)
            self.layers[layer_no].dw = dW
            self.layers[layer_no].db = db

    def update_weights(self):
        """
        Used for updating weights after a forward and a backward pass
        """
        for l in range(1, len(self.layers)):
            self.layers[l].weights = self.layers[l].weights - \
                ((self.learning_rate)*self.layers[l].dw)
            self.layers[l].bias = self.layers[l].bias - \
                ((self.learning_rate)*self.layers[l].db)

    def fit(self, X, Y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        Y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        Returns
        -------
        self : an instance of self
        """
        X = self.train_x
        Y = self.train_y
        self.test_epoch_error = []
        for epoch in range(self.num_epochs):
            X, Y = shuffle(X, Y, random_state=1)
            num_batches = int(X.shape[0]/self.batch_size)
            train_x = np.vsplit(X, num_batches)
            train_y = np.vsplit(Y, num_batches)
            cost, cost_batch = self.fit_batch(train_x, train_y, num_batches)
            avg_epoch_cost = cost/len(cost_batch)
            self.Avgcost_epoch.append(avg_epoch_cost)

            A_test = self.forward_prop(self.test_x)
            self.test_epoch_error.append(self.cost(self.test_y, A_test))
            if epoch % 10 == 0:
                print("Epoch no.", epoch, " Cost:", avg_epoch_cost,
                      "Accuracy:", self.score(X, Y))
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def fit_batch(self, train_x, train_y, num_batches):
        """
        train_x : splitted array of X train set
        train_y : splitted array of Y train set
        num_batches : number of Batches
        cost : returns the total cost for all batches
        cost_batch : returns arr of all costs of batches
        """
        cost = 0
        cost_batch = []
        for i in range(num_batches):
            A = self.forward_prop(train_x[i])
            c = self.cost(train_y[i], A)
            cost += c
            cost_batch.append(c)
            self.backward_prop(train_x[i], train_y[i])
            self.update_weights()
        return cost, cost_batch

    def cost(self, Y, A):
        return -np.mean(Y * np.log(A.T + 1e-8))

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
        A = self.forward_prop(X)
        # return the numpy array y which contains the predicted values
        return A

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
        A = self.predict_proba(X)
        y_pred = np.argmax(A, axis=0).T
        # return the numpy array y which contains the predicted values
        return y_pred

    def score(self, X, Y):
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
        y_pred = self.predict(X)
        Y = np.argmax(Y, axis=1)
        acc = np.mean(y_pred == Y)*100
        # return the numpy array y which contains the predicted values
        return acc

    def saveModel(self, model, filename):
        pickle.dump(model, open(filename, 'wb'))

    def loadModel(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model

    def svd(self, X):
        svd = TruncatedSVD(n_components=20, random_state=0)
        X_svd = svd.fit(X).transform(X)
        return X_svd

    def tsne(self):
        # TSNE
        A = self.forward_prop(self.test_x)
        Y = np.argmax(self.test_y, axis=1)
        plot_mat = self.layers[-2].A.T
        X_svd = self.svd(plot_mat)
        tsn = TSNE(random_state=0, n_components=2)
        X_sn = tsn.fit_transform(X_svd)
        sns.scatterplot(X_sn[:, 0], X_sn[:, 1], hue=Y,
                        legend='full', palette=sns.color_palette("hls", 10))
        plt.savefig("TSNE")
        plt.show()

    def SkLearn(self, activationn, batch_size=200):
        clf = MLPClassifier(activation=activationn, max_iter=100, learning_rate_init=0.1, random_state=1,
                            hidden_layer_sizes=(256, 128, 64), batch_size=batch_size).fit(self.train_x, self.train_y)
        print(activationn, "Score", clf.score(self.test_x, self.test_y))
