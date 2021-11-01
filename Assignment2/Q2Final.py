import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class MyNeuralNetwork:
    # Custom implementation of Neural Network Class
    def __init__(self, N_inputs, N_outputs, N_layers=2, Layer_sizes=[10, 5], activation="sigmoid", learning_rate=0.1, weight_init="random", batch_size=1, num_epochs=200):
        """
        N_inputs: input size
        N_outputs: outputs size
        N_layers: number of hidden layers
        Layer_sizes: list of hidden layer sizes
        activation: activation function to be used (ReLu, Leaky ReLu, sigmoid, linear, tanh, softmax)
        learning_rate: learning rate
        weight_init: weight initialization (zero, random, normal)
        batch_size: batch size
        num_epochs: number of epochs
        """
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.N_layers = N_layers
        self.Layer_sizes = Layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        np.random.seed(0)

        model = {}
        if weight_init == "zero":
            model['W1'] = np.zeros((N_inputs, Layer_sizes[0]))
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' +
                      str(i+1)] = np.zeros((Layer_sizes[i-1], Layer_sizes[i]))
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)
                  ] = np.zeros((Layer_sizes[-1], N_outputs))
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        elif weight_init == "random":
            model['W1'] = np.random.randn(N_inputs, Layer_sizes[0])*0.01
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' + str(i+1)] = np.random.randn(Layer_sizes[i-1],
                                                        Layer_sizes[i])*0.01
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)
                  ] = np.random.randn(Layer_sizes[-1], N_outputs)*0.01
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        elif weight_init == "normal":
            model['W1'] = np.random.normal(
                0, 1, (N_inputs, Layer_sizes[0]))*0.01
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' + str(i+1)] = np.random.normal(0, 1,
                                                         (Layer_sizes[i-1], Layer_sizes[i]))*0.01
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)] = np.random.normal(0,
                                                            1, (Layer_sizes[-1], N_outputs))*0.01
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        else:
            print("Invalid weight initialization")
            return

        self.model = model
        self.activationOutputs = None

    def relu_forward(self, X):
        """
        ReLu activation function for forward propagation
        X: input
        return: output after applying the relu function
        """
        return np.maximum(X, 0)

    def relu_backward(self, X):
        """
        ReLu activation function for backpropagation
        X: input
        return: output after applying the gradient of relu function
        """
        return np.where(X > 0, 1, 0)

    def leaky_relu_forward(self, X):
        """
        Leaky ReLu activation function
        X: input
        return: output after applying the Leaky ReLu function
        """
        return np.maximum(X, 0.01*X)

    def leaky_relu_backward(self, X):
        """
        Leaky ReLu activation function
        X: input
        return: output after applying the gradient of Leaky ReLu function
        """
        return np.where(X > 0, 1, 0.01)

    def sigmoid_forward(self, X):
        """
        Sigmoid activation function
        X: input
        return: output after applying the sigmoid function
        """
        return 1/(1+np.exp(-X))

    def sigmoid_backward(self, X):
        """
        Sigmoid activation function
        X: input
        return: output after applying the gradient of sigmoid function
        """
        return self.sigmoid_forward(X)*(1-self.sigmoid_forward(X))
        # return X*(1-X)

    def linear_forward(self, X):
        """
        Linear activation function
        X: input
        return: output after applying the linear function
        """
        return X

    def linear_backward(self, X):
        """
        Linear activation function
        X: input
        return: output after applying the gradient of linear function
        """
        return np.ones(X.shape)

    def tanh_forward(self, X):
        """
        Tanh activation function
        X: input
        return: output after applying the tanh function
        """
        return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
        # return np.tanh(X)

    def tanh_backward(self, X):
        """
        Tanh activation function
        X: input
        return: output after applying the gradient of tanh function
        """
        return 1-(self.tanh_forward(X)**2)
        # return 1-X**2

    def softmax_forward(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the softmax function
        """
        exp = np.exp(X - np.max(X))
        return exp/np.sum(exp, axis=1, keepdims=True)

    def softmax_backward_actual(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the gradient of softmax function
        """
        s = self.softmax_forward(X).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax_backward(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the gradient of softmax function
        """
        return self.softmax_forward(X)*(1-self.softmax_forward(X))

    def forward(self, X):
        """
        Forward propagation
        X: input
        return: output after applying the activation function
        """
        if self.activation == "relu":
            currentActivationFuntion = self.relu_forward
        elif self.activation == "leaky_relu":
            currentActivationFuntion = self.leaky_relu_forward
        elif self.activation == "sigmoid":
            currentActivationFuntion = self.sigmoid_forward
        elif self.activation == "linear":
            currentActivationFuntion = self.linear_forward
        elif self.activation == "tanh":
            currentActivationFuntion = self.tanh_forward
        elif self.activation == "softmax":
            currentActivationFuntion = self.softmax_forward
        else:
            raise ValueError("Invalid activation function")

        self.activationOutputs = {}

        self.activationOutputs['Z1'] = np.dot(
            X, self.model['W1']) + self.model['b1']
        self.activationOutputs['A1'] = currentActivationFuntion(
            self.activationOutputs['Z1'])
        # self.activationOutputs['A1'] = np.tanh(self.activationOutputs['Z1'])

        for i in range(2, self.N_layers+1):
            self.activationOutputs['Z' + str(i)] = np.dot(self.activationOutputs['A' + str(
                i-1)], self.model['W' + str(i)]) + self.model['b' + str(i)]
            self.activationOutputs['A' + str(i)] = currentActivationFuntion(
                self.activationOutputs['Z' + str(i)])

        self.activationOutputs['Z' + str(self.N_layers+1)] = np.dot(self.activationOutputs['A' + str(
            self.N_layers)], self.model['W' + str(self.N_layers+1)]) + self.model['b' + str(self.N_layers+1)]
        self.activationOutputs['A' + str(self.N_layers+1)] = currentActivationFuntion(
            self.activationOutputs['Z' + str(self.N_layers+1)])

        return self.activationOutputs['A' + str(self.N_layers+1)]

    def backward(self, X, Y):
        """
        Backward propagation
        X: input
        Y: output
        """
        if self.activation == "relu":
            currentActivationFuntion = self.relu_backward
        elif self.activation == "leaky_relu":
            currentActivationFuntion = self.leaky_relu_backward
        elif self.activation == "sigmoid":
            currentActivationFuntion = self.sigmoid_backward
        elif self.activation == "linear":
            currentActivationFuntion = self.linear_backward
        elif self.activation == "tanh":
            currentActivationFuntion = self.tanh_backward
        elif self.activation == "softmax":
            currentActivationFuntion = self.softmax_backward
        else:
            raise ValueError("Invalid activation function")

        # computing the gradients
        self.gradients = {}
        self.gradients['delta' + str(self.N_layers+1)] = (
            self.activationOutputs['A' + str(self.N_layers+1)] - Y)
        self.gradients['dW' + str(self.N_layers+1)] = (1/len(X)) * np.dot(self.activationOutputs['A' + str(
            self.N_layers)].T, self.gradients['delta' + str(self.N_layers+1)])
        self.gradients['db' + str(self.N_layers+1)] = (1/len(X)) * np.sum(
            self.gradients['delta' + str(self.N_layers+1)], axis=0, keepdims=True)

        for i in range(self.N_layers, 1, -1):
            self.gradients['delta' + str(i)] = np.dot(self.gradients['delta' + str(i+1)], self.model['W' + str(
                i+1)].T) * currentActivationFuntion(self.activationOutputs['Z' + str(i)])
            self.gradients['dW' + str(i)] = (1/len(X)) * np.dot(
                self.activationOutputs['A' + str(i-1)].T, self.gradients['delta' + str(i)])
            self.gradients['db' + str(i)] = (1/len(X)) * np.sum(
                self.gradients['delta' + str(i)], axis=0, keepdims=True)

        self.gradients['delta1'] = np.dot(
            self.gradients['delta2'], self.model['W2'].T) * currentActivationFuntion(self.activationOutputs['Z1'])
        self.gradients['dW1'] = (1/len(X)) * \
            np.dot(X.T, self.gradients['delta1'])
        self.gradients['db1'] = (
            1/len(X)) * np.sum(self.gradients['delta1'], axis=0, keepdims=True)

        # updating the model parameters
        for i in range(1, self.N_layers+2):
            self.model['W' + str(i)] -= self.learning_rate * \
                self.gradients['dW' + str(i)]
            self.model['b' + str(i)] -= self.learning_rate * \
                self.gradients['db' + str(i)]

    def oneHotEncoder(self, y, n_classes):
        """
        One hot encoder
        y: input
        return: encoded output
        """
        m = y.shape[0]
        y_oht = np.zeros((m, n_classes))
        y_oht[np.arange(m), y] = 1
        return y_oht

    def crossEntropyLoss(self, y_oht, y_prob):
        """
        Cross entropy loss
        y_oht: one hot encoded output
        y_prob: probabilities for classes
        return: cross entropy loss
        """
        return -np.mean(y_oht * np.log(y_prob + 1e-8))

    def fit(self, X, y, validX=None, validY=None, logs=True):
        """
        Fit the model to the data
        X: input
        Y: output
        epochs: number of epochs
        """
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        classes = self.N_outputs
        batchSize = self.batch_size
        y_oht = self.oneHotEncoder(y, classes)
        if validX is not None and validY is not None:
            y_oht_valid = self.oneHotEncoder(validY, classes)
        for i in range(self.num_epochs):
            for j in range(0, X.shape[0], batchSize):
                X_batch = X[j:j+batchSize]
                y_batch = y_oht[j:j+batchSize]
                y_ = self.forward(X_batch)
                self.backward(X_batch, y_batch)
            y_ = self.forward(X)
            train_loss = self.crossEntropyLoss(y_oht, y_)
            train_losses.append(train_loss)
            if validX is not None and validY is not None:
                y_valid = self.forward(validX)
                valid_loss = self.crossEntropyLoss(y_oht_valid, y_valid)
                valid_losses.append(valid_loss)
                validAcc = self.score(validX, validY)
                valid_accs.append(validAcc)
            trainAcc = self.score(X, y)
            train_accs.append(trainAcc)
            if logs:
                print("Epoch: {}, Loss: {}, Score: {}".format(
                    i, train_loss, trainAcc))
        if validX is not None and validY is not None:
            return train_losses, valid_losses, train_accs, valid_accs
        return train_losses, train_accs

    def predict_proba(self, X):
        """
        Predict probabilities
        X: input
        return: probabilities
        """
        return self.forward(X)

    def predict(self, X):
        """
        Predict classes
        X: input
        return: classes
        """
        return np.argmax(self.forward(X), axis=1)

    def score(self, X, y):
        """
        Score the model
        X: input
        Y: output
        return: accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)*100

    def saveWeights(self, filename):
        """
        Save the weights
        filename: name of the file
        """
        np.save(filename, self.model, allow_pickle=True)

    def loadWeights(self, filename):
        """
        Load the weights
        filename: name of the file
        """
        self.model = np.load(filename, allow_pickle=True).item()


def train_test_split(df, trainSize=0.8, testSize=0.2, random_state=42):
    """
    This function splits the dataset into train and test
    df: The dataset
    trainSize: The size of the train set
    testSize: The size of the test set
    random_state: The random state
    return: The train, valid(if applicable) and test sets
    """
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


def lossVsEpochPlot(trainLoss, validLoss, funtion):
    """
    Plot the loss vs epoch plot
    trainLoss: train loss
    validLoss: valid loss
    filename: name of the file
    """
    plt.plot(trainLoss, label="Train Loss", color="blue")
    plt.plot(validLoss, label="Validation Loss", color="red")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs for " + funtion)
    plt.savefig("Plots/Ques2/part1/{}_lossVsEpochs.png".format(funtion))
    plt.show()


def accVsEpochPlot(trainAcc, validAcc, funtion):
    """
    Plot the accuracy vs epoch plot
    trainLoss: train loss
    validLoss: valid loss
    filename: name of the file
    """
    plt.plot(trainAcc, label="Train Accuracy", color="blue")
    plt.plot(validAcc, label="Validation Accuracy", color="red")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs for " + funtion)
    plt.savefig("Plots/Ques2/part1/{}_accVsEpochs.png".format(funtion))
    plt.show()


def drawImg(X, Y):
    """
    Draw the images
    X: input(numpy array having 784 elements)
    Y: value desired 
    """
    plt.imshow(X.reshape(28, 28), cmap='gray')
    plt.title("Label " + str(Y))
    plt.axis('off')
    plt.show()


# Importing and reading the dataset
with open('Weights/Ques2/extracted/t10k-images-idx3-ubyte', 'rb') as f:
    data = f.read()
    test_images = np.frombuffer(data, dtype=np.uint8)[0x10:].reshape(-1, 784)

with open('Weights/Ques2/extracted/t10k-labels-idx1-ubyte', 'rb') as f:
    data = f.read()
    test_labels = np.frombuffer(data, dtype=np.uint8)[8:]

with open('Weights/Ques2/extracted/train-images-idx3-ubyte', 'rb') as f:
    data = f.read()
    train_images = np.frombuffer(data, dtype=np.uint8)[0x10:].reshape(-1, 784)

with open('Weights/Ques2/extracted/train-labels-idx1-ubyte', 'rb') as f:
    data = f.read()
    train_labels = np.frombuffer(data, dtype=np.uint8)[8:]

# Merging the training and testing data
X = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

# Converting the data in pandas dataframe
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Merging the images and the labels to make the data ready for train test split
allData = pd.concat([X, y], axis=1)

trainData, validData, testData = train_test_split(
    allData, trainSize=0.7, testSize=0.2, random_state=42)
print("Train data size: ", trainData.shape)
print("Valid data size: ", validData.shape)
print("Test data size: ", testData.shape)


# Splitting the data into X and Y parts for all the three sets
trainX = trainData.iloc[:, :-1]
trainY = trainData.iloc[:, -1]
validX = validData.iloc[:, :-1]
validY = validData.iloc[:, -1]
testX = testData.iloc[:, :-1]
testY = testData.iloc[:, -1]


# Normalizing the data
trainX = trainX/255
validX = validX/255
testX = testX/255

# normalize = StandardScaler()
# trainX = normalize.fit_transform(trainX)
# validX = snormalizes.transform(validX)
# testX = snormalizes.transform(testX)

trainMean = trainX.mean()
trainStd = trainX.std()
trainX = (trainX - trainMean)/trainStd
validX = (validX - trainMean)/trainStd
testX = (testX - trainMean)/trainStd


# Task a and b

# Initialising the model For ReLU
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="relu", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/relu.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "relu")
accVsEpochPlot(trainAccs, validAccs, "relu")

# printing the testing accuracy
print("Test Accuracy ReLU: {}".format(model.score(testX, testY)))


# Initialising the model For Leaky ReLU
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="leaky_relu", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/leaky_relu.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "leaky_relu")
accVsEpochPlot(trainAccs, validAccs, "leaky_relu")

# printing the testing accuracy
print("Test Accuracy Leaky ReLU: {}".format(model.score(testX, testY)))


# Initialising the model For Sigmoid
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="sigmoid", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/sigmoid.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "sigmoid")
accVsEpochPlot(trainAccs, validAccs, "sigmoid")

# printing the testing accuracy
print("Test Accuracy Sigmoid: {}".format(model.score(testX, testY)))


# Initialising the model For Linear
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="linear", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/linear.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "linear")
accVsEpochPlot(trainAccs, validAccs, "linear")

# printing the testing accuracy
print("Test Accuracy Linear: {}".format(model.score(testX, testY)))


# Initialising the model For tanh
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="tanh", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/tanh.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "tanh")
accVsEpochPlot(trainAccs, validAccs, "tanh")

# printing the testing accuracy
print("Test Accuracy tanh: {}".format(model.score(testX, testY)))


# Initialising the model For softmax
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="softmax", learning_rate=0.08, weight_init="normal", num_epochs=150, batch_size=len(X)//10)
# fitting the model
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)

# saving the model weights
model.saveWeights("Weights/Ques2/part1/softmax.npy")

# plotting the loss vs epochs and accuracy vs epochs
lossVsEpochPlot(trainLoss, validLoss, "softmax")
accVsEpochPlot(trainAccs, validAccs, "softmax")

# printing the testing accuracy
print("Test Accuracy Softmaxs: {}".format(model.score(testX, testY)))


# Task d
def sklearnModel(trainX, trainY, validX, validY, testX, testY, activationFn):
    """
    This function is used to train the model using sklearn
    trainX: training data
    trainY: training labels
    validX: validation data
    validY: validation labels
    testX: test data
    testY: test labels
    activationFn: activation function to be used
    """
    # initialising the model
    model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation=activationFn,
                          learning_rate_init=0.08, max_iter=150, batch_size=len(X)//10, verbose=True)
    # fitting the model
    model.fit(trainX, trainY)
    # printing the accuracies
    print("Test Accuracy {}: {}".format(
        activationFn, model.score(testX, testY)))
    print("Validation Accuracy {}: {}".format(
        activationFn, model.score(validX, validY)))
    print("Train Accuracy {}: {}".format(
        activationFn, model.score(trainX, trainY)))


# sklearn for ReLu
sklearnModel(trainX, trainY, validX, validY, testX, testY, "relu")

# sklearn for Sigmoid
sklearnModel(trainX, trainY, validX, validY, testX, testY, "logistic")

# sklearn for Linear
sklearnModel(trainX, trainY, validX, validY, testX, testY, "identity")

# sklearn for Tanh
sklearnModel(trainX, trainY, validX, validY, testX, testY, "tanh")


# Task e

# learning_rate = 0.001, epochs = 100
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="tanh", learning_rate=0.001, weight_init="normal", num_epochs=100, batch_size=len(X)//10)
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)
print("Train Accuracy Tanh: {}".format(model.score(trainX, trainY)))
print("Test Accuracy Tanh: {}".format(model.score(testX, testY)))
print("Validation Accuracy Tanh: {}".format(model.score(validX, validY)))

# learning_rate = 0.01, epochs = 100
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="tanh", learning_rate=0.01, weight_init="normal", num_epochs=100, batch_size=len(X)//10)
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)
print("Train Accuracy Tanh: {}".format(model.score(trainX, trainY)))
print("Test Accuracy Tanh: {}".format(model.score(testX, testY)))
print("Validation Accuracy Tanh: {}".format(model.score(validX, validY)))

# learning_rate = 0.1, epochs = 100
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="tanh", learning_rate=0.1, weight_init="normal", num_epochs=100, batch_size=len(X)//10)
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)
print("Train Accuracy Tanh: {}".format(model.score(trainX, trainY)))
print("Test Accuracy Tanh: {}".format(model.score(testX, testY)))
print("Validation Accuracy Tanh: {}".format(model.score(validX, validY)))

# learning_rate = 1, epochs = 100
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=4, Layer_sizes=[
                        256, 128, 64, 32], activation="tanh", learning_rate=1, weight_init="normal", num_epochs=100, batch_size=len(X)//10)
trainLoss, validLoss, trainAccs, validAccs = model.fit(
    trainX, trainY, validX=validX, validY=validY, logs=True)
print("Train Accuracy Tanh: {}".format(model.score(trainX, trainY)))
print("Test Accuracy Tanh: {}".format(model.score(testX, testY)))
print("Validation Accuracy Tanh: {}".format(model.score(validX, validY)))
