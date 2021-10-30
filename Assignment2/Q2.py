import Q1
from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pylab as plt

train_x, train_y = loadlocal_mnist(
    "Weights/Ques2/extracted/train-images-idx3-ubyte", "Weights/Ques2/extracted/train-labels-idx1-ubyte")
test_x, test_y = loadlocal_mnist(
    "Weights/Ques2/extracted/t10k-images-idx3-ubyte", "Weights/Ques2/extracted/t10k-labels-idx1-ubyte")
train_x = preprocessing.normalize(train_x)
test_x = preprocessing.normalize(test_x)
enc = OneHotEncoder(sparse=False, categories='auto')
train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
test_y = enc.transform(test_y.reshape(len(test_y), -1))


network = Q1.MyNeuralNetwork(n_layers=6, layer_sizes=[
                             784, 256, 128, 64, 32, 10], activation="relu", learning_rate=0.08, weight_init="normal", batch_size=200, num_epochs=101)
network.fit(network.train_x, network.train_y)
network.tsne()


network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="relu", learning_rate=0.1, weight_init="normal", batch_size=200, num_epochs=101)
network.fit(network.train_x, network.train_y)
network.saveModel(network, "ReLU_normal")
y_pred_test = network.predict(network.test_x)
print("Test Accuracy:", network.score(network.test_x, network.test_y))
plt.figure()
plt.plot(np.arange(len(network.Avgcost_epoch)),
         network.Avgcost_epoch, "r", label="Training error")
plt.plot(np.arange(len(network.test_epoch_error)),
         network.test_epoch_error, "b", label="Test error")
plt.xlabel("epochs")
plt.ylabel("Error")
plt.savefig("RELU")
plt.show()
print(".....................................................................................................................")

network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="linear", learning_rate=0.1, weight_init="normal", batch_size=200, num_epochs=101)
network.fit(network.train_x, network.train_y)
network.saveModel(network, "Linear_normal")
y_pred_test = network.predict(network.test_x)
print("Test Accuracy:", network.score(network.test_x, network.test_y))
plt.figure()
plt.plot(np.arange(len(network.Avgcost_epoch)),
         network.Avgcost_epoch, "r", label="Training error")
plt.plot(np.arange(len(network.test_epoch_error)),
         network.test_epoch_error, "b", label="Test error")
plt.xlabel("epochs")
plt.ylabel("Error")
plt.savefig("Linear")
plt.show()
print(".....................................................................................................................")

network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="tanh", learning_rate=0.1, weight_init="normal", batch_size=200, num_epochs=101)
network.fit(network.train_x, network.train_y)
network.saveModel(network, "TanH_normal")
y_pred_test = network.predict(network.test_x)
print("Test Accuracy:", network.score(network.test_x, network.test_y))
plt.figure()
plt.plot(np.arange(len(network.Avgcost_epoch)),
         network.Avgcost_epoch, "r", label="Training error")
plt.plot(np.arange(len(network.test_epoch_error)),
         network.test_epoch_error, "b", label="Test error")
plt.xlabel("epochs")
plt.ylabel("Error")
plt.savefig("Tanh")
plt.show()
print(".....................................................................................................................")

network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="sigmoid", learning_rate=0.1, weight_init="normal", batch_size=30, num_epochs=101)
network.fit(network.train_x, network.train_y)
network.saveModel(network, "Sigmoid_normal")
y_pred_test = network.predict(network.test_x)
print("Test Accuracy:", network.score(network.test_x, network.test_y))
plt.figure()
plt.plot(np.arange(len(network.Avgcost_epoch)),
         network.Avgcost_epoch, "r", label="Training error")
plt.plot(np.arange(len(network.test_epoch_error)),
         network.test_epoch_error, "b", label="Test error")
plt.xlabel("epochs")
plt.ylabel("Error")
plt.savefig("Sigmoid")
plt.show()
print(".....................................................................................................................")

network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="sigmoid", learning_rate=0.1, weight_init="normal", batch_size=100, num_epochs=101)
network.SkLearn(batch_size=100, activationn="logistic")
network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="relu", learning_rate=0.1, weight_init="normal", batch_size=100, num_epochs=101)
network.SkLearn(batch_size=100, activationn="relu")
network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="linear", learning_rate=0.1, weight_init="normal", batch_size=100, num_epochs=101)
network.SkLearn(batch_size=100, activationn="identity")
network = Q1.MyNeuralNetwork(n_layers=5, layer_sizes=[
                             784, 256, 128, 64, 10], activation="tanh", learning_rate=0.1, weight_init="normal", batch_size=100, num_epochs=101)
network.SkLearn(batch_size=100, activationn="tanh")
