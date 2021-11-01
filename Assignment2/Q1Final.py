import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
plt.style.use('seaborn')
plt.style.use('ggplot')
plt.style.use('dark_background')


# Loading the dataset
df = pd.read_csv('Weights/Ques1/pm2.5.csv')

# Dropping the No Column
df.drop(["No"], axis=1, inplace=True)

# Dropping the nan values
df.dropna(inplace=True)


def encodeCBWD(val):
    """
    This function encodes the CBWD column
    """
    if val == 'SE':
        return 1
    elif val == 'NW':
        return 2
    elif val == 'NE':
        return 3
    return 4


# Encoding the cbwd entries
df['cbwd'] = df['cbwd'].apply(encodeCBWD)


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


# Splitting the dataset into train, valid and test
trainData, validData, testData = train_test_split(
    df, trainSize=0.70, testSize=0.15, random_state=0)

# Creating the train, validation and test sets with their labels
trainY = trainData['month']
trainX = trainData.drop(['month'], axis=1)
validY = validData['month']
validX = validData.drop(['month'], axis=1)
testY = testData['month']
testX = testData.drop(['month'], axis=1)


def calcAccuracy(y_, y):
    """
    This function calculates the accuracy of the model
    y_: The predicted values
    y: The actual values
    return: The accuracy
    """
    return (y_ == y).sum()*100/len(y_)


# task a
print('Decision Tree on Entropy')
decisionTreeEntropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0)
decisionTreeEntropy.fit(trainX, trainY)
print('train acc: ', decisionTreeEntropy.score(trainX, trainY))
print('valid acc: ', decisionTreeEntropy.score(validX, validY))
print('test acc: ', decisionTreeEntropy.score(testX, testY))

print('Decision Tree on Gini Index')
decisionTreeEntropy = DecisionTreeClassifier(criterion='gini', random_state=0)
decisionTreeEntropy.fit(trainX, trainY)
print('train acc: ', decisionTreeEntropy.score(trainX, trainY))
print('valid acc: ', decisionTreeEntropy.score(validX, validY))
print('test acc: ', decisionTreeEntropy.score(testX, testY))


# task b

# depths for the trees
depths = [2, 4, 8, 10, 15, 30]

# lists to store the accuracies
testingAcc = []
trainingAcc = []
validationAcc = []

# looping over the depths and training the trees and recording the accuracies
for depth in depths:
    decisionTreeEntropy = DecisionTreeClassifier(
        criterion='entropy', max_depth=depth)
    decisionTreeEntropy.fit(trainX, trainY)
    trainingAcc.append(decisionTreeEntropy.score(trainX, trainY))
    validationAcc.append(decisionTreeEntropy.score(validX, validY))
    testingAcc.append(decisionTreeEntropy.score(testX, testY))

# Plotting the curves
plt.plot(depths, trainingAcc, label='Training Accuracy', color='red')
plt.plot(depths, testingAcc, label='Testing Accuracy', color='green')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.legend()
plt.savefig('Plots/Ques1/partb.png')
plt.show()


# task c

# list to store the predictions
predictions = []

for i in range(100):
    # initializing the tree
    decisionTreeRF = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    # randomly generating 50% of the train data
    trainDataRF, testDataRF = train_test_split(
        trainData, trainSize=0.5, testSize=0.5, random_state=i+1)

    # splitting train data in X and y
    trainDataRF_X = trainDataRF.drop(['month'], axis=1)
    trainDataRF_y = trainDataRF['month']
    testDataRF_X = testDataRF.drop(['month'], axis=1)
    testDataRF_y = testDataRF['month']

    # fitting the tree
    decisionTreeRF.fit(trainDataRF_X, trainDataRF_y)

    # predicting the global test data
    predictions.append(decisionTreeRF.predict(testX))


pred = np.array(predictions)

# generating the majority vote
y_ = []
for i in range(len(pred[0])):
    y_.append(np.argmax(np.bincount(pred[:, i])))


# task d
maxDepths = [4, 8, 10, 15, 20, 30]


# dictionaries to store the accuracies and predictions
trainPreds = {}
validPreds = {}
testPreds = {}
trainAccs = {}
validAccs = {}
testAccs = {}
for dep in maxDepths:
    trainPreds[dep] = []
    validPreds[dep] = []
    testPreds[dep] = []
    trainAccs[dep] = []
    validAccs[dep] = []
    testAccs[dep] = []
    for i in tqdm(range(100)):
        # initializing the tree
        decisionTreeRF = DecisionTreeClassifier(
            criterion='entropy', max_depth=dep)
        # generating the train and test data
        trainDataRF, testDataRF = train_test_split(
            trainData, trainSize=0.5, testSize=0.5, random_state=i+1)
        trainDataRF_X = trainDataRF.drop(['month'], axis=1)
        trainDataRF_y = trainDataRF['month']
        testDataRF_X = testDataRF.drop(['month'], axis=1)
        testDataRF_y = testDataRF['month']
        # fitting the tree
        decisionTreeRF.fit(trainDataRF_X, trainDataRF_y)
        # predicting the train data, valid data and test data
        trainPreds[dep].append(decisionTreeRF.predict(trainX))
        validPreds[dep].append(decisionTreeRF.predict(validX))
        testPreds[dep].append(decisionTreeRF.predict(testX))
        # calculating the prediction by taking the majority vote
        trainPred = np.array(trainPreds[dep])
        y_train = []
        for i in range(len(trainPred[0])):
            y_train.append(np.argmax(np.bincount(trainPred[:, i])))
        trainAccs[dep].append(calcAccuracy(y_train, trainY))
        testPred = np.array(testPreds[dep])
        y_test = []
        for i in range(len(testPred[0])):
            y_test.append(np.argmax(np.bincount(testPred[:, i])))
        testAccs[dep].append(calcAccuracy(y_test, testY))
        validPred = np.array(validPreds[dep])
        y_valid = []
        for i in range(len(validPred[0])):
            y_valid.append(np.argmax(np.bincount(validPred[:, i])))
        validAccs[dep].append(calcAccuracy(y_valid, validY))


# plotting the curves and showing the relation between the accuracies, depths and number of trees on the training, testing and validation data
color = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
ctr = 0
for i in maxDepths:
    plt.plot(
        trainAccs[i], label='Training Accuracy for depth ' + str(i), color=color[ctr])
    plt.plot(validAccs[i], label='Validation Accuracy for depth ' +
             str(i), color=color[ctr], linestyle='dashed')
    plt.plot(testAccs[i], label='Testing Accuracy for depth ' +
             str(i), color=color[ctr], linestyle='dotted')
    ctr += 1
plt.xlabel('number of trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('Plots/Ques1/partc.png')
plt.show()


# task e

# initialising the number of estimators
noOfEstimators = [4, 8, 10, 15, 20]


# lists were created to save the accuracies
adaTrainAccs = []
adaTestAccs = []
adaValidAccs = []
for i in tqdm(range(len(noOfEstimators))):
    # initializing the decision tree
    dtForAda = DecisionTreeClassifier(criterion='entropy')
    est = noOfEstimators[i]

    # initializing the adaBoost
    adaBoost = AdaBoostClassifier(base_estimator=dtForAda, n_estimators=est)

    # fitting the adaBoost
    adaBoost.fit(trainX, trainY)

    # getting the accuracies on the training, testing and validation data
    adaTrainAccs.append(adaBoost.score(trainX, trainY))
    adaTestAccs.append(adaBoost.score(testX, testY))
    adaValidAccs.append(adaBoost.score(validX, validY))


# plotting the curve between the number of estimators and the accuracies for the adaboost
plt.plot(noOfEstimators, adaTrainAccs, label='Training Accuracy', color='blue')
plt.plot(noOfEstimators, adaTestAccs, label='Testing Accuracy', color='red')
plt.plot(noOfEstimators, adaValidAccs,
         label='Validation Accuracy', color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Estimators')
plt.legend()
plt.show()
