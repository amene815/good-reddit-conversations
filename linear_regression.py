
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from numpy import random
from sklearn.model_selection import train_test_split



X_train = []
y_train = []
X_test = []
y_test = []

with open('data/train.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        X_train.append(row[:-1])
        y_train.append(row[-1])


with open('data/validation.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        X_test.append(row[:-1])
        y_test.append(row[-1])


model = LinearRegression().fit(X_train, y_train)

r_sq = model.score(post_embeddings, response)
print('R Squared:', r_sq)

coefficients = model.coef_
print("Coefficients")
print(coefficients)

train_predictions = model.predict(X_train)
train_error = 0
for i in range(len(train_predictions)):
    train_error += (train_predictions[i] - y_train[i])**2
train_error = train_error / len(train_predictions)
print("Training Error: ", train_error)

test_predictions = model.predict(X_test)
test_error = 0
for i in range(len(test_predictions)):
    test_error += (test_predictions[i] - y_test[i])**2
test_error = test_error / len(test_predictions)
print("Test Error: ", test_error)
