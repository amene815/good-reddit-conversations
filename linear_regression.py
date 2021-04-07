
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import random
from sklearn.model_selection import train_test_split


word_embeddings = np.array([ np.random.randint(2, size=10) for _ in range(50)])



# number of following conversations.
response = np.array(np.random.randint(50, size=50))


X_train, X_test, y_train, y_test = train_test_split(word_embeddings, response, test_size=0.33, random_state=123)


print("word embeddings shape: " , word_embeddings.shape)
print("response shape: " , response.shape)

model = LinearRegression().fit(X_train, y_train)

r_sq = model.score(word_embeddings, response)
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

