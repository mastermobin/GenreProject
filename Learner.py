from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, AveragePooling1D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from keras.utils import to_categorical

bankdata = pd.read_csv("./Dataset.csv")
print("Read Done")
print(bankdata.shape)

X = bankdata.drop('Class', axis=1).to_numpy()
y = bankdata['Class'].to_numpy()

print(X)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.reshape(-1, 100, 1)
X_test = X_test.reshape(-1, 100, 1)
print("Split Done")

model = Sequential()
model.add(Conv1D(40, 10, activation='relu',
                 padding='causal', input_shape=(100, 1)))
model.add(AveragePooling1D(4))
model.add(Conv1D(20, 5, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model.fit(X_train, y_train, epochs=1, batch_size=10)
dump(model, "NN.model")

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc, 'test_loss', test_loss)
