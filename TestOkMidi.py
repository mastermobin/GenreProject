import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
import os
import mido
from mido import MidiFile
from keras.utils import to_categorical
from collections import Counter


WINDOW_SIZE = 100
STRIDE = 20


def readMiDi(midi_file):
    notes = []
    for track in midi_file.tracks:
        for msg in track:
            if hasattr(msg, 'note') and hasattr(msg, 'velocity'):
                if msg.velocity != 0:
                    notes.append(msg.note)
    return np.array(notes)


def striding(a):
    nrows = ((a.size-WINDOW_SIZE)//STRIDE)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, WINDOW_SIZE), strides=(STRIDE*n, n))


train_data = pd.read_csv("test.csv")
train_data = train_data.set_index(["file"])
print(train_data.head())

clf = load("NN.model")

ta = 0
al = 0
i = 0
for index, data in train_data.iterrows():
    i += 1
    filename = data.name
    label = data.genre

    midi = MidiFile(os.getcwd() + "/Midi/" + filename)
    notes = readMiDi(midi)
    answer = np.ndarray(10, dtype='uint32')
    answer.fill(0)
    if notes.size > WINDOW_SIZE:
        nrows = ((notes.size-WINDOW_SIZE)//STRIDE)+1
        windows = striding(notes).reshape(-1, 100, 1)

        yp = clf.predict_classes(windows)

        for item in yp:
            answer[item] += 1

        print(str(i) + ": " + str(notes.size) + ";\t" +
              str(np.argmax(answer)) + " Vs " + str(label) + "\t", end="")

        al += 1
        if str(np.argmax(answer)) == str(label):
            ta += 1
            print()
        else:
            print(answer)

print(str(ta) + "/" + str(al))
