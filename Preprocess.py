import os
import pandas as pd
import numpy as np
from numpy import random as rn
import mido
from mido import MidiFile

WINDOW_SIZE = 100
STRIDE = 10


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


res = np.ndarray((0, 101), dtype='uint8')
train_data = pd.read_csv("train.csv")
train_data = train_data.set_index(["file"])
print(train_data.head())

i = 0
for index, data in train_data.iterrows():
    i += 1
    filename = data.name
    label = data.genre

    midi = MidiFile(os.getcwd() + "/Midi/" + filename)
    notes = readMiDi(midi)
    print(str(i) + "/" + str(train_data.size) + ":" + str(notes.size))
    if notes.size > WINDOW_SIZE:
        nrows = ((notes.size-WINDOW_SIZE)//STRIDE)+1
        lbs = np.ndarray((nrows), dtype='uint8')
        lbs.fill(label)
        windows = striding(notes)
        windows = np.c_[windows, lbs]
        res = np.concatenate((res, windows), axis=0)

print(res.shape)

df = pd.DataFrame(res)
df = df.rename(columns={WINDOW_SIZE: "Class"})
np.random.shuffle(df.values)

print(df.head())

df.to_csv("Dataset.csv", index=False, header=True)
