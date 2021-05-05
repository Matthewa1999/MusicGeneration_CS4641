from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime
import numpy as np
from midiFeatures import *
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import shutil
import os

def training():

    now = datetime.now()
    modelFileName = now.strftime("%m-%d-%Y_%H_%M_%S")

    midiFeaturesDict, trainingMatrix = processMidi("training")

    labelDict1 = getGroundTruth('annotations/vgmidi_raw_1.json')
    labelDict2 = getGroundTruth('annotations/vgmidi_raw_2.json')

    labelDict = labelDict1.copy()
    labelDict.update(labelDict2)

    moveData(labelDict)

    gData = np.zeros((2))
    counter = 0

    for midiFile in midiFeaturesDict:
        try:
            valence = labelDict[midiFile["Name"]][0]
            arousal = labelDict[midiFile["Name"]][1]
            gData = np.vstack((gData, [valence, arousal]))
            counter += 1
        except:
            trainingMatrix = np.delete(trainingMatrix, counter, axis=0)

    gData = gData[1:]

    clfValence = SVC(C=1.4, kernel='linear', class_weight={0: 1.31, 1: 0.69})
    clfValence.fit(trainingMatrix, gData[:, 0])

    clfArousal = SVC(kernel='linear')
    clfArousal.fit(trainingMatrix, gData[:, 1])

    # pickle.dump(clfValence, open("models/valence.sav", 'wb'))
    # pickle.dump(clfArousal, open("models/arousal.sav", 'wb'))

def splitTrainingData(labelDict):

    high_happy = np.array([])
    low_happy = np.array([])
    high_sad = np.array([])
    low_sad = np.array([])

    for entry in labelDict:
        if labelDict[entry][1] == 1 and labelDict[entry][0] == 1:
            high_happy = np.append(high_happy, entry)
        elif labelDict[entry][1] == 0 and labelDict[entry][0] == 1:
            low_happy = np.append(low_happy, entry)
        elif labelDict[entry][1] == 1 and labelDict[entry][0] == 0:
            high_sad = np.append(high_sad, entry)
        elif labelDict[entry][1] == 0 and labelDict[entry][0] == 0:
            low_sad = np.append(low_sad, entry)

    return high_happy, low_happy, high_sad, low_sad

def clearFolders():

    for file in glob.glob("training/high_happy" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("training/low_happy" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("training/high_sad" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("training/low_sad" + "/*.mid"):
        os.remove(file)

def moveData(labelDict):

    clearFolders()

    high_happy, low_happy, high_sad, low_sad = splitTrainingData(labelDict)

    for file in glob.glob("training" + "/*.mid"):
        if np.isin(file[9:], high_happy):
            original = file
            target = 'training/high_happy/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], low_happy):
            original = file
            target = 'training/low_happy/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], high_sad):
            original = file
            target = 'training/high_sad/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], low_sad):
            original = file
            target = 'training/low_sad/' + file[9:]
            shutil.copyfile(original, target)

if __name__ == '__main__':
    training()