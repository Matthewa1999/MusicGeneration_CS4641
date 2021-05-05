from sklearn import svm
from datetime import datetime
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from midiFeatures import *
import json
import matplotlib.pyplot as plt
from Classify import *

def evaluateSingleModel(emotion):

    preds = np.array(classify(emotion))
    print(preds)

    if emotion == 'high_happy':
        gTruth = [1, 1]
    elif emotion == 'high_sad':
        gTruth = [0, 1]
    elif emotion == 'low_happy':
        gTruth = [1, 0]
    elif emotion == 'low_sad':
        gTruth = [0, 0]

    gTruthArr = np.ones(preds.shape) * gTruth


    valPre = precision_score(gTruthArr[:, 0], preds[:, 0])
    valRec = recall_score(gTruthArr[:, 0], preds[:, 0])
    valF1 = f1_score(gTruthArr[:, 0], preds[:, 0])

    arousPre = precision_score(gTruthArr[:, 1], preds[:, 1])
    arousRec = recall_score(gTruthArr[:, 1], preds[:, 1])
    arousF1 = f1_score(gTruthArr[:, 1], preds[:, 1])

    overallAcc = np.sum(np.all(np.equal(preds, gTruthArr), axis=1)) / preds.shape[0]


    print('Valence:\n')
    print('Precision: ' + str(valPre))
    print('Recall: ' + str(valRec))
    print('F1: ' + str(valF1))

    print('\nArousal:\n')
    print('Precision: ' + str(arousPre))
    print('Recall: ' + str(arousRec))
    print('F1: ' + str(arousF1))

    print('\nOverall Accuracy: ' + str(overallAcc))


def evaluateModel():

    emotions = ['high_happy', 'high_sad', 'low_happy', 'low_sad']
    gTruth = np.array([[1, 1], [0, 1], [1, 0], [0, 0]]).reshape(4, 2)
    preds = np.zeros(2)
    gTruthArr = np.zeros(2)

    valAccArr = np.array([])
    arousAccArr = np.array([])
    overallAccArr = np.array([])

    for i in np.arange(len(emotions)):
        predTemp = classify(emotions[i])
        preds = np.vstack((preds, predTemp))

        gTruthTemp = np.ones(predTemp.shape) * gTruth[i, :]
        gTruthArr = np.vstack((gTruthArr, gTruthTemp))

        valAccArr = np.append(valAccArr, accuracy_score(gTruthTemp[:, 0], predTemp[:, 0]))
        arousAccArr = np.append(arousAccArr, accuracy_score(gTruthTemp[:, 1], predTemp[:, 1]))
        overallAccArr = np.append(overallAccArr, np.sum(np.all(np.equal(predTemp, gTruthTemp), axis=1)) /
                                  predTemp.shape[0])

    preds = preds[1:, :]
    gTruthArr = gTruthArr[1:, :]

    valAcc = accuracy_score(gTruthArr[:, 0], preds[:, 0])

    arousAcc = accuracy_score(gTruthArr[:, 1], preds[:, 1])

    overallAcc = np.sum(np.all(np.equal(preds, gTruthArr), axis=1)) / preds.shape[0]

    print('\nValence:')
    print('Accuracy: ' + str(valAcc))
    print('Array: ' + str(valAccArr))

    print('\nArousal:')
    print('Accuracy: ' + str(arousAcc))
    print('Array: ' + str(arousAccArr))

    print('\nOverall Accuracy: ' + str(overallAcc))
    print('Array: ' + str(overallAccArr))

if __name__ == '__main__':
    evaluateModel()