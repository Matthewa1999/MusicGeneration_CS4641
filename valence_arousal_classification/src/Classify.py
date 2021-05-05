from sklearn import svm
from datetime import datetime
import numpy as np
from tensorflow import keras
from midiFeatures import *
import json
import matplotlib.pyplot as plt

def classify(emotion):

    midiFeaturesDict, testingMatrix = processMidi('outputs/' + emotion)

    valModel = pickle.load(open('models/valence.sav', 'rb'))
    arousModel = pickle.load(open('models/arousal.sav', 'rb'))

    counter = 0

    valPreds = valModel.predict(testingMatrix)
    arousPreds = arousModel.predict(testingMatrix)

    valPreds = valPreds.reshape((valPreds.size, 1))
    arousPreds = arousPreds.reshape((arousPreds.size, 1))

    preds = np.hstack((np.array(valPreds), np.array(arousPreds)))

    for i in np.arange(preds.shape[0]):
        midiFeaturesDict[i]["Valence"] = preds[i, 0]
        midiFeaturesDict[i]["Arousal"] = preds[i, 1]

    jsonDict = {}
    for entry in midiFeaturesDict:
        jsonDict[entry["Name"]] = entry

    with open('annotations/' + emotion + '/newClassifications.json', 'w') as json_file:
        json.dump(midiFeaturesDict, json_file)

    return preds

if __name__ == '__main__':
    classify('low_sad')
