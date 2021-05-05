import json
import numpy as np
import shutil
import glob
import os

def moveData():

    clearFolders()

    high_happy, low_happy, high_sad, low_sad = splitData()

    for file in glob.glob("classify" + "/*.mid"):
        if np.isin(file[9:], high_happy):
            original = file
            target = 'classify/high_happy/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], low_happy):
            original = file
            target = 'classify/low_happy/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], high_sad):
            original = file
            target = 'classify/high_sad/' + file[9:]
            shutil.copyfile(original, target)
        elif np.isin(file[9:], low_sad):
            original = file
            target = 'classify/low_sad/' + file[9:]
            shutil.copyfile(original, target)

def splitData():
    file = open('annotations/newClassifications.json')
    data = json.load(file)

    high_happy = np.array([])
    low_happy = np.array([])
    high_sad = np.array([])
    low_sad = np.array([])

    for entry in data:
        if entry["Arousal"] == 1 and entry["Valence"] == 1:
            high_happy = np.append(high_happy, entry["Name"])
        elif entry["Arousal"] == 0 and entry["Valence"] == 1:
            low_happy = np.append(low_happy, entry["Name"])
        elif entry["Arousal"] == 1 and entry["Valence"] == 0:
            high_sad = np.append(high_sad, entry["Name"])
        elif entry["Arousal"] == 0 and entry["Valence"] == 0:
            low_sad = np.append(low_sad, entry["Name"])

    return high_happy, low_happy, high_sad, low_sad

def clearFolders():

    for file in glob.glob("classify/high_happy" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("classify/low_happy" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("classify/high_sad" + "/*.mid"):
        os.remove(file)
    for file in glob.glob("classify/low_sad" + "/*.mid"):
        os.remove(file)

if __name__ == '__main__':
    moveData()