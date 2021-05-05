import glob
import pickle
import numpy as np
import matplotlib
from music21 import converter, instrument, note, chord, tempo
import json
import jsonlines
import scipy

def processMidi(midiDir):
    pitchClassMatrix = np.zeros((1, 12))
    tempoMeanArr = np.array([])
    tempoStdArr = np.array([])
    chordCounts = np.array([])
    counter = 0
    pitchClasses = np.array(['C', 'D-', 'D', 'E-', 'E', 'F', 'G-', 'G', 'A-', 'A', 'B-', 'B'])

    midiFeaturesDict = []

    for file in glob.glob(midiDir + "/*.mid"):

        if counter % 10 == 0:
            print(str(counter) + " Files")

        fileName = file.split('/')[1]
        chordCounter = 0

        if counter > 0:
            pitchClassMatrix = np.vstack((pitchClassMatrix, np.zeros((1, 12))))
        midi = converter.parse(file)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
            if len(notes_to_parse.elements) < 10:
                notes_to_parse = s2.parts[1].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        pieceTempo = np.array([])

        for element in notes_to_parse:
            shift = 0
            if hasattr(element, 'tonicPitchNameWithCase'):
                try:
                    key = element.tonicPitchNameWithCase.upper()
                    shift = -1 * np.where(pitchClasses == key)[0][0]
                except:
                    try:
                        shift = -1 * np.where(pitchClasses == key[0])[0][0]
                    except:
                        # print(fileName)
                        pass

            if isinstance(element, tempo.MetronomeMark):
                try:
                    bpm = element.getQuarterBPM()
                    pieceTempo = np.append(pieceTempo, bpm)


                except:
                    pass

            if isinstance(element, note.Note):
                noteName = str(element.pitch)

                if len(noteName) == 2:
                    index = np.where(pitchClasses == noteName[0])[0][0]
                    pitchClassMatrix[counter, index] = pitchClassMatrix[counter, index] + 1
                else:
                    index = np.where(pitchClasses == noteName[:1])[0][0]
                    pitchClassMatrix[counter, index] = pitchClassMatrix[counter, index] + 1
            elif isinstance(element, chord.Chord):
                chordCounter += 1
                for n in element.normalOrder:
                    pitchClassMatrix[counter, n] = pitchClassMatrix[counter, n] + 1

        if all(pitchClassMatrix[counter, :] == 0):
            pitchClassMatrix = np.delete(pitchClassMatrix, counter, axis=0)
            if pitchClassMatrix.size == 0:
                pitchClassMatrix = np.zeros((1, 12))

        else:
            pitchClassMatrix[counter, :] = pitchClassMatrix[counter, :] / np.linalg.norm(pitchClassMatrix[counter, :])
            pitchClassMatrix[counter, :] = np.roll(pitchClassMatrix[counter, :], shift)

            tempoMeanArr = np.append(tempoMeanArr, np.mean(pieceTempo))
            tempoStdArr = np.append(tempoStdArr, np.std(pieceTempo))

            chordCounts = np.append(chordCounts, chordCounter)

            newDict = {"Name": fileName, "PitchClass": pitchClassMatrix[counter, :].tolist(), "TempoMean": np.mean(pieceTempo),
                       "TempoStd": np.std(pieceTempo), "Valence": 0, "Arousal": 0}

            midiFeaturesDict.append(newDict)
            counter += 1

    tempoMeanArr[np.where(np.isnan(tempoMeanArr))[0]] = 0
    tempoStdArr[np.where(np.isnan(tempoStdArr))[0]] = 0

    featureMatrix = np.hstack((np.round(pitchClassMatrix, 4), tempoMeanArr.reshape((tempoMeanArr.size, 1))))
    featureMatrix = np.hstack((featureMatrix, tempoStdArr.reshape((tempoStdArr.size, 1))))
    featureMatrix = np.hstack((featureMatrix, chordCounts.reshape((chordCounts.size, 1))))

    return midiFeaturesDict, featureMatrix


def getGroundTruth(path):
    with open(path) as midi:
        midi_data = json.load(midi)

    pieceValence = np.array([])
    pieceArousal = np.array([])
    labelDict = {}
    errorPieceList = []
    pieceName = list(midi_data["annotations"].keys())[0].split('_')[0]

    for piece in midi_data["annotations"]:

        try:
            if piece.split('_')[0] != pieceName:
                valence = (np.sign(np.mean(pieceValence)) + 1) / 2
                arousal = (np.sign(np.mean(pieceArousal)) + 1) / 2
                midiName = midi_data["pieces"][pieceName]["midi"]
                pieceDict = {midiName: [valence, arousal]}
                labelDict.update(pieceDict)

                pieceValence = np.array([])
                pieceArousal = np.array([])

                pieceName = piece.split('_')[0]

            localValence = np.array([])
            localArousal = np.array([])

            for valence in midi_data["annotations"][piece]["valence"]:
                localValence = np.append(localValence, valence)
            for arousal in midi_data["annotations"][piece]["arousal"]:
                localArousal = np.append(localArousal, arousal)

            localValence = max(localValence.min(), localValence.max(), key=abs)
            localArousal = max(localArousal.min(), localArousal.max(), key=abs)

            pieceValence = np.append(pieceValence, localValence)
            pieceArousal = np.append(pieceArousal, localArousal)
        except:
            if pieceName not in errorPieceList:
                errorPieceList.append(pieceName)

    #For last entry because loop will stop
    try:
        valence = (np.sign(np.mean(pieceValence)) + 1) / 2
        arousal = (np.sign(np.mean(pieceArousal)) + 1) / 2
        midiName = midi_data["pieces"][pieceName]["midi"]
        pieceDict = {midiName: [valence, arousal]}
        labelDict.update(pieceDict)
    except:
        if pieceName not in errorPieceList:
            errorPieceList.append(pieceName)

    # print(errorPieceList)

    return labelDict