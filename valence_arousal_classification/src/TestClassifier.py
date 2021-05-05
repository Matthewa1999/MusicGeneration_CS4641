from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import numpy as np
from tensorflow import keras
from midiFeatures import *
import matplotlib.pyplot as plt
from TrainClassifier import*

def experiment():
    midiFeaturesDict, featureMatrix = processMidi("training")
    pValArr = np.array([])
    rValArr = np.array([])
    fValArr = np.array([])
    pArousArr = np.array([])
    rArousArr = np.array([])
    fArousArr = np.array([])

    testSplit = 0.7

    for i in np.arange(15):

        gDataTraining = np.zeros((2))
        gDataTesting = np.zeros((2))

        trainingRows = np.sort(np.random.choice(featureMatrix.shape[0], int(featureMatrix.shape[0]*testSplit),
                                                replace=False))
        testingRows = np.sort(np.setdiff1d(np.arange(featureMatrix.shape[0]), trainingRows))

        trainingMatrix = featureMatrix[trainingRows, :]
        testingMatrix = featureMatrix[testingRows, :]

        labelDict1 = getGroundTruth('annotations/vgmidi_raw_1.json')
        labelDict2 = getGroundTruth('annotations/vgmidi_raw_2.json')
        labelDict = labelDict1.copy()
        labelDict.update(labelDict2)
        counter = 0



        for j in np.arange(len(midiFeaturesDict)):
            try:
                if not np.isin(j, trainingRows):
                    continue
                midiFile = midiFeaturesDict[j]
                valence = labelDict[midiFile["Name"]][0]
                arousal = labelDict[midiFile["Name"]][1]
                gDataTraining = np.vstack((gDataTraining, [valence, arousal]))
                counter += 1
            except:
                trainingMatrix = np.delete(trainingMatrix, counter, axis=0)

        gDataTraining = gDataTraining[1:]
        clfValence = SVC(C=1.4, kernel='linear', class_weight={0: 1.31, 1: 0.69})
        clfValence.fit(trainingMatrix, gDataTraining[:, 0])

        clfArousal = SVC(kernel='linear')
        clfArousal.fit(trainingMatrix, gDataTraining[:, 1])

        counter = 0

        for k in np.arange(len(midiFeaturesDict)):
            try:
                if not np.isin(k, testingRows):
                    continue
                midiFile = midiFeaturesDict[k]
                valence = labelDict[midiFile["Name"]][0]
                arousal = labelDict[midiFile["Name"]][1]
                gDataTesting = np.vstack((gDataTesting, [valence, arousal]))
                counter += 1
            except:
                testingMatrix = np.delete(testingMatrix, counter, axis=0)

        valPreds = clfValence.predict(testingMatrix)
        arousPreds = clfArousal.predict(testingMatrix)

        valPreds = valPreds.reshape((valPreds.size, 1))
        arousPreds = arousPreds.reshape((arousPreds.size, 1))

        preds = np.hstack((valPreds, arousPreds))

        gDataTesting = gDataTesting[1:]

        print('pred 0\'s: ' + str(np.where(preds[:, 0] == 0)[0].size))
        print('gData 0\'s: ' + str(np.where(gDataTesting[:, 0] == 0)[0].size))
        print('pred 1\'s: ' + str(np.where(preds[:, 0] == 1)[0].size))
        print('gData 1\'s: ' + str(np.where(gDataTesting[:, 0] == 1)[0].size))

        pVal, rVal, fVal, pArous, rArous, fArous = evaluate(preds, gDataTesting, plot=False, print=False)

        pValArr = np.append(pValArr, pVal)
        rValArr = np.append(rValArr, rVal)
        fValArr = np.append(fValArr, fVal)
        pArousArr = np.append(pArousArr, pArous)
        rArousArr = np.append(rArousArr, rArous)
        fArousArr = np.append(fArousArr, fArous)

        print(i)

    print('\n')

    print(pValArr)
    print(rValArr)
    print(fValArr)
    print(pArousArr)
    print(rArousArr)
    print(fArousArr)

    print("\nprecision_valence: " + str(np.round(np.mean(pValArr), 3)))
    print("recall_valence: " + str(np.round(np.mean(rValArr), 3)))
    print("f1_valence: " + str(np.round(np.mean(fValArr), 3)))
    print("precision_arousal: " + str(np.round(np.mean(pArousArr), 3)))
    print("recall_arousal: " + str(np.round(np.mean(rArousArr), 3)))
    print("f1_arousal: " + str(np.round(np.mean(fArousArr), 3)))


def test(modelName):

    midiFeaturesDict, testingMatrix = processMidi("test")

    model = pickle.load(open('models/' + modelName, 'rb'))


    # valModel = keras.models.load_model("models/" + modelName + "/Valence.sav")
    # arousalModel = keras.models.load_model("models/" + modelName + "/Arousal.sav")
    # valPreds = valModel.predict(testingMatrix)
    # arousalPreds = arousalModel.predict(testingMatrix)
    # preds = np.hstack((valPreds, arousalPreds))

    labelDict1 = getGroundTruth('annotations/vgmidi_raw_1.json')
    labelDict2 = getGroundTruth('annotations/vgmidi_raw_2.json')

    labelDict = labelDict1.copy()
    labelDict.update(labelDict2) # [valence, arousal]

    gData = np.zeros((2))
    counter = 0

    preds = model.predict(testingMatrix)

    for midiFile in midiFeaturesDict:
        try:
            valence = labelDict[midiFile["Name"]][0]
            arousal = labelDict[midiFile["Name"]][1]
            gData = np.vstack((gData, [valence, arousal]))
            counter += 1
        except:
            preds = np.delete(preds, counter, axis=0)
            testingMatrix = np.delete(testingMatrix, counter, axis=0)

    pVal, rVal, fVal, pArous, rArous, fArous = evaluate(preds, gData[1:], plot=False)




def evaluate(preds, gData, plot=True, print=True):
    gtVal = gData[:, 0]
    predVal = preds[:, 0]
    gtArous = gData[:, 1]
    predArous = preds[:, 1]

    if plot:
        x = np.arange(len(preds))
        fig1 = plt.figure(1) #Valence
        plt.plot(x, predVal)
        plt.plot(x, gtVal)
        plt.legend(['Predicted', 'Ground truth'])
        fig1.suptitle("Valence")
        plt.xlabel('File')
        plt.ylabel('Valence')

        fig2 = plt.figure(2) #Arousal
        plt.plot(x, predArous)
        plt.plot(x, gtArous)
        plt.legend(['Predicted', 'Ground truth'])
        fig2.suptitle("Arousal")
        plt.xlabel('File')
        plt.ylabel('Arousal')

        plt.show()



    pVal = np.round(precision_score(gtVal, predVal), 4)
    rVal = np.round(recall_score(gtVal, predVal), 4)
    fVal = np.round(f1_score(gtVal, predVal), 4)

    pArous = np.round(precision_score(gtArous, predArous), 4)
    rArous = np.round(recall_score(gtArous, predArous), 4)
    fArous = np.round(f1_score(gtArous, predArous), 4)

    if print:
        print("         Valence    Arousal")
        print("Precision: " + str(pVal) + '    ' + str(pArous))
        print("Recall: " + str(rVal) + '    ' + str(rArous))
        print("F1: " + str(fVal) + '    ' + str(fArous))

    return pVal, rVal, fVal, pArous, rArous, fArous

if __name__ == '__main__':
    # test('04-13-2021_19_36_46.sav')
    experiment()