from exceptions import tooSmall
from batch import Batch
from sys import exit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def RandomForest(data):
    try:

        BATCH_PERCENT = 0.5
        EPOCHS = 500
        LEARNING_RATE = 0.0001
        ACCURACY_COEFF = 0.3

        EPOCHS = round(EPOCHS)

        exceptionz = []
        exception2 = 0 < BATCH_PERCENT <= 1
        exceptionz.append(exception2)
        exception3 = EPOCHS >= 1
        exceptionz.append(exception3)
        exception5 = 0 < LEARNING_RATE <= 1
        exceptionz.append(exception5)
        exception8 = 0 < ACCURACY_COEFF < 1
        exceptionz.append(exception8)
        exceptionz = np.array(exceptionz)

        if not exceptionz.any():
            raise tooSmall(name='ALL')
        if (exceptionz == False).sum() >= 2:
            raise tooSmall(name='ANY')
        if not exception2:
            raise tooSmall(name='BATCH_PERCENT', b=BATCH_PERCENT)
        if not exception3:
            raise tooSmall(name='EPOCHS', b=EPOCHS)
        if not exception5:
            raise tooSmall(name='LEARNING_RATE', b=LEARNING_RATE)
        if not exception8:
            raise tooSmall(name='ACCURACY_COEFF', a=0.9999999999, b=ACCURACY_COEFF)

    except tooSmall as e:
        if e.name == 'ALL':
            BATCH_PERCENT = e.b
            EPOCHS = e.b
            LEARNING_RATE = e.b
            ACCURACY_COEFF = e.b - 0.0000000001
        if e.name == 'DATA_SET_SIZE':
            DATA_SET_SIZE = e.b
        if e.name == 'BATCH_PERCENT':
            BATCH_PERCENT = e.b
        if e.name == 'EPOCHS':
            EPOCHS = e.b
        if e.name == 'BATCH_SIZE':
            BATCH_SIZE = e.b
        if e.name == 'LEARNING_RATE':
            LEARNING_RATE = e.b
        if e.name == 'NOISE':
            NOISE = e.b
        if e.name == 'FEATURES_NUMBER':
            FEATURES_NUMBER = e.b
        if e.name == 'ACCURACY_COEFF':
            ACCURACY_COEFF = e.b - 0.0000000001
        if e.name == 'ANY':
            exit(1)
    finally:

        # Preprocessing:

        labels = data.loc[:, 'Survived']
        data = data.loc[:, ['Pclass', 'Age',  'Sex', 'Mother', 'Children', 'SWomenAndChildrenInFamily', 'SAllWomenAndChildrenInFamily',]]
        data.Sex = (data.Sex == 'male')

        # Splitting the data:
        dataTr, dataTe, labelsTr, labelsTe = train_test_split(data, labels, test_size=0.2, random_state=1)

        # Building the model:
        model = RandomForestClassifier(n_estimators=500)

        # Fitting the model:
        model.fit(dataTr, labelsTr)

        # Printing the train score:
        print(f'Train accuracy: {model.score(dataTr, labelsTr)}')

        # Evaluating and printing the test score:
        print(f'Test accuracy: {model.score(dataTe, labelsTe)}')
