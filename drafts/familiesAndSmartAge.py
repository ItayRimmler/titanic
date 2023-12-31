# Here I'll convert the age that is NaN in a smart way: Master will be the avg age of children,
# Miss will be the avg age of randomly young adult or children (will be randomised according
# to the proportion between them, Mr and Mrs will be the avg age of adults.
# Also, I'll make sure to add Familiysize flag, and Familysurvivesize, and the algorithm will correlate whether it's
# important or no. -- ERR I DON'T KNOW HOW TO HANDLE THESE BECAUSE SIBSP AND PARCH ARE F'ED UP

import pandas as pd
from exceptions import tooSmall
from modelClasses import Model
from batch import Batch
from sys import exit
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import numpy as np


def FamiliesAndSmartAge(data):

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

        # PREPROCESSING THE DATA:

        # We start by getting what we need from the dataframe:
        labels = data.loc[:, 'Survived']
        data = data.loc[:, ['Age', 'Name', 'Sex', 'Pclass', 'Fare']]

        dataTr, dataTe, labelsTr, labelsTe = train_test_split(data.loc[:, ['Age', 'Pclass', 'Fare']], labels, test_size=0.2, random_state=1)

        labelsTr = t.tensor(labelsTr.values, dtype=t.float32)
        labelsTr = labelsTr.view(labelsTr.size(0), -1)
        labelsTe = t.tensor(labelsTe.values, dtype=t.float32)
        labelsTe = labelsTe.view(labelsTe.size(0), -1)
        dataTr = t.tensor(dataTr.values, dtype=t.float32)
        dataTr = dataTr.view(dataTr.size(0), -1)
        dataTe = t.tensor(dataTe.values, dtype=t.float32)
        dataTe = dataTe.view(dataTe.size(0), -1)

        # Setting the model up:
        model = Model(3, 1, 100)
        criterion = nn.BCELoss()
        optimizer = t.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):

            # Setting the data to provide gradients:
            dataTr.requires_grad_(True)

            # Preparing the batch:
            x, y, size = Batch(dataTr, labelsTr, dataTr.size(0) * BATCH_PERCENT)

            # Setting the optimizer:
            optimizer.zero_grad()

            # Forward step:
            outputs = model.cont2binNonLin2(x)

            # MAYBE THE PROBLEM IS IN THE ROUNDING.... I'LL CHECK IT TOMORROW

            # Rounding results in order to get either 0 or 1:
            temp = outputs
            temp = temp.detach()
            temp = np.array(temp)
            modeAni = mode(temp)[0][0]
            # if not round(0.1 * EPOCHS) == 0:
            #     if epoch % round(0.1 * EPOCHS) == 0:
            #         temp1 = np.array(y)
            #         print(modeAni)
            #         k, bins, myhist = plt.hist(temp, bins=400)
            #         k, bins, myhist1 = plt.hist(temp1, bins=400)
            #         plt.show()
            with t.no_grad():
                outputs = (outputs > modeAni).float()
            outputs.requires_grad_(True)

            # Calculating loss:
            loss = criterion(outputs, y)

            # Fitting:
            loss.backward()
            optimizer.step()

            # Calculating accuracy:
            acc = (outputs == y).float().sum()
            finalAcc = acc / (size)

            # Printing and results:
            if not round(0.1 * EPOCHS) == 0:
                if epoch % round(0.1 * EPOCHS) == 0:
                    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item()}, Accuracy: {finalAcc}')

            if epoch == EPOCHS - 1:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item()}, Accuracy: {finalAcc}')

            # EVALUATION:

            # Setting model to evaluation mode:
        model.eval()

        # Forward step:
        outputs = model.cont2binNonLin(dataTe)

        # Rounding results in order to get either 0 or 1:
        outputs = t.round(outputs)

        # Calculating loss:
        loss = criterion(outputs, labelsTe)

        # Calculating accuracy:
        acc = (outputs == labelsTe).float().sum()
        finalAcc = acc / labelsTe.size(0)

        # Printing result:
        print(f'Test results, Loss: {loss.item()}, Accuracy: {finalAcc}')

