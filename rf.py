from exceptions import tooSmall
from sys import exit
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def RandomForest(data):
    try:
        EPOCHS = 5000

        EPOCHS = round(EPOCHS)

        exceptionz = []
        exception1 = EPOCHS >= 1
        exceptionz.append(exception1)
        exceptionz = np.array(exceptionz)

        # if not exceptionz.any():
        #     raise tooSmall(name='ALL')
        if (exceptionz == False).sum() >= 2:
            raise tooSmall(name='ANY')
        if not exception1:
            raise tooSmall(name='EPOCHS', b=EPOCHS)

    except tooSmall as e:
        if e.name == 'ALL':
            EPOCHS = e.b
        if e.name == 'EPOCHS':
            EPOCHS = e.b
        if e.name == 'ANY':
            exit(1)
    finally:

        # Preprocessing:

        labels = data.loc[:, 'Survived']
        data = data.loc[:, ['Pclass', 'Age',  'Sex', 'Mother', 'Children', 'SurvivedWomenAndChildrenInFamily', 'SurvivedAllWomenAndChildrenInFamily',]]
        data.Sex = (data.Sex == 'male')

        # Splitting the data:
        dataTr, dataTe, labelsTr, labelsTe = train_test_split(data, labels, test_size=0.2, random_state=1)

        # Building the model:
        model = RandomForestClassifier(n_estimators=EPOCHS)

        # Fitting the model:
        model.fit(dataTr, labelsTr)

        # Printing the train score:
        print(f'Train accuracy: {model.score(dataTr, labelsTr)}')

        # Evaluating and printing the test score:
        print(f'Test accuracy: {model.score(dataTe, labelsTe)}')
