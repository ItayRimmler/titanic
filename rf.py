from exceptions import tooSmall
from sys import exit
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def RandomForest(data, test):
    try:
        EPOCHS = 100

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
        passengerId = test.PassengerId
        labels = data.loc[:, 'Survived']
        data = data.drop(['PassengerId','Pclass','Ln','TicketInitials', 'Name','Survived', 'SibSp', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
        data.Sex = (data.Sex == 'male')
        test = test.drop(['PassengerId','Pclass', 'Ln','TicketInitials','Name', 'SibSp', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
        test.Sex = (test.Sex == 'male')

        missingInData = list(set(test.columns) - set(data.columns))
        missingInTest = list(set(data.columns) - set(test.columns))

        for col in missingInData:
            data[col] = 0

        for col in missingInTest:
            test[col] = 0

        data = data.sort_index(axis=1)
        test = test.sort_index(axis=1)

        # Building the model:
        model = RandomForestClassifier(n_estimators=EPOCHS)

        # Fitting the model:
        model.fit(data, labels)

        # Printing the train score:
        print(f'Train accuracy: {model.score(data, labels)}')

        print('Test:')

        temp = model.predict(test)
        result = pd.DataFrame(passengerId, columns=['PassengerId'])
        result['Survived'] = temp
        result = result.sort_values(by='PassengerId')


        return result
