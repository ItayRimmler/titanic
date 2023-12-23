import pandas as pd


def SmartAge(data):

    notNans = data.loc[(data.Age % 1 == 0) | (data.Age % 1 < 1)]

    # The data which will replace NaN values (or data that helps us find data that replaces NaN values):
    minorMaleMeanAge = notNans.loc[
        (notNans.Age < 18) & (notNans.Sex == 'male'), 'Age'].mean()  # The average "Master"'s age

    youngFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Sex == 'female')].shape[0]
    youngAdultFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'female')].shape[0]
    minorFemaleCount = notNans.loc[(notNans.Age < 18) & (notNans.Sex == 'female')].shape[0]
    youngAdultFemaleProportion = youngAdultFemaleCount / youngFemaleCount
    minorFemaleProportion = minorFemaleCount / youngFemaleCount
    youngAdultMeanAge = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'female'), 'Age'].mean()
    minorFemaleMeanAge = notNans.loc[(notNans.Age < 18) & (notNans.Sex == 'female'), 'Age'].mean()
    youngFemaleMeanAge = minorFemaleMeanAge * minorFemaleProportion + youngAdultMeanAge * youngAdultFemaleProportion

    adultMaleMeanAge = notNans.loc[(notNans.Age >= 18) & (notNans.Sex == 'male'), 'Age'].mean()
    adultFemaleMeanAge = notNans.loc[(notNans.Age >= 18) & (notNans.Sex == 'female'), 'Age'].mean()

    # Changing the nan ages into the values we got:
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & data.Name.str.contains('Master'), 'Age'] = minorMaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & (data.Name.str.contains('Miss') | data.Name.str.contains('Ms.')), 'Age'] = youngFemaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & data.Name.str.contains('Mrs'), 'Age'] = adultFemaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & ~(data.Name.str.contains('Mrs')) & ~(data.Name.str.contains('Miss')) & ~(data.Name.str.contains('Master')), 'Age'] = adultMaleMeanAge

    return data

def MothersAndChildren(data):

    motherAge = 16
    maybeMother = ((data.Sex == 'female') & (data.Age >= motherAge))
    data['Mother'] = maybeMother
    temp1 = data.loc[data.Mother == True].Name
    temp1 = temp1.apply(lambda x:  x.split()[0] if x.split()[0][-1] == ',' else str(x.split()[0] + x.split()[1]))
    temp2 = data.loc[data.Mother == False].Name
    temp2[:] = temp2.apply(lambda x:  x.split()[0] if x.split()[0][-1] == ',' else str(x.split()[0] + x.split()[1]))
    temp = pd.concat([temp1,temp2], axis=0)
    temp = temp.sort_index()
    data['Ln'] = temp
    data = data.sort_values(by='Name')
    data.index = range(data.shape[0])
    data['Children'] = 0
    data['SWomenAndChildrenInFamily'] = 0
    data['SAllWomenAndChildrenInFamily'] = 0
    for i in range(data.shape[0]):
        if not data.loc[i, 'Mother']:
            continue
        h = 1
        done1 = False
        k = data.shift(0).loc[i, 'Ln']
        while not done1:
            new = data.shift(h).loc[i, 'Ln']
            if not k == new:
                done1 = True
            elif data.shift(0).loc[i, 'Age'] - data.shift(h).loc[i, 'Age'] > motherAge and data.shift(0).loc[i, 'Fare'] == data.shift(h).loc[i, 'Fare']:
                data.loc[i, 'Children'] += 1
                if data.shift(h).loc[i, 'Survived'] == 1:
                    data.loc[i, 'SWomenAndChildrenInFamily'] += 1
            h += 1
        if data.loc[i, 'Survived']:
            data.loc[i, 'SWomenAndChildrenInFamily'] += 1
        if data.loc[i, 'Survived'] and data.loc[i, 'Children'] + 1 == data.loc[i, 'SWomenAndChildrenInFamily']:
            data.loc[i, 'SAllWomenAndChildrenInFamily'] = 1
            for j in range(data.loc[i, 'Children']):
                data.shift(j + 1).loc[i, 'SAllWomenAndChildrenInFamily'] = 1
        else:
            data.loc[i, 'SAllWomenAndChildrenInFamily'] = 2
            for j in range(data.loc[i, 'Children']):
                data.shift(j + 1).loc[i, 'SAllWomenAndChildrenInFamily'] = 2
        #print(f"Survived (0/1): {data.loc[i, 'Survived']}, Children: {data.loc[i, 'Children']}, Survived women & children : {data.loc[i, 'SWomenAndChildrenInFamily']}, All women & children survived(1/2) : {data.loc[i, 'SAllWomenAndChildrenInFamily']}, Name : {data.shift(0).loc[i, 'Name']}")

    return data
