import pandas as pd
import numpy as np
from exceptions import tooSmall

# The following function replaces our NaN values in the Age column into actual values in a smart way:
def SmartAge(data):

    # We start by saving in a variable the rows that have a non-NaN value at the Age column, which we'll work with:
    notNans = data.loc[(data.Age % 1 == 0) | (data.Age % 1 < 1)]

    #     The data which will replace NaN values (or data that helps us find data that replaces NaN values):

    # We calculate the mean of the Age column of minor males only. That will replace NaN ages for Master Anyname:
    minorMaleMeanAge = notNans.loc[(notNans.Age < 14) & (notNans.Sex == 'male'), 'Age'].mean()

    # Miss Anyname is likely either a female young adult or a female minor. Therefore, we will start by calculating the mean of
    # female young adults, and female minors:
    femaleYoungAdultMeanAge = notNans.loc[(notNans.Age < 27) & (notNans.Age > 14) & (notNans.Sex == 'female'), 'Age'].mean()
    minorFemaleMeanAge = notNans.loc[(notNans.Age < 15) & (notNans.Sex == 'female'), 'Age'].mean()

    # We count the amount of young females, the amount of young adult females, and minor females:
    youngFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Sex == 'female')].shape[0]
    youngAdultFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Age > 14) & (notNans.Sex == 'female')].shape[0]
    minorFemaleCount = notNans.loc[(notNans.Age < 15) & (notNans.Sex == 'female')].shape[0]

    # We calculate the proportion of minor and young adults out of all the young females:
    youngAdultFemaleProportion = youngAdultFemaleCount / youngFemaleCount
    minorFemaleProportion = minorFemaleCount / youngFemaleCount

    # We finish by calculating the weighted mean of young females:
    youngFemaleMeanAge = minorFemaleMeanAge * minorFemaleProportion + femaleYoungAdultMeanAge * youngAdultFemaleProportion

    # Mr. Anyname is likely either a male young adult or a male adult. Therefore, we will start by calculating the mean of
    # male young adults, and male older adults:
    maleYoungAdultMeanAge = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'male'), 'Age'].mean()
    olderAdultMaleMeanAge = notNans.loc[(notNans.Age > 26) & (notNans.Sex == 'male'), 'Age'].mean()

    # We count the amount of adult males, the amount of young adult males, and older adult females:
    adultMaleCount = notNans.loc[(notNans.Age >= 15) & (notNans.Sex == 'male')].shape[0]
    youngAdultMaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Age > 14) & (notNans.Sex == 'male')].shape[0]
    olderAdultMaleCount = notNans.loc[(notNans.Age > 27) & (notNans.Sex == 'male')].shape[0]

    # We calculate the proportion of minor and young adults out of all the young females:
    youngAdultMaleProportion = youngAdultMaleCount / adultMaleCount
    olderMaleProportion = olderAdultMaleCount / olderAdultMaleCount

    # We finish by calculating the weighted mean of adult males:
    adultMaleMeanAge = olderMaleProportion * olderAdultMaleMeanAge + youngAdultMaleProportion * maleYoungAdultMeanAge

    # As for Mrs. Anyname... We assume it is likely an older woman:
    adultFemaleMeanAge = notNans.loc[(notNans.Age >= 27) & (notNans.Sex == 'female'), 'Age'].mean()

    # Changing the NaN ages into the values we got:
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & data.Name.str.contains('Master'), 'Age'] = minorMaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & (data.Name.str.contains('Miss') | data.Name.str.contains('Ms.')), 'Age'] = youngFemaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & data.Name.str.contains('Mrs'), 'Age'] = adultFemaleMeanAge
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & ~(data.Name.str.contains('Mrs')) & ~(data.Name.str.contains('Miss') | data.Name.str.contains('Ms.')) & ~(data.Name.str.contains('Master')), 'Age'] = adultMaleMeanAge

    return data

# The following function returns the data with additional columns: boolean Mother column, Children column,
# SurvivedWomenAndChildrenInFamily column, and SurvivedAllWomenAndChildrenInFamily column:
def MothersAndChildren(data, key=None):

    # CONSTANTS:
    try:
        MOTHER_AGE = 18

        MOTHER_AGE = round(MOTHER_AGE)

        exceptionz = []
        exception1 = 15 < MOTHER_AGE
        exceptionz.append(exception1)
        exceptionz = np.array(exceptionz)

        # if not exceptionz.any():
        #     raise tooSmall(name='ALL')
        if (exceptionz == False).sum() >= 2:
            raise tooSmall(name='ANY')
        if not exception1:
            raise tooSmall(name='MOTHER_AGE',a=15, b=MOTHER_AGE)

    except tooSmall as e:
        if e.name == 'ALL':
            MOTHER_AGE = e.b
        if e.name == 'MOTHER_AGE':
            MOTHER_AGE = e.b
        if e.name == 'ANY':
            exit(1)

    finally:

        if key:
            data['Survived'] = 0
        # We start off by finding which passenger might be a mother:
        maybeMother = ((data.Sex == 'female') & (data.Age >= MOTHER_AGE))

        # We add the Mother column to our data:
        data['Mother'] = maybeMother

        # We get the mothers' names:
        motherNames = data.loc[data.Mother == True].Name

        # We get only their last names. In case of a two word last name (like van Gogh) we get both the 'van' and the 'Gogh':
        motherNames = motherNames.apply(lambda x:  x.split()[0] if x.split()[0][-1] == ',' else str(x.split()[0] + x.split()[1]))

        # We get the non-mothers' names:
        nonMotherNames = data.loc[data.Mother == False].Name

        # We get only their last names. In case of a two word last name (like van Gogh) we get both the 'van' and the 'Gogh':
        nonMotherNames = nonMotherNames.apply(lambda x:  x.split()[0] if x.split()[0][-1] == ',' else str(x.split()[0] + x.split()[1]))

        # We temporarily concatenate the last names we obtained:
        temp = pd.concat([motherNames, nonMotherNames], axis=0)

        # We sort them by index, so it will match to the rows in data:
        temp = temp.sort_index()

        # We add it to data as a column:
        data['Ln'] = temp

        # We sort our data by name for later usage:
        data = data.sort_values(by='Name')

        # We replace the older, unsorted by name index, in a new, sorted by name index, so we could iterate correctly in a later
        # to come loop:
        data.index = range(data.shape[0])

        # We add the Children column to our data:
        data['Children'] = 0

        # We add the SurvivedWomenAndChildrenInFamily column to our data:
        data['SurvivedWomenAndChildrenInFamily'] = 0

        # We add the SurvivedAllWomenAndChildrenInFamily column to our data:
        data['SurvivedAllWomenAndChildrenInFamily'] = 0

        data['Papa'] = 0

        # We now iterate over the data:
        for i in range(data.shape[0]):

            papa = 'Mr' in data.loc[i, 'Name']

            # We skip on all the non-maybe-mothers:
            if not data.loc[i, 'Mother'] and not papa:
                continue

            #     We check Mrs. Anyname, and the rows above her with the same last name. They might be her children. Rows below
            # Mrs. Anyname aren't interesting because sorting by name assures us that Mrs. Anyname will be below Mr. Anyname,
            # Miss Anyname and Master Anyname.

            # Flags initialization:
            shiftAmount = 1
            childCounter = 0
            done = False
            anyname = data.shift(0).loc[i, 'Ln']

            # The loop:
            while not done:

                # We get the shifted's name:
                shifted = data.shift(shiftAmount).loc[i, 'Ln']

                # If it isn't Anyname:
                if not anyname == shifted:
                    done = True

                # If it is, we need it to be of a person that: may be a child of Mrs. Anyname and paid the same fare as her:
                elif data.shift(0).loc[i, 'Age'] - data.shift(shiftAmount).loc[i, 'Age'] > MOTHER_AGE and data.shift(0).loc[i, 'Fare'] == data.shift(shiftAmount).loc[i, 'Fare']:
                    if not papa:
                        data.loc[i, 'Children'] += 1
                    else:
                        childCounter += 1

                    # And if it is a child, we check if it survived, and update the mother's SurvivedWomenAndChildrenInFamily:
                    if data.shift(shiftAmount).loc[i, 'Survived'] == 1:
                        data.loc[i, 'SurvivedWomenAndChildrenInFamily'] += 1

                # Finally for the while loop, we update the shift amount:
                shiftAmount += 1

            # We increment the SurvivedWomenAndChildrenInFamily if the (possible) mother survived:
            if data.loc[i, 'Survived']:
                data.loc[i, 'SurvivedWomenAndChildrenInFamily'] += 1

            # If the survived amount of children (+ the possibly survived mother) equals to SurvivedWomenAndChildrenInFamily:
            if data.loc[i, 'Survived'] and data.loc[i, 'Children'] + data.loc[i, 'Survived'] == data.loc[i, 'SurvivedWomenAndChildrenInFamily']:

                # Then we can set SurvivedAllWomenAndChildrenInFamily for the whole portion of women and children to the family
                # to True:
                data.loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 1
                for j in range(data.loc[i, 'Children']):
                    data.shift(j + 1).loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 1
                if papa:
                    for j in range(childCounter):
                        data.shift(j + 1).loc[i, 'Papa'] = 1
            # If it's not equal:
            else:

                # Then we can set SurvivedAllWomenAndChildrenInFamily for the whole portion of women and children to the family
                # to False:
                data.loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 2
                for j in range(data.loc[i, 'Children']):
                    data.shift(j + 1).loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 2
                if papa:
                    for j in range(childCounter):
                        data.shift(j + 1).loc[i, 'Papa'] = 1

        temp = pd.get_dummies(data['Ln'], prefix='Ln')
        data = pd.concat([data, temp], axis=1)
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                if not i == j and data.loc[i, 'Ln'] == data.loc[j, 'Ln'] and data.loc[i, 'Fare'] == data.loc[j, 'Fare']:
                    data.loc[i, 'Parch'] = data.loc[:, 'Ln_' + data.loc[i, 'Ln']].sum()

        return data

def TicketsAndEmbarked(data):

    #     Giving the embarked column a score instead of a char:
    embarkedScore = pd.get_dummies(data['Embarked'], prefix='Embarked')
    embarkedScore.columns = ['EmbarkedQ', 'EmbarkedC', 'EmbarkedS']
    data = pd.concat([data, embarkedScore], axis=1)

    #      Now we want to save the initial 3 digits\characters of the tickets:

    # We get the initial 3 characters of our data:
    data['TicketInitials'] = data['Ticket'].str[:]

    # We create one-hot-coded dummy columns:
    initials = data['TicketInitials'].str.get_dummies()

    # We concatenate the new dummy columns with the original data frame:
    data = pd.concat([data, initials], axis=1)

    return data

def convertToNum(char):
    if char == 'A':
        return 11100
    if char == 'C':
        return 22200
    if char == '.':
        return 3001
    if char == '/':
        return 4001
    if char == 'P':
        return 50050
    if char == 'W':
        return 66600
    if char == 'a':
        return 7000
    if char == 'O':
        return 5500
    if char == 'T':
        return 9000
    if char == 'E':
        return 8800
    if char == 'F':
        return 70000
    if char == 'L':
        return 80000
    if char == 'I':
        return 900
    if char == 'S':
        return 40000
    else:
        return 100000
