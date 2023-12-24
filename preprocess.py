import pandas as pd
import numpy as np
from exceptions import tooSmall

# The following function replaces our NaN values in the Age column into actual values in a smart way:
def SmartAge(data):

    # We start by saving in a variable the rows that have a non-NaN value at the Age column, which we'll work with:
    notNans = data.loc[(data.Age % 1 == 0) | (data.Age % 1 < 1)]

    #     The data which will replace NaN values (or data that helps us find data that replaces NaN values):

    # We calculate the mean of the Age column of minor males only. That will replace NaN ages for Master Anyname:
    minorMaleMeanAge = notNans.loc[(notNans.Age < 18) & (notNans.Sex == 'male'), 'Age'].mean()

    # Miss Anyname is likely either a female young adult or a female minor. Therefore, we will start by calculating the mean of
    # female young adults, and female minors:
    femaleYoungAdultMeanAge = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'female'), 'Age'].mean()
    minorFemaleMeanAge = notNans.loc[(notNans.Age < 18) & (notNans.Sex == 'female'), 'Age'].mean()

    # We count the amount of young females, the amount of young adult females, and minor females:
    youngFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Sex == 'female')].shape[0]
    youngAdultFemaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'female')].shape[0]
    minorFemaleCount = notNans.loc[(notNans.Age < 18) & (notNans.Sex == 'female')].shape[0]

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
    adultMaleCount = notNans.loc[(notNans.Age >= 18) & (notNans.Sex == 'male')].shape[0]
    youngAdultMaleCount = notNans.loc[(notNans.Age < 27) & (notNans.Age > 17) & (notNans.Sex == 'male')].shape[0]
    olderAdultMaleCount = notNans.loc[(notNans.Age > 26) & (notNans.Sex == 'male')].shape[0]

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
    data.loc[(data.Age % 1 != 0) & ~(data.Age % 1 < 1) & ~(data.Name.str.contains('Mrs')) & ~(data.Name.str.contains('Miss')) & ~(data.Name.str.contains('Master')), 'Age'] = adultMaleMeanAge

    return data

# The following function returns the data with additional columns: boolean Mother column, Children column,
# SurvivedWomenAndChildrenInFamily column, and SurvivedAllWomenAndChildrenInFamily column:
def MothersAndChildren(data):

    # CONSTANTS:
    try:
        MOTHER_AGE = 16

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
            raise tooSmall(name='MOTHER_AGE', b=MOTHER_AGE)

    except tooSmall as e:
        if e.name == 'ALL':
            MOTHER_AGE = e.b
        if e.name == 'MOTHER_AGE':
            MOTHER_AGE = e.b
        if e.name == 'ANY':
            exit(1)

    finally:

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

        # We add the SurvivedAllWomenAndChildrenInFamily column to our data (0 = irrelevant (for any non potentially a mother),
        # 1 = True, 2 = False):
        data['SurvivedAllWomenAndChildrenInFamily'] = 0

        # We now iterate over the data:
        for i in range(data.shape[0]):

            # We skip on all the non-maybe-mothers:
            if not data.loc[i, 'Mother']:
                continue

            #     We check Mrs. Anyname, and the rows above her with the same last name. They might be her children. Rows below
            # Mrs. Anyname aren't interesting because sorting by name assures us that Mrs. Anyname will be below Mr. Anyname,
            # Miss Anyname and Master Anyname.

            # Flags initialization:
            shiftAmount = 1
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
                    data.loc[i, 'Children'] += 1

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
            # If it's not equal:
            else:

                # Then we can set SurvivedAllWomenAndChildrenInFamily for the whole portion of women and children to the family
                # to False:
                data.loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 2
                for j in range(data.loc[i, 'Children']):
                    data.shift(j + 1).loc[i, 'SurvivedAllWomenAndChildrenInFamily'] = 2

        return data
