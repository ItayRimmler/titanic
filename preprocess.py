import pandas as pd
import numpy as np
import torch

from exceptions import tooSmall

# The following function replaces our NaN values in the Age column into actual values in a smart way. It also one-hot-encode any
# children and babies:
def SmartAge(data):

    #    Replacing NaN values:

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

    #    One-hot-encoding Children, Elders and Babies:

    # Such a simple line of code:
    data['Baby'] = data.Age < 5
    data['Child'] = (5 <= data.Age) & (data.Age < 16)
    data['Elder'] = 60 <= data.Age

    return data

# The following function returns the data with additional columns: boolean isMother column, Children column,
# hasMother column, hasFather column, and :
def SmartFamily(data):

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

        # We start off by finding which passenger might be a mother:
        maybeMother = ((data.Sex == 'female') & (data.Age >= MOTHER_AGE))

        # We add the Mother column to our data:
        data['isMother'] = maybeMother

        # We get the mothers' names:
        motherNames = data.loc[data.isMother == True].Name

        # We get only their last names. In case of a two word last name (like van Gogh) we get both the 'van' and the 'Gogh':
        motherNames = motherNames.apply(lambda x:  x.split()[0] if x.split()[0][-1] == ',' else str(x.split()[0] + x.split()[1]))

        # We get the non-mothers' names:
        nonMotherNames = data.loc[data.isMother == False].Name

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

        # We add the hasMother column to our data:
        data['hasMother'] = 0

        # We add the hasFather column to our data:
        data['hasFather'] = 0

        # We now iterate over the data:
        for i in range(data.shape[0]):

            # The papa flag tells us if we are currently iterating over a potential father (above the age of 60 is probably
            # either a grandfather, a father to adults, or just not someone that can improve survival chances overall)
            papa = ('Mr' in data.loc[i, 'Name']) and data.loc[i, 'Age'] < 60

            # We skip on all the non-maybe-mothers and non-maybe-fathers:
            if not data.loc[i, 'isMother'] and not papa:
                continue

            #     We check Mrs. Anyname, and the rows above her with the same last name. They might be her children. Rows below
            # Mrs. Anyname aren't interesting because sorting by name assures us that Mrs. Anyname will be below Mr. Anyname,
            # Miss Anyname and Master Anyname. However, Mr.Anyname could be above one of his children.

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

                # If it is, we need it to be of a person that: may be a child of Mrs. or Mr. Anyname and paid the same fare as them:
                elif data.shift(0).loc[i, 'Age'] - data.shift(shiftAmount).loc[i, 'Age'] > MOTHER_AGE and data.shift(0).loc[i, 'Fare'] == data.shift(shiftAmount).loc[i, 'Fare']:
                    if not papa:
                        data.loc[i, 'Children'] += 1
                        data.loc[i - shiftAmount, 'hasMother'] = 1
                    else:
                        data.loc[i, 'Children'] += 1
                        data.loc[i - shiftAmount, 'hasFather'] = 1

                # Finally for the while loop, we update the shift amount:
                shiftAmount += 1

                # Flags initialization:
                shiftAmount = -1
                done = False

                # The loop:
                while not done:

                    # We get the shifted's name:
                    shifted = data.shift(shiftAmount).loc[i, 'Ln']

                    # If it isn't Anyname:
                    if not anyname == shifted:
                        done = True

                    # If it is, we need it to be of a person that: may be a child of Mr. Anyname and paid the same fare as him:
                    elif data.shift(0).loc[i, 'Age'] - data.shift(shiftAmount).loc[i, 'Age'] > MOTHER_AGE and data.shift(0).loc[i, 'Fare'] == data.shift(shiftAmount).loc[i, 'Fare']:
                        data.loc[i, 'Children'] += 1
                        data.loc[i - shiftAmount, 'hasFather'] = 1

                    # Finally for the while loop, we update the shift amount:
                    shiftAmount -= 1

        # We also want to create an Alone flag:
        data['Alone'] = 0

        # We now turn the Parch column to a column that tells us how many family members there are. We turn the SibSp to a flag that
        # tells us if the person has both parents. We do it with a nested loop:
        for i in range(data.shape[0]):
            flag = 1
            if data.loc[i,'hasMother']  == 1 and data.loc[i,'hasFather']  == 1:
                data.loc[i,'SibSp'] = 1
            for j in range(data.shape[0]):
                if not i == j and data.loc[i, 'Ln'] == data.loc[j, 'Ln'] and data.loc[i, 'Fare'] == data.loc[j, 'Fare']:
                    flag += 1
            data.loc[i, 'Parch'] = flag
            if flag == 1:
                data.loc[i,'Alone'] = 1

        # In 2 lines, we one-hot-encode all the last names
        temp = pd.get_dummies(data['Ln'], prefix='Ln')
        data = pd.concat([data, temp], axis=1)

        return data

# Replaces the cabin, tickets and embarked columns in a smart way:
def SmartCabinTicketsAndEmbarked(data):

    #     Giving the embarked column a score instead of a char:
    embarkedScore = pd.get_dummies(data['Embarked'], prefix='Embarked')
    embarkedScore.columns = ['EmbarkedQ', 'EmbarkedC', 'EmbarkedS']
    data = pd.concat([data, embarkedScore], axis=1)

    #      Now we want to save the all but the final 2 digits\characters of the tickets, also the first character of our cabin:

    # We get all but the final 2 characters of our data, also the first character of our cabin:
    data['TicketInitials'] = data['Ticket'].str[:-2]
    data['Cabin'] = data['Cabin'].str[0]

    # We create one-hot-coded dummy columns:
    initials = data['TicketInitials'].str.get_dummies()
    initials1 = data['Cabin'].str.get_dummies()

    # We concatenate the new dummy columns with the original data frame:
    data = pd.concat([data, initials, initials1], axis=1)

    # We check who didn't pay for a ticket (probably someone with connections or a crewmate, both implications are major):
    data['Free'] = (data.Fare == 0) | (data.Fare == 'nan')

    return data

# This function gives us one-hot-encoded Pclasses, and adds a column with a flag that says whether we have a rare title:
def SmartTitle(data):

    # ADD A COMMENT
    rares = ['Countess', 'Don', 'Jonkheer']
    data['RareTitles'] = rares[0] in data.Name or rares[1] in data.Name or rares[2] in data.Name

    # ADD A COMMENT
    temp = pd.get_dummies(data['Pclass'], prefix='Pclass')
    data = pd.concat([data, temp], axis=1)
    return data
