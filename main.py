import pandas as pd
from preprocess import SmartAge, SmartFamily, SmartCabinTicketsAndEmbarked, SmartTitle
from rf import RandomForest

# Loading the data frames:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Filling in the missing values:
train = SmartAge(train)
test = SmartAge(test)

# Adding more columns that indicate if the person is a mother, how many children does it have, and how many are saved in the
# family:
train = SmartFamily(train)
test = SmartFamily(test)

train = SmartTitle(train)
test = SmartTitle(test)


# Replacing char Embarked values with numbers and analysing the ticket numbers:
train = SmartCabinTicketsAndEmbarked(train)
test = SmartCabinTicketsAndEmbarked(test)

# Now we do what we came to do:
result = RandomForest(train, test)
result.to_csv('output.csv', index=False)
