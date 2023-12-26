import pandas as pd
from preprocess import SmartAge, MothersAndChildren, TicketsAndEmbarked
from rf import RandomForest

# Loading the data frames:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Filling in the missing values:
train = SmartAge(train)
test = SmartAge(test)

# Adding more columns that indicate if the person is a mother, how many children does it have, and how many are saved in the
# family:
train = MothersAndChildren(train)
test = MothersAndChildren(test, True)

# Replacing char Embarked values with numbers and analysing the ticket numbers:
train = TicketsAndEmbarked(train)
test = TicketsAndEmbarked(test)

# Now we do what we came to do:
result = RandomForest(train, test)
result.to_csv('output.csv', index=False)
