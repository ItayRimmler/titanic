import pandas as pd
from preprocess import SmartAge, MothersAndChildren
from rf import RandomForest

# Loading the data frames:
train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")

# Filling in the missing values:
train = SmartAge(train)
# test = SmartAge(test)

# Adding more columns that indicate if the person is a mother, how many children does it have, and how many are saved in the
# family:
train = MothersAndChildren(train)
# test = MothersAndChildren(test)


RandomForest(train)
# uh... well RandomForest(test) won't work because test doesn't have a Survived column...
