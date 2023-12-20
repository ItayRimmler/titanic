import pandas as pd
from classOnly import ClassOnly
from classAndAge import ClassAndAge
from familiesAndSmartAge import FamiliesAndSmartAge

# Loading the data frame:
train = pd.read_csv("train.csv")

#FamiliesAndSmartAge(train)

# Getting only what's important for the following functions:
train = train.set_index("Name")
train = train.set_index("PassengerId")

# Now, I don't know yet how to train a model using classed data and continuous data... I don't know a lot of
# things, so what I will do is divide my task into small batches:

# Let's start with analysing only according to class:
ClassOnly(train)
