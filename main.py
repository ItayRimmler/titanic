# I DON'T REALLY KNOW WHICH LAYERS TO PUT IN EACH PLACE NECESSARILY, I USUALLY READ ABOUT IT BEFORE BUT I TEND
# TO FORGET, AND I WANT TO PROGRESS ALREADY.
# I WANT TO LEARN FROM KAGGLE, ALSO TO UNDERSTAND THE CAKE'S CHIMESTRY, ALSO PRACTICE EVERYTHING, INCLUDING SKLEARN
# FORGETTING CONSTANTLY THE PIPELINE OF WORK IN NN
# I WANT TO ADVANCE WITH MY PROJECT
# I WANT TO BEGIN WITH THE TITANIC
# I CONSTANTLY CHECK THINGS WITH CHATGPT AND DON'T LEARN
# I FORGOT A LOT FROM MY LAST PROJECT, AND NOW I JUST COPY AND PASTE THINGS FROM THERE, AS IF I DIDN'T LEARN
# ANYTHING
# PROBABLY FORGOT MORE STUFF... I'LL CONTINUE WRITING HERE WHEN I'LL BE LESS TIRED

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
