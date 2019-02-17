import numpy as np
import scipy as sp
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("fights_all.csv")

print(data.info())
print(data.describe(include=['O']))
# print(data.head())


