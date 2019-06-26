import pandas as pd
import numpy as np
import re
import csv


from collections import Counter


df = pd.read_csv('reviews.csv') #to read the csv file data and store in df 

msk = np.random.rand(len(df)) <= 0.8 #random data division of dataset

train = df[msk] #80 percent training data 
test = df[~msk] #reamaining 20% not included in msk for test data
train.to_csv('train.csv', index=False, header=None) #conversion to separate csv files
test.to_csv('test.csv', index=False, header=None)