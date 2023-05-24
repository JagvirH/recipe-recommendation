import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
import os


os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/lab9")
df = pd.read_csv('Adult_Census_Income.csv')

print("hellpo")

'''
#print(df.head(10))

##remove ? and spaces
df = df.replace(' ', 0)
df = df.replace('?', 0)


df["income"] = (df["income"].str.replace(" ", "")==">50K").astype(int)
#print(df.head(10))

df2 = df.select_dtypes(include=[np.number])
print(df2)
dataset = df2.values
print(dataset)


X = dataset[:,0:6]
y = dataset[:,6]
model = Sequential()
model.add(Dense(10, input_dim=6, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# compile the keras model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
predictions = model.predict_classes(X)
_, accuracy = model.evaluate(X, y)
print("Accuracy: %.2f" % (accuracy*100))
# summarize the first 5 cases
for i in range(5):
    print("%s => %d (expected %d)" % (X[i].tolist(), predictions[i], y[i]))#X = dataset[:,0:6]

'''