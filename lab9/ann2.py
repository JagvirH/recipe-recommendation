import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("Adult_Census_Income.csv")

##remove ? and spaces
df.replace(regex=r'\?', value=np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
df.replace(regex=r'^\s', value='', inplace=True)

'''
def income_cond_hists(df, plot_cols, grid_col):
	for col in plot_cols:
		g = sns.FacetGrid(df, col=grid_col, margin_titles=True)
		g.map(plt.hist, col)
		plt.show()
## plotting		
income_cond_hists (df, df.select_dtypes(include=[np.number]).columns, "income")

def income_boxplot(df):
	for col in df.select_dtypes(include=[np.number]).columns:
		fig = plt.figure(figsize=(6, 6))
		fig.clf()
		ax = fig.gca()
		df.boxplot(column=[col], ax=ax, by=['income'])
		plt.show()
	return('Done')
income_boxplot(df)
'''

#print(df.head(10))
df['income'] = (df['income'].str.replace(' ', '')=='>50K').astype(int)
#print(df.head(10))

df = df.select_dtypes(include=[np.number])

print(df)
'''
X = df.iloc[:,0:6]
y = df.iloc[:,6]
model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
predictions = model.predict(X)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
# summarize the first 5 cases
for i in range(5):
	print('predicted %d (actual %d)' % (predictions[i], y[i]))
'''