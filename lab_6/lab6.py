import os
import csv
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/lab_6")
'''
df = pd.read_csv("store_data.csv")

column_names = df.columns.tolist()

#df = df.fillna(0)

# Print the first item in the list, which should be the title
#print(column_names)
#print(df.head(10))

#df= df.head(10)
#list_of_lists = df.values.tolist()

# Print the resulting list of lists

print(column_names)
for row in list_of_lists:
    print(row)

df = pd.read_csv('store_data.csv', header=None)

#df = df.fillna(0)

records = df.values.tolist()

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

'''
df = pd.read_csv('store_data.csv', header=None)

df.isnull().sum()

df.dropna(inplace=True)  # Drop rows with missing values
df.dropna(axis=1, inplace=True)  # Drop columns with missing values
df = df.astype(str)

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
df = pd.DataFrame(te_ary, columns=te.columns_)
