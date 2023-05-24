from pandas import Series
from pandas import DataFrame
import numpy as np
a = np.array([[1.0,2],[3,4]])
df = DataFrame(a)
df1 = DataFrame(np.array([[1,2],[3,4]]),columns=["a","b"])

df1["a"]

df1["a"][0]

print(df1.iloc[1])
'''
df1[["a", "b"]]
s1 = Series(np.arange(0.0,5))
s2 = Series(np.arange(1.0,2.0, 0.2))
df2 = DataFrame({"one": s1, "two": s2})
df2.describe() #computes basic statistics for each column
'''