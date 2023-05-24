import numpy as np
from pandas import Series
s1 = Series([0.1, 1.2, 2.3, 3.4, 4.5])
a = np.array([0.1, 1.2, 2.3, 3.4, 4.5])
s2 = Series(a) # NumPy array to Series
#s3 = Series([0.1, 1.2, 2.3, 3.4, 4.5], index = [’a’,’b’,’c’,’d’,’e’]) 
s3 = Series([0.1, 1.2, 2.3, 3.4, 4.5], index = ["a", "b", "c", "d", "e"])
s3["a"]
s3[0]
s3[["a","c"]]
s3[s3>2]
s3.index
s3.values
s3.describe() #computes s3 statistics

print(s3.describe())
