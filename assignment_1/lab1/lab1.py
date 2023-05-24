import math
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt

'''
x = linspace(0,1,100)
y = x**3 - 4
fig,ax=plt.subplots()
ax.plot(x,y)
ax.plot(x,y,'-r',label='line')
leg=ax.legend();
plt.plot(y)
plt.show()

print(math.pi) # pi value
x = np.random.randn(100,100)
y = np.mean(x,0)
plt.plot(y)
plt.show()

'''





check_email = open("msg.txt", "r")

for line in check_email:
    print( line.find("@"))
        
    

