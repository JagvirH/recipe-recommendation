import csv
import pandas as pd 

class DataHandler(object):
    def __init__(self, datafile):
        self.df = pd.read_csv(datafile)
    def shape(self): # number of rows and columns
        return self.df.shape
    def first_rows(self):
        return self.df.head(3)
    def last_rows(self):
        return self.df.tail(3)
    
## Testing the code
try:
    table = DataHandler('Oxford.csv') 
    print(table.first_rows())
    print(table.last_rows())
except OSError:
    print('File Not Found')


