#implement a class system
import csv
import pandas as pd

class Rainfall:
    def __init__(self, city, year, month, rain):
        self.city = city
        self.year = year
        self.month = month
        self.rain = rain
    
    def display():
        global dr
        which = str(input("Which city would you like to view?  "))
        if which[-4:] != ".csv":
            which = which + ".csv"
        dr = pd.read_csv(which)
        print(dr)

    def averageRainfall(): #calculated the average rainfall
        readen = pd.read_csv('Aberporth.csv')

    #Not yet implemented.

        # average = 0
        # sum = 0
        # row_count = 0
        # for row in readen:
        #     for column in row.split(','):
        #         n = float(column)
        #         sum += n
        #     row_count += 1
        # average = sum / len(column)
        # return average

    def displayCYA(): #outputs the City, Year, Average Rainfall for 12 months over the records
        pass
print('')
Rainfall.display()
print('')
print('')
#Rainfall.averageRainfall()
print('')

#provide operations for
#inserting, extracting, deleting and displaying information from the system with
#appropriate Exception handling.