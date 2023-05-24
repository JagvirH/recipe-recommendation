import csv
import pandas as pd

class Rainfall:
    def __init__(self, city, year, start_month, end_month): #rain
        
        self.city = pd.read_csv(city+".cvs")
        self.year = year
        self.start_month = start_month
        self.end_month = end_month
        self.rain = 0


print("optionas are: (1) Calculate average rainfall")
choice = input("select the number of what you would like to check: ")

if choice == "1":
    #city = input("choose a city: ")
    #year = input("choose a year: ")
    #start_month = input("choose the start month (in number) : ")
    #end_month = input("choose the end month (in numbers) : ")
    city, year, start_month, end_month = "Aberporth", 1941, 2,4
    r1 = Rainfall(city,year,start_month,end_month)
    print(r1.city)
    
else:
    print("there is no option for that")


