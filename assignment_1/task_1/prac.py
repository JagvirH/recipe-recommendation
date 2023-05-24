import csv
import pandas as pd

class Rainfall:
    def __init__(self, city, year, start_month, end_month):
        self.city = pd.read_csv(city)
        self.year = year
        self.start_month = start_month
        self.end_month = end_month

    def check_avrg_rainfall(self):
        avg = []
        with open(city,'r') as self.city:
        #print(self.city)
            for row in self.city:
                row = row.split(",")
                if row[0] == self.year:
                    if int(row[1]) >= self.start_month and  int(row[1]) <= self.end_month:
                        avg.append(float(row[5]))
        final = 0
        counter = 0
        for i in avg:
            final=final + i
            counter += 1
        ans = final/counter


        #return ("the average rainfall in " + city + "in the year " + self.year + " from the months of " + str(self.start_month) "to" + str(self.end_month) + "is " + ans)
        return (ans)
                
city, year, start_month, end_month = "Oxford.csv", '1941', 2,4
r1 = Rainfall(city,year,start_month,end_month)

print(r1.check_avrg_rainfall())