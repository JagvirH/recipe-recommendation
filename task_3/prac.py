import csv
import pandas as pd


class Archive:
    def __init__(self, file):
        self.records = []
        
    def get_headers(self):
        return(self.headers)

    def add_rainfall(self,year,month,rainfall):
        for line in self.file():
            check=line.split(",")
            if check[0] == year and check[1] == month:
                print("yesssssssssss")

        return("yes")


f1=Archive("Armagh.csv")
h = f1.add_rainfall(1853,1,10000000)
