import pandas as pd
import csv
import os

class Archive:
    def __init__(self, city): # setting the class to make sure the database has headings
        self.df = pd.read_csv(city)

        headers = ['City', 'Year', 'Month', 'Rainfall']
        with open ("Database.csv", 'a') as db:
            file_is_empty = os.stat('Database.csv').st_size == 0
            writer = csv.writer(db)
            if file_is_empty:
                writer.writerow(headers)
                print('Headers added')
        self.df = pd.read_csv("Database.csv")


    def get_rainfall(self, city, month, year ):
        df = pd.read_csv(city)
        filtered_df = df[(df.iloc[:, 0] == year) & (df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        rainfall = filtered_df.iloc[0,5] #Index where the rainfall would be found in the csv file.
        return rainfall

    def insert_database(self, city, month, year):
        rainfall = self.get_rainfall(city, month, year)
        if rainfall is not None:
            with open('Database.csv', 'a') as db:
                wr = csv.writer(db)
                wr.writerow([city[:-4], year, month, rainfall])
        else:
            print('Data not found')
    
    def delete(self,city, year, month):
        self.df = self.df[~((self.df.iloc[:,0] == city[:-4]) & (self.df.iloc[:, 1] == year) & (self.df.iloc[:, 2] == month))]
        self.df.to_csv(('Database.csv'), index=False)
        
os.chdir("/Users/filip/Documents/Year 2/Semester 2/CO2106/Assignment 1")

city = str(input("Which file would you like to access?  ")) 
if city[-4:] != ".csv":
    city = city + ".csv"


data = Archive(city)

print('Please select one of the following options: \n [1] Getting Rainfall \n [2] Inserting Rainfall into Database \n [3] Deleting ')
option = int(input(''))

    #Option 1 fetches the rainfall using the class function get_rainfall
if option == 1:
    year = int(input("Enter the year: "))
    month = int(input("Enter the month: "))
    rainfall = data.get_rainfall(city, month, year)
    print(f"Rainfall: {rainfall}")

elif option == 2:
    year = int(input("Enter the year: "))
    month = int(input("Enter the month: "))
    data.insert_database(city, month, year)
    print('New data has been added to Database.csv')

elif option == 3:
    year = int(input("Enter the year: "))
    month = int(input("Enter the month: "))
    data.delete(city, year, month)
    print('Data has been deleted from Database.csv')

    
