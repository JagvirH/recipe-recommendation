import pandas as pd
import csv
import os
from tracemalloc import start

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


    def insert_database(self, city, month, year):
        df = pd.read_csv(city)
        filtered_df = df[(df.iloc[:, 0] == year) & (df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        rainfall = filtered_df.iloc[0,5] #Index where the rainfall would be found in the csv file.
        with open('Database.csv', 'r') as check:
            for row in check:
                row = row.split(",")
                if str(row[0]) == str(city[:-4]) and int(row[1]) == int(year) and int(row[2]) == int(month):
                    return "The data already exists"
        if rainfall is not None:
            with open('Database.csv', 'a') as db:
                print([city[:-4], year, month, rainfall])
                wr = csv.writer(db)
                wr.writerow([city[:-4], year, month, rainfall])
                return "The data has been entered"
        else:
            print('Data not found')

    def delete(self,city, year, month):
        self.df = self.df[~((self.df['City'] == city[:-4]) & (self.df['Year'] == year) & (self.df['Month'] == month))]
        #if self.df.empty:
            #return None
        self.df.to_csv("Database.csv", index=False)
        #return "the data has been deleted"

    def sma(self,city,year1,year2,k):
        df = pd.read_csv(city)
        data = df[(df.iloc[:, 0] >= year1) & (df.iloc[:, 0] <= year2)]
        #print(data)
        # Compute the moving average
        data.loc[:, 'moving_avg'] = data['rain'].rolling(int(k)).mean()
        return data[['yyyy', 'mm', 'moving_avg']]

    

        
os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/task_3")

city = str(input("Which file would you like to access?  ")) 
if city[-4:] != ".csv":
    city = city + ".csv"

data = Archive(city)

print('Please select one of the following options: \n [1] Getting Rainfall \n [2] Inserting data into Database \n [3] Deleting from the database \n [4] calculate sma ')
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
    print(data.insert_database(city, month, year))
    #print('New data has been added to Database.csv')

elif option == 3:
    year = int(input("Enter the year: "))
    month = int(input("Enter the month: "))
    print(data.delete(city, month, year))

elif option == 4:
    year1 = int(input("Enter the year: "))
    year2 = int(input("Enter the year: "))
    k = input("Enter the number of months: ")
    print(data.sma(city,year1,year2,k))
    
    
