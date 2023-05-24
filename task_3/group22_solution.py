from tracemalloc import start
import pandas as pd #Pandas will be used to access and make changes to the csv files.
import os #Makes it easier to navigate directories.
import csv

#Creating a class will help keep the methods in one place and managing operations. Reading the csv file with the use of variable.
#Each function goes back to 4 tasks and are used multiple times.
#get_rainfall is called when the year and month are input and the rainfall figure has to be found.
#del_value deletes the data from the row when year and month are given.
#insert_value creates a new row in which users values are inserted. 
#insert_quater asks the user for the season and user inputs the rainfall for that seasons months.
#I use iloc to seach the columns via index instead of headers since this works with all, as Oxford doesnt have headers.
class RainfallData:
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def get_average(self,year,start_month,end_month): #Gets the average of every rainfall for the given months
        avg=[]
        with open(file,'r') as self.df: #Goes through the data to find the values
            for row in self.df:
                row = row.split(",")
                if row[0] == str(year):
                    #print("Yes")
                    if row[1] >= start_month and row[1] <= str(end_month):
                        avg.append(float(row[5])) #appends all the needed values to a list called avg
        final = 0
        counter = 0
        for i in avg: # this calculates all the given values to find the average
            final=final + i
            counter += 1
        ans = final/counter
        ans = round(float(ans))
        return "the average rainfall in the year " + year + " from the months of " + str(start_month) + " to " + str(end_month) + " is " + str(ans) + " rounded to the nearest whole number. "
        
    def get_rainfall(self, year, month):
        filtered_df = self.df[(self.df.iloc[:, 0] == year) & (self.df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        rainfall = filtered_df.iloc[0,5] #Index where the rainfall would be found in the csv file.
        return rainfall
    
    def del_value(self, month, year, file):
        self.df = self.df[~((self.df.iloc[:, 0] == year) & (self.df.iloc[:, 1] == month))]
        self.df.to_csv(file, index=False) #Saves the csv file
    
    def insert_value(self, year, month, rainfall):
       
        d = {
            0: year,
            1: month,
            2: "---",
            3: "---",
            4: "---",
            5: rainfall,
            6: "---"
        }
        temp_frame = pd.DataFrame(d, index=[0])
        temp_frame.to_csv(file, mode = "a", index = False, header=False)

    def insert_quater(self, year, quarter, file):
        quarters = {
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
            "winter": [12, 1, 2]
        }

        months = quarters.get(quarter.lower(), [])
        if not months:
            print('Invalid Season Name.')
            return

        for month in months: #Loops and deletes so that it can be replaced in the next loop with the new data.
            self.del_value(month, year, file)
        
        for month in months: #Loops through the months and asks the user for new rainfall input.
            rainfall = float(input(f'Enter the rainfall for month {month}: '))
            self.insert_value(year, month, rainfall)


class Archive:

    def __init__(self,city): # setting the class to make sure the database has headings
        self.df = pd.read_csv(city)

        headers = ['City', 'Year', 'Month', 'Rainfall'] #makes sure a database csv file exists and has headers
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

    def insert(self, city, month, year):
        rainfall = self.get_rainfall(city, month, year)
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

    def delete(self, city, year, month):
        self.df = self.df[~((self.df.iloc[:,0] == city[:-4]) & (self.df.iloc[:, 1] == year) & (self.df.iloc[:, 2] == month))]
        self.df.to_csv(('Database.csv'), index=False)


    def sma(self,city,year1,year2,k):
        df = pd.read_csv(city)
        data = df[(df.iloc[:, 0] >= year1) & (df.iloc[:, 0] <= year2)]
        #print(data)
        # Compute the moving average
        data.loc[:, 'moving_avg'] = data['rain'].rolling(int(k)).mean()
        return data[[df.iloc[:,0], 'mm', 'moving_avg']]


os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/task_3")
#Menu system, user can choose what to do include exit.
run = True
while run:

    print('Would you like to either \n [1] Use Existing files \n [2] Archive data ')
    choice = int(input('Enter: '))

    if choice == 2:

        city = str(input("Which file would you like to access?  ")) 
        if city[-4:] != ".csv":
            city = city + ".csv"
        data = Archive(city)

        print('Please select one of the following options: \n [1] Inserting data into Database \n [2] Deleting from the database \n [3] calculate sma ')
        option = int(input(''))


        if option == 1:
            year = int(input("Enter the year: "))
            month = int(input("Enter the month: "))
            print(data.insert(city, month, year))
            #print('New data has been added to Database.csv')

        elif option == 2:
            year = int(input("Enter the year: "))
            month = int(input("Enter the month: "))
            data.delete(city, year, month)
            print('Deleted')

        elif option == 3:
            year1 = int(input("Enter the year: "))
            year2 = int(input("Enter the year: "))
            k = input("Enter the number of months: ")
            print(data.sma(city,year1,year2,k))
        
        else:
            print('Invalid option selected.')


    elif choice == 1:

        #Enter the file name, user can enter with 'csv' or without 'csv'.
        file = str(input("Which file would you like to access?  ")) 
        if file[-4:] != ".csv":
            file = file + ".csv"
        data = RainfallData(file)  
        
    
        print('Please select one of the following options: \n [0] Finding average \n [1] Finding Rainfall \n [2] Deleting Rainfall \n [3] Inserting New Rainfall Data \n [4] Inserting Data for Quarter \n [5] Exit')
        option = int(input('Enter: '))

        #Option 0 cal
        if option == 0:
            year = input("Enter the year: ")
            start_month = input("Enter the start month: ")
            end_month = input("Enter the end month: ")
            print(data.get_average(year,start_month,end_month))

        #Option 1 fetches the rainfall using the class function get_rainfall
        if option == 1:
            year = int(input("Enter the year: "))
            month = int(input("Enter the month: "))
            rainfall = data.get_rainfall(year, month)
            print(f"Rainfall: {rainfall}")

        #Option 2 deletes by first outputting the rainfall and then deleting that row.
        elif option == 2:
            year = int(input("Enter the year: "))
            month = int(input("Enter the month: "))
            rainfall = data.get_rainfall(year, month)
            print(f'The rainfall value of {rainfall} has been deleted.')
            data.del_value(month, year, file)

        #Option 3 inserts based on user inputs of year, month and rainfall which deletes any current and replaces it with new data.
        elif option == 3:
            year = int(input("Enter the year: "))
            month = int(input("Enter the month: "))
            rainfall = float(input("Enter the rainfall: "))
            data.del_value(month, year, file)
            data.insert_value(year, month, rainfall)

        #Option 4 asks the user on which season along with the year and uses the insert_quater function.
        elif option == 4:
            year = int(input("Enter the year: "))
            quarter = str(input("Enter a season: ['spring','summer','autumn','winter']: "))
            data.insert_quater(year, quarter, file)

        elif option == 5:
            print('Program will be closed.')
            run = False

        else:
            print('Invalid option selected.')
    
    else:
        print('Invalid option selected.')
        
    
