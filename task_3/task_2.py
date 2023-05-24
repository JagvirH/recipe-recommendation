import pandas as pd #Pandas will be used to access and make changes to the csv files.
import os #Makes it easier to navigate directories.


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
        
    def get_rainfall(self, year, month):
        filtered_df = self.df[(self.df.iloc[:, 0] == year) & (self.df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
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
        

#os.chdir("/Users/filip/Documents/Year 2/Semester 2/CO2106/Assignment 1")
os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/task_3")

#Enter the file name, user can enter with 'csv' or without 'csv'.
file = str(input("Which file would you like to access?  ")) 
if file[-4:] != ".csv":
    file = file + ".csv"


data = RainfallData(file)

#Menu system, user can choose what to do include exit.
run = True
while run:
    
    print('Please select one of the following options: \n [0] Finding Average \n [1] Finding Rainfall \n [2] Deleting Rainfall \n [3] Inserting New Rainfall Data \n [4] Inserting Data for Quarter \n [5] Exit')
    option = int(input(''))

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
        
    