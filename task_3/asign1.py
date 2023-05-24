from tracemalloc import start
import pandas as pd #Pandas will be used to access and make changes to the csv files.
import os #Makes it easier to navigate directories.
import csv


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
        return "the average rainfall in the year " + year + " from the months of " + str(start_month) + " to " + str(end_month) + " is " + str(ans)
        
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

    def __init__(self): # setting the class to make sure the database has headings
        headers = ['City', 'Year', 'Month', 'Rainfall']
        with open ("Database.csv", 'a') as db:
            file_is_empty = os.stat('Database.csv').st_size == 0    
            writer = csv.writer(db)
            if file_is_empty:
                writer.writerow(headers)
                print('headers added')
        self.df = pd.read_csv("Database.csv")
    

    def insert(self, city, month, year ):
        df = pd.read_csv(city)
        filtered_df = df[(df.iloc[:, 0] == year) & (df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        rainfall = filtered_df.iloc[0,5] #Index where the rainfall would be found in the csv file.
        
        return "through"
        '''
        heading = ['city', 'year', 'month', 'rainfall']
        with open("Database.csv", 'a') as db:
            file_empty = os.stat("Database.csv").st_size == 0    
            writer = csv.writer(db)
            if file_empty:
                writer.writerow(heading)
        self.df = pd.read_csv("Database.csv")

    def insert(self, city, year, month): # finds the data first to see if it exists and then add its to the database csv file
        self.df = pd.read_csv(city)
        
        #reader = csv.reader(city)
        #print(df.iloc[:,0])

        filtered_df = self.df[(self.df.iloc[:, 0] == year) & (self.df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        rainfall = filtered_df.iloc[0,5] #Index where the rainfall would be found in the csv file.
        return rainfall
        
        
        
        
        check1 = df.iloc[: ,0] == 1853
        check2 = df.iloc[: ,1] == 1
        matching = df[check1]
        
        #print(matching.iloc[:,0]==year)

        filtered = df[check1 & check2]

        #filtered_df = df[(df.iloc[:, 0] == year) & (df.iloc[:, 1] == month)]

        
        print(filtered.iloc[:1])
        print('-=======================8')
        if filtered.empty:
            print(filtered.columns)  # print the column labels
            print(filtered.iloc[0:5])
            return "NOPE"
        
        #print(filtered_df.iloc[0])
        

        

        return "--------"

        
        df = pd.read_csv(city)
        print(df.iloc[:, 0])
        if df.iloc[:, 0] == year:
            print("YES")
        filtered_df = df[(df.iloc[:, 0] == year) & (df.iloc[:, 1] == month)] #Variables of year and month are found in row 1, 2 and therefore must be equal to input for it to match
        if filtered_df.empty:
            return None
        #rainfall = filtered_df.iloc[0,5]
        print(filtered_df)
        '''
        
        '''
        check = False
        with open(city, 'r') as data_file:
            reader = csv.reader(data_file)
            next(reader)  # skip header row
            for row in reader:
                #print(row[0] + " " +row[1])
                if row[0] == year and row[1] == month:
                    print (row)
                    rainfall = row[5]
                    check = True
                    break
        if check == False:
            return "Data does not exist"
        
        else:
            check = False
            with open("Database.csv", 'r') as db:
                reader = csv.reader("Database.csv")
                next(reader)
                for row in reader:
                    if row[0] == year and row[1] == month:
                        check = True
            if check == False: 
                add = {
                            "city": city,
                            "year": year,
                            "month": month,
                            "rainfall": rainfall
                        }
                write = pd.DataFrame([add],index=[0]) #this adds it to the database
                write.to_csv("Database.csv", mode="a", index=False, header=False)
                print("added to database file")
            else:
                return "it has been added"
                    


    def delete(self,city,year,month):
        df = df[~((df.iloc[:, 0] == year) & (df.iloc[:, 1] == month))]
        df.to_csv(file, index=False)


    def sma(self):
        return()

'''

os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/task_3")


run = True
while run:

    print('Would you like to either \n [1] Use existing files \n [2] Archive data ')
    choice = int(input('Enter: '))

    if choice == 2:
        print('Would you like to either \n [1] Add a rainfall \n [2] Delete a rainfall \n [3] Calculate sma ')
        archive_choice = int(input('Enter: '))
    
        if archive_choice == 1: # this choice is to insert data into the database
            city = str(input("Which file would you like to access?  ")) 
            if city[-4:] != ".csv":
                city = city + ".csv"
            year = input("Enter the year: ")
            month = input("Enter the month: ")
            check = Archive()
            print(check.insert(city,year,month))
        
        elif archive_choice == 2: #this choice is to delete data in the database
            #Enter the file name, user can enter with 'csv' or without 'csv'.
            city = str(input("Which file would you like to access?  ")) 
            if city[-4:] != ".csv":
                city = city + ".csv"
            year = input("Enter the year: ")
            month = input("Enter the month: ")
            check = Archive()
            check.delete(city,year,month)
        
        elif archive_choice == 3:
            print()
        
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
        
    



