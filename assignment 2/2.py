import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from bs4 import BeautifulSoup
#import requests
from sklearn.metrics.pairwise import euclidean_distances


os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/assignment 2")
class recomendation:
    def __init__(self,df):
        self.df = df

    def check_missing(self):
        print(self.df.isnull().sum())
        self.df.fillna('miss')
        
    def stats(self):
        return(self.df.describe())
    
    def top_10(self):
        top_10_ratings = df.nlargest(10, 'rating_avg')
        return top_10_ratings

    def chart(self):
        # Filter the data to only include rows with "number of ratings" less than 100
        count_below_10 = len(self.df[self.df['rating_val'] < 10])

        # Print the count
        print("Number of rows where 'number of ratings' is below 10:", count_below_10)
        filtered_data = self.df[(self.df['rating_val'] > 10)]
        
        # Create a scatter plot using the filtered data
        plt.scatter(filtered_data['rating_avg'], filtered_data['rating_val'])
        
        # Add labels and title
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Ratings')
        plt.title('Relationship between Average Rating and Number of Ratings (Number of Ratings < 100)')
        
        # Show the plot
        plt.show()

    def make_combine_features(self):
        features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']
        # Create a new column called combine_features by joining all the features together with space
        self.df['combine_features'] = self.df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        #print(self.df['combine_features'])
    
    def cosine_similarity_matrix(self):
        # create an instance of CountVectorizer
        cv = CountVectorizer()
        # fit_transform the "combine_features" column to create a sparse matrix of token counts
        count_matrix = cv.fit_transform(self.df['combine_features'])
        print(count_matrix)
        # compute the cosine similarity matrix using the sparse matrix
        cosine_sim_matrix = cosine_similarity(count_matrix)
        return cosine_sim_matrix
    
    def top_10_curries(self):
        recipe_index = self.df[self.df['title'] == 'Chicken and coconut curry'].index[0]
        recipe_similarities = list(enumerate(self.cosine_similarity_matrix()[recipe_index]))
        sorted_similarities = sorted(recipe_similarities, key=lambda x: x[1], reverse=True)
        top_similarities = sorted_similarities[1:11]
        # Retrieve the recipe titles corresponding to these indices
        top_indices = [i[0] for i in top_similarities]
        top_recipes = self.df['title'].iloc[top_indices].values
        # Display the top 10 recommendations
        print("Top 10 Recommendations for 'Chicken and coconut curry':")
        for recipe in top_recipes:
            print(recipe)




    

df = pd.read_csv('recipes.csv')
check1 = recomendation(df)
#check1.chart()
check1.make_combine_features()
check1.top_10_curries()
#The Euclidean distance
#user_input = input("Enter a recipe title: ")
user_input = "Almond tart"
#check1.vec_space_method(user_input)



