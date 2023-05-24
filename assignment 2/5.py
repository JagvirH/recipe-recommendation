import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.metrics.pairwise import euclidean_distances
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

'''

'''


#This is so the program can find the csv file
os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/assignment 2")

#this is the name of the class 
class recomendation:

    def __init__(self,df):
        #This takes the dataframe in and dsaves it as a class variable
        self.df = df

    def check_missing(self): #The use of this class is to check missing valyes and fill them in
        #This prints the sum of all the missing values in every column
        print(self.df.isnull().sum())
        #This sets any of these missing areas as NA
        self.df.fillna('NA')
        
    def stats(self): #This class returns all the statistcs
        #This returns a description of the data set with all the statistics
        return(self.df.describe())
    
    def top_10(self):
        #Checks the 
        top_10_ratings = df.nlargest(10, 'rating_avg')
        return top_10_ratings["title"]

    def chart(self):
        # Filter the data to only include rows with "number of ratings" less than 100
        count_below_10 = len(self.df[self.df['rating_val'] < 10])

        # Print the count
        print("Number of rows where 'number of ratings' is below 10:", count_below_10)

        # Filter the data to only include rows with "number of ratings" greater than or equal to 10
        filtered_data = self.df[(self.df['rating_val'] >= 10)]

        # Create a scatter plot using the filtered data
        plt.scatter(filtered_data['rating_avg'], filtered_data['rating_val'], c='blue') # set color for values greater than or equal to 10
        plt.scatter(self.df[self.df['rating_val'] < 10]['rating_avg'], self.df[self.df['rating_val'] < 10]['rating_val'], c='red') # set color for values less than 10
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
        # compute the cosine similarity matrix using the sparse matrix
        cosine_sim_matrix = cosine_similarity(count_matrix)
        return cosine_sim_matrix

    def top_10_curries(self):
        recipe_index = self.df[self.df['title'] == 'Chicken and coconut curry'].index[0]
        recipe_similarities = list(enumerate(self.cosine_similarity_matrix()[recipe_index]))
        sorted_similarities = sorted(recipe_similarities, key=lambda x: x[1], reverse=True)
        top_similarities = sorted_similarities[1:11]
        # Retrieve the recipe titles and similarity scores corresponding to these indices
        top_indices = [i[0] for i in top_similarities]
        top_similarities_scores = [i[1] for i in top_similarities]
        top_recipes = self.df['title'].iloc[top_indices].values
        # Display the top 10 recommendations with their similarity scores as percentages
        print("Top 10 Recommendations for 'Chicken and coconut curry':")
        for i, recipe in enumerate(top_recipes):
            similarity_score = top_similarities_scores[i] * 100
            print(f"{i+1}. {recipe}: {similarity_score:.2f}% similarity")

    def vec_space_method(self,title):
        # Create a new DataFrame containing only the title, category, cuisine, and ingredients columns
        #df = recipes_df[["title", "category", "cuisine", "ingredients"]].copy()
        
        # Create a CountVectorizer object to convert the recipe ingredients into a bag of words representation
        vectorizer = CountVectorizer(stop_words="english")
        
        # Fit and transform the recipe ingredients using the vectorizer
        matrix = vectorizer.fit_transform(self.df["ingredients"])
        
        # Find the index of the input recipe
        title_index = self.df.index[self.df["title"] == title].tolist()[0]
        
        # Calculate the Euclidean distances between the input recipe and all other recipes
        distances = euclidean_distances(matrix[title_index], matrix)
        
        # Exclude the index of the input recipe from the list of similar recipes
        similar_indices = distances.argsort()[0]
        similar_indices = similar_indices[similar_indices != title_index][:10]
        
        # Sort the distances and return the most similar recipes with their accuracies
        similar_recipes = df.loc[similar_indices, "title"]
        similar_accuracies = [f"{1/(1+distance):.2%}" for distance in distances[0][similar_indices]]
        similar_recipes_with_accuracies = [f"{title} ({accuracy})" for title, accuracy in zip(similar_recipes, similar_accuracies)]
        print("The top 10 similar recipies to " + title + " is:")
        for title in similar_recipes_with_accuracies:
            print(title)

    
    def knn_similarity(self, recipe, n_neighbors=11):
        recipes_df = pd.read_csv('recipes.csv')

        vectorizer = TfidfVectorizer(stop_words='english')
        recipe_vector = vectorizer.fit_transform([recipe])

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(vectorizer.transform(recipes_df['ingredients']))
        neighbor_distances, neighbor_indices = knn.kneighbors(recipe_vector, return_distance=True)

        neighbor_indices = neighbor_indices[neighbor_indices != recipes_df.index[recipes_df['title'] == recipe][0]]
        neighbor_titles = recipes_df.loc[neighbor_indices]['title'].tolist()
        similarity_scores = (1 - neighbor_distances.flatten()) * 100
        similarity_scores = np.around(similarity_scores, decimals=1) # round to 2 decimal points

        results = []
        for title, score in zip(neighbor_titles, similarity_scores):
            results.append((title, f"{score}% similar"))
        return results

df = pd.read_csv('recipes.csv')
check1 = recomendation(df)

print("-------------------------------------Q1.1-------------------------------------")
print("-------the missing values stats-------")
#this class shows all the missing values, and inputs any missing values as NA
check1.check_missing()
print("------------the statistics------------")
#this shows the description and details of the dataframe that holds all the recipie values
print(check1.stats())
print("----top 10 highest rated recipies----")
#This gets the 10 highest rated recipies in the csv files and displays them
print(check1.top_10())

print("-------------------------------------Q1.2-------------------------------------")
'''
After seeing the data and researching, the threshold i decided to go with is only allow 
values with a number of rating that are above 10, and this can be shown on the grapgh,
with the values in red under the threshold and blue that is above
'''
check1.chart()

print("------------------------------------Q1.3 A------------------------------------")
# this function makes a new column in the data frame which combines all the features of every other column in the dataframe
check1.make_combine_features()

print("-----------------------------------Q1.3 B/C------------------------------------")
#This uses the combine features columns and uses cosine similarity to compare and find other similar recipies.
check1.top_10_curries()

print("-------------------------------------Q2.4-------------------------------------")
check1.vec_space_method("Angel food cake with lemon curd")
print("-------------------------------------Q2.5-------------------------------------")
#print(f"Top 10 Recommendations for '{recipe}':")
top_10_knn = check1.knn_similarity("Apple and raisin muffins")
for recipe_title in top_10_knn:
    print(recipe_title)
print("-------------------------------------Q2.6-------------------------------------")

print("---------Chicken tikka masala---------")
print("-----using vec_space_method-----")
check1.vec_space_method("Chicken tikka masala")
print("------------using KNN------------")
top_10_knn = check1.knn_similarity("Chicken tikka masala")
for recipe_title in top_10_knn:
    print(recipe_title)

print("------Albanian baked lamb with rice------")
print("-----using vec_space_method-----")
check1.vec_space_method("Albanian baked lamb with rice (Tavë kosi)")
print("------------using KNN------------")
top_10_knn = check1.knn_similarity("Albanian baked lamb with rice (Tavë kosi)")
for recipe_title in top_10_knn:
    print(recipe_title)

print("------Baked salmon with chorizo rice------")
print("-----using vec_space_method-----")
check1.vec_space_method("Baked salmon with chorizo rice")
print("------------using KNN------------")
top_10_knn = check1.knn_similarity("Baked salmon with chorizo rice")
for recipe_title in top_10_knn:
    print(recipe_title)

print("------------Almond lentil stew------------")
print("-----using vec_space_method-----")
check1.vec_space_method("Almond lentil stew")
print("------------using KNN------------")
top_10_knn = check1.knn_similarity("Almond lentil stew")
for recipe_title in top_10_knn:
    print(recipe_title)

print("-------------------------------------Q2.7-------------------------------------")