import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from bs4 import BeautifulSoup
import requests

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
        plt.scatter(self.df['rating_avg'] , self.df['rating_val'] )
        # Add labels and title
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Ratings')
        plt.title('Relationship between Average Rating and Number of Ratings')
        # Show the plot
        #plt.ylim(5, 50)
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
        # Retrieve the recipe titles corresponding to these indices
        top_indices = [i[0] for i in top_similarities]
        top_recipes = self.df['title'].iloc[top_indices].values
        # Display the top 10 recommendations
        print("Top 10 Recommendations for 'Chicken and coconut curry':")
        for recipe in top_recipes:
            print(recipe)


    def vec_space_method(self):

        return "yep"

'''
# to run the top 10 curries
df = pd.read_csv('recipes.csv')
check1 = recomendation(df)
check1.make_combine_features()
check1.top_10_curries()
'''

'''
check_df = pd.DataFrame()
features = ['title','category', 'cuisine', 'ingredients']
check_df["detail"] = df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
#print(check_df.head())
cv = CountVectorizer()
count_matrix = cv.fit_transform(check_df["detail"])
cosine_sim_matrix = cosine_similarity(count_matrix)
recipe_index = df[df['title'] == 'Chicken and coconut curry'].index[0]
recipe_similarities = list(enumerate(cosine_sim_matrix[recipe_index]))
sorted_similarities = sorted(recipe_similarities, key=lambda x: x[1], reverse=True)
top_similarities = sorted_similarities[1:11]
top_indices = [i[0] for i in top_similarities]
top_recipes = df['title'].iloc[top_indices].values
print("Top 10 Recommendations for 'Chicken and coconut curry':")
for recipe in top_recipes:
    print(recipe)
'''