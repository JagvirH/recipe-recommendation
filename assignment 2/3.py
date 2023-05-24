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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/assignment 2")
class recomendation:
    def __init__(self,df):
        self.df = df

    def check_missing(self):
        print(self.df.isnull().sum())
        self.df.fillna('NA')
        
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
        self.df['combine_features'] = self.df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        
    def cosine_similarity_matrix(self):
        # create an instance of CountVectorizer
        cv = CountVectorizer()
        # fit_transform the "combine_features" column to create a sparse matrix of token counts
        count_matrix = cv.fit_transform(self.df['combine_features'])
        print(count_matrix)
        # compute the cosine similarity matrix using the sparse matrix
        cosine_sim_matrix = cosine_similarity(count_matrix)
        return cosine_sim_matrix

    def top_10(self):
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

    def return_recipe(self, url):
        response = None
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")
        if response is not None:
            soup = BeautifulSoup(response.content, 'html.parser')
            try:
                ingredients_list = soup.find('ul', class_='recipe-ingredients__list')
                ingredients = list(ingredients_list.find_all('li'))
            except AttributeError:
                ingredients = []
            final = []
            for i in ingredients:
                final.append(i.text.strip())
            method_list = soup.find('ol', class_='recipe-method__list')
            if method_list is not None:
                method = method_list.find_all('li')
                for i in method:
                    final.append(i.text.strip())
            return final
        else:
            return []

    def euclidean_distance(self, user_input):
        filtered_df = self.df[self.df['title'] == user_input]
        url = filtered_df.iloc[0]['recipe_url']
        user = self.return_recipe(url)

        recipe_list = [{'title': title, 'recipe': self.return_recipe(self.df.loc[df['title'] == title, 'recipe_url'].values[0])} for title in self.df["title"] if title != user_input]
        recipie_df = pd.DataFrame(recipe_list, columns=['title', 'recipe'])
        recipie_df = pd.concat([recipie_df, pd.DataFrame(recipe_list)], ignore_index=True)
        recipie_df = recipie_df[recipie_df['recipe'].apply(lambda x: len(x)>0)]

        cv = CountVectorizer()
        recipe_str = recipie_df['recipe'].apply(lambda x: ' '.join(x)).astype(str)
        count_matrix = cv.fit_transform(recipe_str)
        count_user_recipe = cv.transform([' '.join(user)])

        user_distances = euclidean_distances(count_user_recipe, count_matrix)
        recipe_similarities = list(enumerate(user_distances[0]))
        sorted_similarities = sorted(recipe_similarities, key=lambda x: x[1])

        top_similarities = []
        seen_titles = set()  # keep track of titles already seen
        for index, distance in sorted_similarities:
            title = recipie_df.loc[index, 'title']
            if title not in seen_titles:  # check if title has already been seen
                similarity_score = np.around(((1 / (1 + distance)) * 100), decimals=2)
                top_similarities.append((title, f"{similarity_score}% similarity"))
                seen_titles.add(title)
                if len(top_similarities) == 10:  # stop after finding 10 unique titles
                    break
            
        return top_similarities
    
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
check1.check_missing()
print(check1.stats())
print(check1.top_10())
print("-------------------------------------Q1.2-------------------------------------")
#check1.chart()
print("------------------------------------Q1.3 A------------------------------------")
#check1.make_combine_features()
print("------------------------------------Q1.3 B------------------------------------")
#Have made this class called 
print("------------------------------------Q1.3 C------------------------------------")

print("-------------------------------------Q2.4-------------------------------------")

print("-------------------------------------Q2.5-------------------------------------")



