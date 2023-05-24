import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.metrics.pairwise import euclidean_distances

os.chdir("/Users/jagvir/Desktop/University/semester_2/data_analytics/assignment 2")

'''


#df = pd.read_csv('recipes.csv', usecols=['title', 'category', 'cuisine', 'ingredients'])

# Load the dataset
recipes_df = pd.read_csv("recipes.csv")

def vec_space_method(title):
    # Create a new DataFrame containing only the title, category, cuisine, and ingredients columns
    df = recipes_df[["title", "category", "cuisine", "ingredients"]].copy()
    
    # Create a CountVectorizer object to convert the recipe ingredients into a bag of words representation
    vectorizer = CountVectorizer(stop_words="english")
    
    # Fit and transform the recipe ingredients using the vectorizer
    matrix = vectorizer.fit_transform(df["ingredients"])
    
    # Find the index of the input recipe
    title_index = df.index[df["title"] == title].tolist()[0]
    
    # Calculate the Euclidean distances between the input recipe and all other recipes
    distances = euclidean_distances(matrix[title_index], matrix)
    
    # Sort the distances and return the most similar recipes
    similar_indices = distances.argsort()[0][:10]
    similar_recipes = df.loc[similar_indices, "title"]
    return similar_recipes

# Test the function with an example recipe title
similar_recipes = vec_space_method("Beef and bean burrito")
print(similar_recipes)


recipes_df = pd.read_csv("recipes.csv")

def vec_space_method(title):
    # Create a new DataFrame containing only the title, category, cuisine, and ingredients columns
    df = recipes_df[["title", "category", "cuisine", "ingredients"]].copy()
    
    # Create a CountVectorizer object to convert the recipe ingredients into a bag of words representation
    vectorizer = CountVectorizer(stop_words="english")
    
    # Fit and transform the recipe ingredients using the vectorizer
    matrix = vectorizer.fit_transform(df["ingredients"])
    
    # Find the index of the input recipe
    title_index = df.index[df["title"] == title].tolist()[0]
    
    # Calculate the Euclidean distances between the input recipe and all other recipes
    distances = euclidean_distances(matrix[title_index], matrix)
    
    # Sort the distances and return the most similar recipes with their accuracies
    similar_indices = distances.argsort()[0][:10]
    similar_recipes = df.loc[similar_indices, "title"]
    similar_accuracies = [f"{1/(1+distance):.2%}" for distance in distances[0][similar_indices]]
    similar_recipes_with_accuracies = [f"{title} ({accuracy})" for title, accuracy in zip(similar_recipes, similar_accuracies)]
    for i in similar_recipes_with_accuracies:
        print(i)
    #return similar_recipes_with_accuracies

# Test the function with an example recipe title
similar_recipes = vec_space_method("Beef and bean burrito")
print(similar_recipes)


'''

recipes_df = pd.read_csv("recipes.csv")

def vec_space_method(title):
    # Create a new DataFrame containing only the title, category, cuisine, and ingredients columns
    df = recipes_df[["title", "category", "cuisine", "ingredients"]].copy()
    
    # Create a CountVectorizer object to convert the recipe ingredients into a bag of words representation
    vectorizer = CountVectorizer(stop_words="english")
    
    # Fit and transform the recipe ingredients using the vectorizer
    matrix = vectorizer.fit_transform(df["ingredients"])
    
    # Find the index of the input recipe
    title_index = df.index[df["title"] == title].tolist()[0]
    
    # Calculate the Euclidean distances between the input recipe and all other recipes
    distances = euclidean_distances(matrix[title_index], matrix)
    
    # Exclude the index of the input recipe from the list of similar recipes
    similar_indices = distances.argsort()[0]
    similar_indices = similar_indices[similar_indices != title_index][:10]
    
    # Sort the distances and return the most similar recipes with their accuracies
    similar_recipes = df.loc[similar_indices, "title"]
    similar_accuracies = [f"{1/(1+distance):.2%}" for distance in distances[0][similar_indices]]
    similar_recipes_with_accuracies = [f"{title} ({accuracy})" for title, accuracy in zip(similar_recipes, similar_accuracies)]
    for title in similar_recipes_with_accuracies:
        print(title)

# Test the function with an example recipe title
similar_recipes = vec_space_method("Beef and bean burrito")








