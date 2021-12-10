import streamlit as st
import pandas as pd
import numpy as np

# text processing libraries
import re
import string

# sklearn 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

#######################################################################

def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    return frame



def get_scores_recipes(df_food, df_recipe):
    for item in np.arange(0,len(df_food)):
        #print('--------------------------')
        food_ = df_food.Name.iloc[item]
        score_ = df_food.score.iloc[item]
        if df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),:].shape[0] >0:
            #print(ingred.loc[ingred.ingredients_cl.str.contains(food_),:].index)
            df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score'] = score_+df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score']            
            #print(food_,score_)
        else:
            df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score'] = df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score']
            #print('No food')
    return df_recipe

def recommended_recipies(df,recipe_id, sparse_matrix, k,metric='cosine'):

    name_recipe = df.loc[recipe_id, 'name']
    recipe_to_assess = sparse_matrix[recipe_id]
    neighbour_ids = []
    kNN = NearestNeighbors(n_neighbors=k, 
                           metric=metric)
    kNN.fit(sparse_matrix)

    neighbour = kNN.kneighbors(recipe_to_assess, return_distance=False)
    
    for i in range(1,k):
        n = neighbour.item(i)
        neighbour_ids.append(n)
    return neighbours_id
    
    
##############################################################################################
# PAGE STYLING
##############################################################################################
st.set_page_config(page_title="Sophie Dashboard ", 
                   page_icon="🍝",
                   layout='wide')
                   
st.title("Recommender system made by Sophie! 🍃")
"""
[vegans] (https://en.wikipedia.org/wiki/Titanic)
blabla blaa
"""

#######################################################################

#Loading the data
portion_scores_df = pd.read_csv('portion_scores.csv', index_col=0)
vegan_recipes = pd.read_csv('vegan_recipes.csv',index_col=0)

##Food 
nutrient_focus = ['zinc', 'calcium', 'potassium'] #fixed set of ingredients
sum_frame_by_column(portion_scores_df,'score',nutrient_focus).sort_values('score',ascending=False)
portion_scores_df.loc[:,'score'] = round(portion_scores_df.loc[:,'score']/(portion_scores_df.score.max())*100,2)


## Vegan recipes
vegan_recipes['score'] = 0

portion_scores_recipes = get_scores_recipes(portion_scores_df, vegan_recipes)
portion_scores_recipes.sort_values('score',ascending=False)

