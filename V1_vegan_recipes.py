#Import common modules
import streamlit as st
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
# text processing libraries
import re
import string
from PIL import Image

# sklearn 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


##############################################################################################
# PAGE STYLING
##############################################################################################
st.set_page_config(page_title="Vegan Recipe Recommender", 
                   page_icon=":carrot:",
                   layout='wide')
                   
st.title("Vegan Recipe Recommender system :carrot:")
"""
[Vegan diet] (https://en.wikipedia.org/wiki/Veganism)
The vegan diet often focuses on what not to eat, through this recommender system, I focus on the foods and recipes to eat in order to increase the consumption of the selected nutrients. 
"""
# Page styling
title_image = Image.open("vegan_image.jpg")
st.image(title_image)
st.markdown("***'Select the nutrients you want to focus on' ***")

st.header("**Food nutrients**")
"""We all need nutrients to susrvive. Some are brought to us from the food we eat, some from the sun, some are transformed in our guts and bodies. In the Western world we are currently eating too much. In this data analysis project, I would like to focus on the nutrients we should be eating more of. In a typical vegan diet, some nutrients might be lacking. B12 for instance is not part of any plant based foods and needs to be added to the diet through supplements. Other nutrients that might be low for vegan eaters are: calcium, iron, protein and zinc. """


##############################################################################################
# loading the functions
##############################################################################################

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
            df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score'] = score_ + df_recipe.loc[df_recipe.ingredients_clean_processed.str.contains(food_),'score']            
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

    print(f"Because you want to focus on these nutrients {nutrient_focus}, you should cook:")
    print('-------------------------------------.----------')
    print(df.loc[neighbour_ids, 'name'])
    print('===========================================')

##############################################################################################
# loading the data
##############################################################################################

portion_scores_df = pd.read_csv('portion_scores.csv', index_col=0)
color_map1 = {'Fat':"#3265B2", 
             'Kilocalories': "#FFBF7F",
             'Carbohydrates':"#FAFE93",        
             'Fibres':"#5AD181",
             'Protein':"#FF0076",
             'Vitamin_A':"#FFAEAA",
             'Vitamin_B1':"#FFAEAA",
             'Vitamin_B2':"#FFAEAA",
             'Vitamin_B6':"#FFAEAA",
             'Vitamin_B12':"#FFAEAA",
             'Vitamin_B3':"#FFAEAA",
             'Vitamin_B9':"#FFAEAA",
             'Vitamin_B5':"#FFAEAA",
             'Vitamin_C':"#FFAEAA",
             'Vitamin_D':"#FFAEAA",
             'Vitamin_E':"#FFAEAA",
             'Potassium':"#CCCCCC",
             'Sodium':"#CCCCCC",
             'Calcium':"#CCCCCC",
             'Magnesium':"#CCCCCC",
             'Iron':"#C4A9D5",
            'Zinc':"#C4A9D5",
            'Selenium':"#C4A9D5"}






##############################################################################################    
# Selecting the nutrients to focus on 
##############################################################################################

all_nutrients = portion_scores_df.columns.tolist()
st.subheader('**Select the nutrients you want to focus on**')
nutrient_focus = st.multiselect(' ',options=all_nutrients, default=all_nutrients)
#nutrient_focus  = nutrient_focus.split()

sum_frame_by_column(portion_scores_df,'score',nutrient_focus).sort_values('score',ascending=False)


portion_scores_df[['Name','score']]
