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
#st.markdown("***'Select the nutrients you want to focus on' ***")

st.header("**Food nutrients**")
"""We all need nutrients to susrvive. Some are brought to us from the food we eat, some from the sun, some are transformed in our guts and bodies. In the Western world we are currently consuming too many kalories but at the same time not enough vitamins, minerals or oligoelements. There are mainly only traces of these element in the foods we eat, nevertheless they are essential for our wellbeing. 
In this data analysis project, I would like to focus on the nutrients we should be eating more of. In a typical vegan diet, some nutrients might be lacking. B12 for instance is not part of any plant based foods and needs to be added to the diet through supplements. Other nutrients that might be low for vegan eaters can usually still be found in plant based foods. 
Typically these can include a definciency in: calcium, iron, protein and zinc. """


##############################################################################################
# loading the functions
##############################################################################################

def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    return frame



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

    print(f"Because you want to focus on these nutrients {nutrient_focus}, and you liked {name_recipe} you should cook:")
    print('-------------------------------------.----------')
    print(df.loc[neighbour_ids, 'name'])
    print('===========================================')

##############################################################################################
# loading the data
##############################################################################################

portion_scores_df = pd.read_csv('portion_scores.csv', index_col=0)
recommender_recipes = pd.read_csv('recommender_recipes1.csv',index_col=0)




##############################################################################################    
# Selecting the nutrients to focus on 
##############################################################################################

all_nutrients = recommender_recipes.iloc[:,6:].columns.tolist()
st.subheader('**Select the nutrients you want to focus on**')
nutrient_focus = st.multiselect(' ',options=all_nutrients, default=None)
#nutrient_focus  = nutrient_focus.split()

sum_frame_by_column(recommender_recipes,'score',nutrient_focus).sort_values('score',ascending=False)

recommender_recipes.loc[:,'score'] = round(recommender_recipes.loc[:,'score']/(recommender_recipes.score.max())*100,2)


##############################################################################################    
# Creating the recipe score 
##############################################################################################


recommender_df = recommender_recipes[recommender_recipes['score']>5]

recommender_df = recommender_df.sort_values('score',ascending=False)

""" Because you want to focus on the selected nutrients, I recommend you cook:"""
recommender_df.iloc[:10,0]



##############################################################################################    
# Selecting the type of food 
##############################################################################################
st.header("**Types of recipes**")
"""If you want to narrow it down some more, select the type of food you want to cook""" 

food_type = recommender_df.type.unique().tolist()
st.subheader('**Would you like a recipe for breakfast, a main meal, a desert or a snack?**')

#food_type_focus = st.multiselect(' ',options=food_type, default=None)
food_type_focus = st.selectbox('Which food type are you interested in?',food_type)
recommender_system_type = recommender_df[recommender_df['type']== food_type]

recommender_system_type.iloc[:10,0]

##############################################################################################    
# Recommender system
##############################################################################################

top_50 = recommender_system_type.sort_values('score',ascending=False).head(50).reset_index()

tfidf = TfidfVectorizer()
sparse_matrix = tfidf.fit_transform(top_50['ingredients_clean_processed'])

recommended_recipies(top_50,4,sparse_matrix,5, metric='cosine')

