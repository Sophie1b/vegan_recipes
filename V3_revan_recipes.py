#Import common modules
import streamlit as st
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
# text processing libraries
import re
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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
Other nutrients, that might be low for vegan eaters can include calcium, iron, protein and zinc. Nevertheless these can still be found in plant based foods."""


##############################################################################################
# loading the functions
##############################################################################################

def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    return frame



def recommended_recipes(df,recipe_id, sparse_matrix, k,metric='cosine'):

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

    recommend_recipes = df.loc[neighbour_ids, 'name']
    return recommend_recipes

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
recommender_system_type = recommender_df[recommender_df['type']== food_type_focus].reset_index()

col1, col2 = st.beta_columns(2)
with col1:
    st.subheader('Distribution of recipes bla bla bla')
    recommender_system_type.iloc[:10,1]
    
with col2:
  st.subheader(f"Top words according to their job positions")
  st.markdown("The higher the size, the more frequent the word is")
  # Create and generate a word cloud image:
  wordcloud = WordCloud(max_words= 50,background_color="white", collocations= False,
                   max_font_size= 500).generate(" ".join(recommender_system_type['ingredients_clean_processed']))
  st.image(wordcloud.to_array())

##############################################################################################    
# Recommender system
##############################################################################################
st.header("**Let's try a recommender**")
"""Bored of the above recipes? try these ones instead""" 
top_50 = recommender_system_type.sort_values('score',ascending=False).head(50).reset_index()
#top_50
tfidf = TfidfVectorizer()
sparse_matrix = tfidf.fit_transform(top_50['ingredients_clean_processed'])
#sparse_matrix
recipe_to_copy_nb = st.slider("What's the number of the recipe you would like ot see more of?", min_value=0, max_value=9, step=1)
recipe_to_copy_nb

recommended = recommended_recipes(top_50,recipe_to_copy_nb,sparse_matrix,5, metric='cosine')
recommended
