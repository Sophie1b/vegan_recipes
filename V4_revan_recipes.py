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

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
st.image(title_image, width=500)
#st.markdown("***'Select the nutrients you want to focus on' ***")

st.header("**Food nutrients**")
"""We all need nutrients to survive. Some are brought to us from the food we eat, some from the sun, some are transformed in our guts and bodies. The scientific community have currently labels over 150 nutrients, but some believe there is still a lot more  to be discovered - a black matter of trace nutrients in the foods we eat. 
Nutrients are often classified betweent the macro nutrients - fat, carbohydrates and protein that together constitute the kilocalories, and micro nutrients - vitamins, minerals, oligoelements, that are equaly important but in smaller quantities.""" 


"""In the Western world, we are currently consuming too many kalories but at the same time not enough of the essential micronutrients - vitamins, minerals or oligoelements. There are mainly only traces of these element in the foods we eat, nevertheless they are essential for our wellbeing. 
In this data analysis project, I would like to focus on the nutrients we should be eating more of. In a typical vegan diet, some nutrients might be lacking. B12 for instance is not part of any plant based foods and needs to be added to the diet through supplements.  
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

    recommend_recipes = df.loc[neighbour_ids, 'name':'ingredients']
    return recommend_recipes

##############################################################################################
# loading the data
##############################################################################################

df_forplot = pd.read_csv('circular_bar_vegan_clean.csv', index_col=0)
recommender_recipes = pd.read_csv('recommender_recipes1.csv',index_col=0)

df_forplot.drop(['Category','Vegan_cat','portion','vegan_score','vegan_score_2',
                 'Food_category','count'],axis=1,inplace=True)


color_list = ["#FF7574","#F1C2A6","#65823F","#FF996B","#F8DE70","#F8DE70","#F8DE70","#F8DE70","#F8DE70","#F8DE70","#F8DE70",
                  "#F8DE70","#F8DE70","#F8DE70","#F8DE70","#C5E6C7","#C5E6C7","#C5E6C7","#C5E6C7","#76B996",
                  "#76B996","#76B996"]

##############################################################################################    
# Selecting the nutrients to focus on 
##############################################################################################

all_nutrients = recommender_recipes.iloc[:,6:].columns.tolist()
st.subheader('**Select the nutrients you want to focus on**')
nutrient_focus = st.multiselect(' ',options=all_nutrients, default=None)


sum_frame_by_column(recommender_recipes,'score',nutrient_focus).sort_values('score',ascending=False)

recommender_recipes.loc[:,'score'] = round(recommender_recipes.loc[:,'score']/(recommender_recipes.score.max())*100,2)

##############################################################################################    
# spider plot with top 3 foods
##############################################################################################

st.header("**Top food for that nutrient**")



df_top3 = df_forplot.sort_values(nutrient_focus,ascending=False).head(3)
name_1 = df_top3.iloc[1,22]
name_1
#st.subheader(f"The values of the nutrients")

#col1, col2 = st.beta_columns(2)
#with col1:
df = df_forplot[df_forplot['Name']==name_1]
df = df.T
df.drop('Name',inplace=True)
df = df.reset_index()
df.rename(columns={'index':'Name',df.columns[1]:'Value'},inplace=True)
#df.rename(columns={df.columns[0]:'Value'},inplace=True)
st.dataframe(df)
st.subheader(f"The distribution of nutrients")
fig = px.bar(df, x = 'Name', y = 'Value', color= 'Name', color_discrete_sequence = color_list)
st.plotly_chart(fig)

#chart_data = pd.DataFrame(
#     df.Value.unique().tolist(),
#     columns=["Nutrients"])
#st.bar_chart(chart_data)

#with col2:


#fig = px.bar(df, x='Name', y='Value')

#fig.show()
# fig = px.bar_polar(df, r='Value', theta=np.arange(0,360,17))
                   #, color='Name')
     #              template="simple_white",width=1000,height=1000,
      #             color_discrete_sequence= color_list,hover_data={"Value":True, "Name":True})

#fig.update_polars(angularaxis = dict(visible=False),radialaxis=dict(visible=False))


##############################################################################################    
# Creating the recipe score 
##############################################################################################

st.header("* * * *")
recommender_df = recommender_recipes[recommender_recipes['score']>5]

recommender_df = recommender_df.sort_values('score',ascending=False)

#""" Because you want to focus on the selected nutrients, I recommend you look at the following recipes:"""
#recommender_df.iloc[:10,0]



##############################################################################################    
# Selecting the type of food 
##############################################################################################
st.header("**Types of recipes**")

food_type = recommender_df.type.unique().tolist()
st.subheader('**To narrow it down, select the type of food you want to cook. Would you like a recipe for a breakfast food, a meal, a desert or a snack?**')

#food_type_focus = st.multiselect(' ',options=food_type, default=None)
food_type_focus = st.selectbox(' ',food_type)
recommender_system_type = recommender_df[recommender_df['type']== food_type_focus].reset_index()

col1, col2 = st.beta_columns(2)
with col1:
  st.subheader('Recommended recipes:')
  st.dataframe(recommender_system_type.iloc[:10,1:4])
    
with col2:
  st.subheader("The top foods in the recipes suggested to you")
 # st.markdown("(The higher the size, the more frequent the word appears)")
  # Create and generate a word cloud image:
  wordcloud = WordCloud(max_words= 50,background_color="white", collocations= False,
                   max_font_size= 500).generate(" ".join(recommender_system_type['ingredients_clean_processed']))
  st.image(wordcloud.to_array())

##############################################################################################    
# Recommender system
##############################################################################################
st.header("**Bored of the above recipes? Here are some more recommendations**")
st.subheader("Using the slider, select the number corresponding to the recipe above for which you would like more recommendations.")
top_50 = recommender_system_type.sort_values('score',ascending=False).head(50).reset_index()

tfidf = TfidfVectorizer()
sparse_matrix = tfidf.fit_transform(top_50['ingredients_clean_processed'])
recipe_to_copy_nb = st.slider(" ", min_value=0, max_value=6, step=1)
recipe_to_copy_nb

recommended = recommended_recipes(top_50,recipe_to_copy_nb,sparse_matrix,5, metric='cosine')

recommended
