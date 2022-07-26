import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime,date
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

page = st.sidebar.selectbox(
    "Navigation Bar",
    ("Rating Prediction(K-Nearest Neighbours)", "Rating Prediction(Linear Regression)", "App Recommendation(Nearest Neighbours)","Data Analysis"))

if page=='Rating Prediction(K-Nearest Neighbours)' or page=='Rating Prediction(Linear Regression)':
	df=pd.read_csv('/Users/chiranthdg/Downloads/archive-4/googleplaystoredata.csv')
	df.drop(labels = ['Current Ver','Android Ver','App'], axis = 1, inplace = True)
	le = preprocessing.LabelEncoder()
	df['Genres'] = le.fit_transform(df['Genres'])
	genre_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
	df['Content Rating'] = le.fit_transform(df['Content Rating'])
	content_rate_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
	df['Type'] = df['Type'].replace({'Free':0, "Paid":1})
	type_mapping={'Free':0, "Paid":1}
	categories=['ART_AND_DESIGN',
	 'AUTO_AND_VEHICLES',
	 'BEAUTY',
	 'BOOKS_AND_REFERENCE',
	 'BUSINESS',
	 'COMICS',
	 'COMMUNICATION',
	 'DATING',
	 'EDUCATION',
	 'ENTERTAINMENT',
	 'EVENTS',
	 'FAMILY',
	 'FINANCE',
	 'FOOD_AND_DRINK',
	 'GAME',
	 'HEALTH_AND_FITNESS',
	 'HOUSE_AND_HOME',
	 'LIBRARIES_AND_DEMO',
	 'LIFESTYLE',
	 'MAPS_AND_NAVIGATION',
	 'MEDICAL',
	 'NEWS_AND_MAGAZINES',
	 'PARENTING',
	 'PERSONALIZATION',
	 'PHOTOGRAPHY',
	 'PRODUCTIVITY',
	 'SHOPPING',
	 'SOCIAL',
	 'SPORTS',
	 'TOOLS',
	 'TRAVEL_AND_LOCAL',
	 'VIDEO_PLAYERS',
	 'WEATHER']
	l=[]
	st.title('Rating Prediction')
	l.append(st.number_input('Enter the number of reviews received',step=1))
	l.append(st.number_input('Enter the size of the app'))
	l.append(st.number_input('Enter the number of Installs of the app',step=1))
	l.append(type_mapping[st.selectbox('Select the type of app',('Free','Paid'))])
	l.append(st.number_input('Enter the price of the app'))
	l.append(content_rate_mapping[st.selectbox('Enter the content rating of the app',('Adults only 18+',
	 'Everyone',
	 'Everyone 10+',
	 'Mature 17+',
	 'Teen',
	 'Unrated'))])
	l.append(genre_name_mapping[st.selectbox('Select the genre of App',
										     ('Action',
										 'Action;Action & Adventure',
										 'Adventure',
										 'Adventure;Action & Adventure',
										 'Adventure;Brain Games',
										 'Adventure;Education',
										 'Arcade',
										 'Arcade;Action & Adventure',
										 'Arcade;Pretend Play',
										 'Art & Design',
										 'Art & Design;Creativity',
										 'Art & Design;Pretend Play',
										 'Auto & Vehicles',
										 'Beauty',
										 'Board',
										 'Board;Action & Adventure',
										 'Board;Brain Games',
										 'Board;Pretend Play',
										 'Books & Reference',
										 'Books & Reference;Education',
										 'Business',
										 'Card',
										 'Card;Action & Adventure',
										 'Card;Brain Games',
										 'Casino',
										 'Casual',
										 'Casual;Action & Adventure',
										 'Casual;Brain Games',
										 'Casual;Creativity',
										 'Casual;Education',
										 'Casual;Music & Video',
										 'Casual;Pretend Play',
										 'Comics',
										 'Comics;Creativity',
										 'Communication',
										 'Communication;Creativity',
										 'Dating',
										 'Education',
										 'Education;Action & Adventure',
										 'Education;Brain Games',
										 'Education;Creativity',
										 'Education;Education',
										 'Education;Music & Video',
										 'Education;Pretend Play',
										 'Educational',
										 'Educational;Action & Adventure',
										 'Educational;Brain Games',
										 'Educational;Creativity',
										 'Educational;Education',
										 'Educational;Pretend Play',
										 'Entertainment',
										 'Entertainment;Action & Adventure',
										 'Entertainment;Brain Games',
										 'Entertainment;Creativity',
										 'Entertainment;Education',
										 'Entertainment;Music & Video',
										 'Entertainment;Pretend Play',
										 'Events',
										 'Finance',
										 'Food & Drink',
										 'Health & Fitness',
										 'Health & Fitness;Action & Adventure',
										 'Health & Fitness;Education',
										 'House & Home',
										 'Libraries & Demo',
										 'Lifestyle',
										 'Lifestyle;Education',
										 'Lifestyle;Pretend Play',
										 'Maps & Navigation',
										 'Medical',
										 'Music',
										 'Music & Audio;Music & Video',
										 'Music;Music & Video',
										 'News & Magazines',
										 'Parenting',
										 'Parenting;Brain Games',
										 'Parenting;Education',
										 'Parenting;Music & Video',
										 'Personalization',
										 'Photography',
										 'Productivity',
										 'Puzzle',
										 'Puzzle;Action & Adventure',
										 'Puzzle;Brain Games',
										 'Puzzle;Creativity',
										 'Puzzle;Education',
										 'Racing',
										 'Racing;Action & Adventure',
										 'Racing;Pretend Play',
										 'Role Playing',
										 'Role Playing;Action & Adventure',
										 'Role Playing;Brain Games',
										 'Role Playing;Pretend Play',
										 'Shopping',
										 'Simulation',
										 'Simulation;Action & Adventure',
										 'Simulation;Education',
										 'Simulation;Pretend Play',
										 'Social',
										 'Sports',
										 'Sports;Action & Adventure',
										 'Strategy',
										 'Strategy;Action & Adventure',
										 'Strategy;Creativity',
										 'Strategy;Education',
										 'Tools',
										 'Tools;Education',
										 'Travel & Local',
										 'Travel & Local;Action & Adventure',
										 'Trivia',
										 'Video Players & Editors',
										 'Video Players & Editors;Creativity',
										 'Video Players & Editors;Music & Video',
										 'Weather',
										 'Word'))])
	val=st.selectbox('Enter the catefory of the app',categories)
	for i in categories:
		if val==i:
			l.append(1)
		else:
			l.append(0)
	dates=st.date_input('Enter the date the app received it\'s last update',value=date(2014, 7, 6),
	     min_value=date(1970, 1, 1),max_value=date(2018,1,1))
	df.dropna(inplace = True)
	max_date=date(2018,1,1)
	dval=max_date-dates
	l.append(dval.days)
	if page=='Rating Prediction(K-Nearest Neighbours)':
		new = pickle.load(open("model_gplafin","rb"))
	else:
		new = pickle.load(open("linear_model","rb"))
	y_pred=new.predict([l])
	st.title('Predicted Rating for the app is:')
	st.title(y_pred)


if page=="App Recommendation(Nearest Neighbours)":
	ngb=pd.read_csv('CleanedData.csv')
	app_names=pd.read_csv('App_Names.csv')
	rec = pickle.load(open("recommendapps","rb"))
	val=st.selectbox('Select the app name:',app_names)
	print(ngb.head())
	print(app_names.head())
	st.write(ngb.columns)
	ent=(ngb.loc[ngb['App']==val]).drop('App',axis=1)
	st.write(ent)
	d,neighbors= rec.kneighbors(ent,n_neighbors=6)
	print(neighbors[0][1:])
	similar_apps = []
	for neighbor in neighbors[0][1:]:
	    similar_apps.append(app_names.loc[neighbor][0])
	fin = pd.DataFrame({'App':similar_apps})
	st.table(fin)



if page=="Data Analysis":
	df=pd.read_csv('/Users/chiranthdg/Downloads/archive-4/googleplaystoredata.csv')
	fig = px.box(df,"Size")
	f2= px.histogram(df,"Size")
	f3=px.box(df,"Rating")
	f4=px.histogram(df,"Rating")
	f5=px.histogram(df,"Price")
	k1=px.histogram(df,x="Category")
	f6=px.box(df.sort_values('Rating',ascending=False),x="Genres",y="Rating")


	st.header("Charts and Graphs")
	st.plotly_chart(fig)
	st.plotly_chart(f2)
	st.plotly_chart(f3)
	st.plotly_chart(f4)
	st.plotly_chart(f5)
	st.plotly_chart(k1)
	st.plotly_chart(f6)
