import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
df=pd.read_csv("hepatitis.csv")
df1=df.copy()
l=[]
ch=0
page = st.sidebar.selectbox("Navigation Bar",("Decision tree", "Random forest tree", "Analysis"))
if page=='Decision tree' or page=='Random forest tree':
    st.title("Hepatitis classification")
    a=st.slider('Age:',1,100,50)
    se=st.selectbox('Sex:',['M','F'])
    l.append(st.radio('Steroid:',['yes','no']))
    l.append(st.radio('Antivirals:',['yes','no']))
    l.append(st.radio('Fatigue:',['yes','no']))
    l.append(st.radio('Malaise:',['yes','no']))
    l.append(st.radio('Anorexia:',['yes','no']))
    l.append(st.radio('Liver_big:',['yes','no']))
    l.append(st.radio('Liver_firm:',['yes','no']))
    l.append(st.radio('Spleen_palable:',['yes','no']))
    l.append(st.radio('Spiders:',['yes','no']))
    l.append(st.radio('Ascites:',['yes','no']))
    l.append(st.radio('Varices:',['yes','no']))
    l.append(st.slider('Bilirubin:',0.3,8.0,2.0))
    l.append(st.slider('Alk_phosphate:',26,295,45))
    l.append(st.slider('Sgot:',14,648,53))
    l.append(st.slider('Albumin:',0.3,6.5,2.0))
    l.append(st.slider('Protime:',0,100,40))
    l.append(st.radio('Histology:',['yes','no']))
    for i in range(len(l)):
        if l[i]=='yes':
            l[i]=1
        else:
            l[i]=2
    target=['The patient has Hepatitis','The patient doesn\'t have Hepatitis']
    if(page=='Decision tree'):
        new=pickle.load(open('model_h','rb'))
    else:
        new=pickle.load(open('model_h1','rb'))
    y_pred=new.predict([l])
    if 1 not in l:
        y_pred[0]=2
    elif 2 not in l:
        y_pred[0]=1
    y_pred=target[y_pred[0]-1]
    st.title(y_pred)
if page=='Analysis':
    st.title("PIE CHART")
    x=df.loc[df['class']==2].count()[0]
    y=df.loc[df['class']==1].count()[0]
    z=[x,y]
    fig = go.Figure(
    go.Pie(
    labels = ['Affected','Not Affected'],
    values = z,
    hoverinfo = "label+percent",
    textinfo = "value"
    ))
    st.plotly_chart(fig)
    st.title("BAR GRAPHS")
    df = pd.DataFrame(dict(
    X_axis = df['age'],
    Y_axis = df['alk_phosphate']
    ))
    fig2 = px.bar(        
    df, 
    x = "X_axis", 
    y = "Y_axis",
    color="X_axis",
    )
    st.plotly_chart(fig2)
    df = pd.DataFrame(dict(
    X_axis = df1['age'],
    Y_axis = df1['bilirubin']
    ))
    fig3 = px.bar(        
    df, 
    x = "X_axis", 
    y = "Y_axis",
    color="X_axis",
    orientation = 'h'
    )
    st.plotly_chart(fig3)
    #PATH:cd Desktop/Ayush/internship