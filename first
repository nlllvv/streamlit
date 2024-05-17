import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model
import base64  # to open .gif files in Streamlit app
import matplotlib.pyplot as plt  # for plotting

@st.cache(suppress_st_warning=True)
def get_fvalue(val):    
    feature_dict = {"No": 1, "Yes": 2}    
    for key, value in feature_dict.items():        
        if val == key:            
            return value

def get_value(val, my_dict):    
    for key, value in my_dict.items():        
        if val == key:            
            return value

app_mode = st.sidebar.selectbox('Select Page', ['Email Counts VS Time', 'From To'])  # two pages

if app_mode == 'Email Counts VS Time':    
    st.title('LOAN PREDICTION :')      
    st.image(r'C:\Users\HP\Downloads\Mail.jpg') 
    st.markdown('Dataset :')    
    data = pd.read_csv(r'C:\Users\HP\Downloads\datasets\myemail.csv')    
     
    # Drop rows with invalid dates
    data = data.dropna(subset=['Date Received'])
    
    # Sort data by date
    data = data.sort_values(by='Date Received')
    
    st.write(data)    
    
    st.markdown('E-Mail From VS E-Mail To')

    # Aggregate data to count emails per day
    email_counts = data['Date Received'].value_counts().sort_index()

    # Creating a simple plot
    plt.figure(figsize=(30, 8))
    plt.plot(email_counts.index, email_counts.values)
    plt.xlabel('Date Received')
    plt.ylabel('Number of Emails')
    plt.title('Number of Emails Received Over Time')

    # Format the x-axis to show dates nicely
    plt.xticks(rotation=45)
    
    # Display the plot with st.pyplot
    st.pyplot(plt)

if app_mode=='From To':    
    st.title('LOAN PREDICTION :')      
    st.image(r'C:\Users\HP\Downloads\Mail.jpg') 
    st.markdown('Dataset :')    
    data=pd.read_csv(r'C:\Users\HP\Downloads\datasets\myemail.csv')    
    st.write(data)    
    st.markdown('E-Mail From VS E-Mail To ')    
    st.bar_chart(data[['E-Mail From','E-Mail To']])
