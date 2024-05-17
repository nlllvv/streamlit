import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)

st.title("Welcome to E-mailyser")
st.image(r'C:\Users\HP\Downloads\Mail.jpg') 

# Allow user to upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    st.title('Content:')      
    
    # Read uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.markdown('**Dataset:**')
    st.write(data)
    
    # Determine the app mode based on user selection
    app_mode = st.sidebar.selectbox('Select Visualization Type', ['Graphs', 'Word Cloud', 'Pie Chart'])
    
    if app_mode == 'Graphs':    
        st.title('Graphs:')      
        
        # Determine the graph type based on user selection
        graph_type = st.sidebar.selectbox('Select Graph Type', ['Line', 'Bar'])
        
        if graph_type == 'Line':    
            st.markdown('**E-Mail From VS E-Mail To**')
            
            # Allow user to select x-axis and y-axis columns for line plot
            x_axis = st.selectbox('Select X-Axis Column', data.columns)
            y_axis = st.selectbox('Select Y-Axis Column', data.columns)
            
            # Create line plot
            plt.figure(figsize=(30, 8))
            plt.plot(data[x_axis], data[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f'{y_axis} Over {x_axis}')

            # Display the plot
            st.pyplot(plt)

        elif graph_type == 'Bar':    
            st.markdown('**E-Mail From VS E-Mail To**')
            
            # Allow user to select x-axis and y-axis columns for bar chart
            x_axis = st.selectbox('Select X-Axis Column', data.columns)
            y_axis = st.selectbox('Select Y-Axis Column', data.columns)
            
            # Create bar chart
            st.bar_chart(data[[x_axis, y_axis]])
    
    elif app_mode == 'Word Cloud':
        st.title('Word Cloud')
        text = ' '.join(data.dropna().astype(str).values.flatten().tolist())  # Concatenate all text data
        generate_wordcloud(text)
    
    elif app_mode == 'Pie Chart':
        st.title('Pie Chart')
        column = st.selectbox('Select Column', data.columns)
        
        # Count the frequency of each value in the selected column
        value_counts = data[column].value_counts()
        
        # Create pie chart
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Pie Chart for {column}')
        
        # Display the plot
        st.pyplot(fig)
