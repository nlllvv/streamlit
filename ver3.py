import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import io
from fpdf import FPDF
import os

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def sanitize_text(text):
    sanitized = ''.join(e for e in text if e.isalnum() or e.isspace())
    return sanitized

# Function to convert matplotlib figure to in-memory file
def fig_to_buffer(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

# Function to generate word cloud from text
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    return fig

# Function to plot line chart
def plot_line_chart(data, x_axis, y_axis):
    fig = plt.figure(figsize=(15, 8))
    plt.plot(data[x_axis], data[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} Over {x_axis}')
    plt.xticks(rotation=45)
    return fig

# Function to plot bar chart
def plot_bar_chart(data, x_axis, y_axis):
    fig = plt.figure(figsize=(15, 8))
    plt.bar(data[x_axis], data[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} Bar Chart')
    plt.xticks(rotation=45)
    return fig

# Function to plot pie chart
def plot_pie_chart(data, column):
    value_counts = data[column].value_counts()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title(f'Pie Chart for {column}')
    return fig

# Function to plot network graph
def plot_network_graph(data, source_column, target_column):
    G = nx.from_pandas_edgelist(data, source_column, target_column)
    fig = plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G, pos=None)  # Allow nodes to be moved

    # Nodes
    sources = set(data[source_column].unique())
    targets = set(data[target_column].unique())

    # Assign different colors for source and target nodes
    node_colors = ['blue' if node in sources else 'green' for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='black')

    # Draw labels with adjusted positions to prevent overlap
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif',
                            verticalalignment='bottom', horizontalalignment='right')
    plt.title('Network Graph')
    plt.axis('on')
    return fig

# Streamlit app
st.title("文件管理和数据处理示例")

# 获取用户输入的新文件夹名称
new_folder_name = st.text_input("输入新文件夹名称")

# 存储数据到浏览器的LocalStorage中
if new_folder_name:
    new_folder_path = os.path.abspath(new_folder_name)  # Get the absolute path of the new folder
    os.makedirs(new_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    st.query_params["new_folder"] = new_folder_name
    st.success(f"文件夹 '{new_folder_name}' 已创建并存储在'{new_folder_path}'")


st.title("Welcome to E-mailyser")
st.image('Mail.jpg')

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.title('Content:')
    st.write(data)
    
    app_mode = st.sidebar.selectbox('Select Visualization Type', ['Graphs', 'Counts', 'Network Graph', 'Word Cloud', 'Pie Chart'])
    
    if app_mode == 'Graphs':
        st.title('Graphs:')
        graph_type = st.sidebar.selectbox('Select Graph Type', ['Line', 'Timeline','Bar'])
        
        x_axis = st.selectbox('Select X-Axis Column', data.columns)
        y_axis = st.selectbox('Select Y-Axis Column', data.columns)
        clean_data = data.dropna(subset=[x_axis, y_axis])
        
        if graph_type == 'Line':
            clean_data = clean_data.sort_values(by=[x_axis])
            fig = plot_line_chart(clean_data, x_axis, y_axis)
            st.pyplot(fig)
            buf = fig_to_buffer(fig)
            st.download_button(label='Download Line Chart', data=buf, file_name='line_chart.png', mime='image/png')
        elif graph_type == 'Timeline':    
            st.markdown('**Timeline**')
            date_column = st.selectbox('Select Index:', clean_data.columns)

            # Set the specified date column as the index
            clean_data.set_index(date_column, inplace=True)

            # Sort data by the index
            clean_data.sort_index(inplace=True)
                    
            st.bar_chart(clean_data[[x_axis, y_axis]])

        elif graph_type == 'Bar':    
            st.markdown('**E-Mail From VS E-Mail To**')
            clean_data = clean_data.sort_values(by=[x_axis])
            st.bar_chart(clean_data[[x_axis, y_axis]])
    
    elif app_mode == 'Counts':
        st.title('Counts')
        
        # Select the x-axis column
        x_axis = st.selectbox('Select X-Axis Column', data.columns)
        
        # Drop rows with NaN values in the selected column
        clean_data = data.dropna(subset=[x_axis])
        
        try:
            # Count the occurrences of each unique value in the selected column
            email_counts = clean_data[x_axis].value_counts().sort_index()

            # Create the plot
            fig = plt.figure(figsize=(30, 8))
            plt.bar(email_counts.index, email_counts.values)
            plt.xlabel(x_axis)
            plt.ylabel('Counts')
            plt.title(f'Counts per {x_axis}')
            plt.xticks(rotation=45)

            # Display the plot
            st.pyplot(fig)
            buf = fig_to_buffer(fig)
            st.download_button(label='Download Counts Plot', data=buf, file_name='counts_plot.png', mime='image/png')
            plt.clf()  # Clear the figure

            # Display the counts as a table
            st.title('Counts Table')
            counts_table = pd.DataFrame(email_counts).reset_index()
            counts_table.columns = [x_axis, 'Counts']
            st.table(counts_table)
            
        except Exception as e:
            st.error(f"An error occurred while generating the email counts plot: {e}")

    elif app_mode == 'Network Graph':
        st.title('Network Graph')
        source_column = st.selectbox('Select Source Column', data.columns)
        target_column = st.selectbox('Select Target Column', data.columns)
        fig = plot_network_graph(data, source_column, target_column)
        st.pyplot(fig)
        buf = fig_to_buffer(fig)
        st.download_button(label='Download Network Graph', data=buf, file_name='network_graph.png', mime='image/png')

    elif app_mode == 'Word Cloud':
        text = ' '.join(data.dropna().astype(str).values.flatten().tolist())
        sanitized_text = sanitize_text(text)
        fig = generate_wordcloud(sanitized_text)
        st.pyplot(fig)
        buf = fig_to_buffer(fig)
        st.download_button(label='Download Word Cloud', data=buf, file_name='wordcloud.png', mime='image/png')
    
    elif app_mode == 'Pie Chart':
        column = st.selectbox('Select Column', data.columns)
        clean_data = data.dropna(subset=[column])
        fig = plot_pie_chart(clean_data, column)
        st.pyplot(fig)
        buf = fig_to_buffer(fig)
        st.download_button(label='Download Pie Chart', data=buf, file_name='pie_chart.png', mime='image/png')

   
