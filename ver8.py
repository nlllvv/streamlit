import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from wordcloud import WordCloud
import io
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
import re

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def sanitize_text(text):
    sanitized = ''.join(e for e in text if e.isalnum() or e.isspace())
    return sanitized

def extract_emails(text):
    # Regex pattern for extracting email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails

def identify_isps(emails):
    isps = [email.split('@')[1] for email in emails]
    return isps

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

def st_line_chart(data, axes, index_column):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        # Set the specified index column
        data.set_index(index_column, inplace=True)
        
        # Display the line chart using Streamlit
        st.line_chart(data[axes])

        # Display a table based on the selected index column and y-axis columns
        table_data = data[axes]  # Extract relevant columns
        st.markdown('Table:')
        st.table(table_data)

def st_bar_chart(data, axes, index_column):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
    # Set the specified index column
        data.set_index(index_column, inplace=True)
        
        # Display the line chart using Streamlit
        st.bar_chart(data[axes])

        # Display a table based on the selected index column and y-axis columns
        table_data = data[axes]  # Extract relevant columns
        st.table(table_data)

# Define the scatter chart function
def st_scatter_chart(data):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        st.scatter_chart(data)

def st_bar_chart_counts(data, x_axis):
    # Count the occurrences of each category in the y-axis column
    chart_data = data[x_axis].value_counts().reset_index()
    chart_data.columns = [x_axis, 'Counts']

    # Set the index to the x-axis column for plotting
    chart_data.set_index(x_axis, inplace=True)

    # Plot the bar chart using Streamlit
    st.bar_chart(chart_data)

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

def plot_network_graph(data, source_column, target_column):
    # Create the graph from the pandas DataFrame
    G = nx.from_pandas_edgelist(data, source_column, target_column)
    
    # Generate the layout for the graph
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    
    # Define node colors based on whether they are source or target nodes
    sources = set(data[source_column].unique())
    node_colors = ['blue' if node in sources else 'green' for node in G.nodes()]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='black')
    
    # Draw labels with adjusted positions to prevent overlap
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_family='sans-serif')
    
    # Create legend
    blue_patch = mpatches.Patch(color='blue', label='Source Nodes')
    green_patch = mpatches.Patch(color='green', label='Target Nodes')
    plt.legend(handles=[blue_patch, green_patch], loc='best')

    # Set the title and axis
    plt.title('Network Graph')
    plt.axis('on')

    return fig

# Function to create and display network graph with different colors for source and target nodes
def create_network_graph(data, source_column, target_column):
    G = nx.from_pandas_edgelist(data, source_column, target_column)
    
    net = Network(notebook=True, width="100%", height="800px")
    
    sources = set(data[source_column])
    targets = set(data[target_column])
    
    # Add nodes with specific colors
    for node in G.nodes:
        if node in sources:
            net.add_node(node, color='blue', title=node)
        elif node in targets:
            net.add_node(node, color='green', title=node)
        else:
            net.add_node(node, title=node)  # Default color for other nodes

    # Add edges
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])


    # Temporary file to save and display the network graph
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name

# Function to check case id
def is_valid_caseid(caseid):
    if not caseid:
        return False, "Please enter a case ID to continue. e.g. ForensiCase_01"
    if not re.match("^[a-zA-Z0-9_-]*$", caseid):
        return False, "Case ID can only contain letters, numbers, hyphens, and underscores."
    # Add any additional checks here, such as uniqueness if needed
    return True, ""

# E-mailyser Streamlit app
col1, col2 = st.columns([1, 6])
with col1:
    st.image('logo.png')
with col2:
    st.title("Welcome to E-mailyser")
    
caseid = st.text_input("Case ID")
valid, message = is_valid_caseid(caseid)

if valid:
    st.write("New Case ID created:", caseid)
else:
    if message == "Please enter a case ID to continue. e.g. ForensiCase_01":
        st.info(message)
    elif message == "Case ID can only contain letters, numbers, hyphens, and underscores.":
        st.warning(message)

# Disable file uploader if Case ID is not valid
if valid:
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            with st.expander("Expand to view more details on dataset"):
                st.header('Dataset:')
                data.index += 1
                st.write(data)  

                # Extract email addresses and identify ISPs
                all_text = ' '.join(data.astype(str).values.flatten())
                emails = extract_emails(all_text)
                isps = identify_isps(emails)

                # Remove duplicates
                unique_emails = sorted(set(emails))
                unique_isps = sorted(set(isps))

                col1, col2 = st.columns(2)
                with col1:
                    # Create DataFrame for emails and ISPs
                    email_df = pd.DataFrame(unique_emails, columns=["Emails"])
                    email_df.index += 1  # Add 1 to the index to start from 1
                    email_df.columns = ['Emails']
                    st.markdown("Emails Found")
                    st.write(email_df)
                with col2:
                    isp_df = pd.DataFrame(unique_isps, columns=["ISPs"])
                    isp_df.index += 1  # Add 1 to the index to start from 1
                    isp_df.columns = ['ISPs']
                    st.markdown("Identified ISPs")
                    st.write(isp_df)
                            

            st.logo('logo.png')
            st.sidebar.markdown('Case ID analysing:') 
            st.sidebar.subheader(caseid)
            app_mode = st.sidebar.selectbox('Select Analysis Method', ['Link Analysis', 'Timeline Analysis'])
            
            if app_mode == 'Link Analysis':
                graph_type = st.sidebar.selectbox('Select Visualization Graph Type', ['Network','Line','Bar'])
                
                if graph_type == 'Network':
                    st.header('Network Graph')
                    source_column = st.selectbox('Select Source Column', data.columns)
                    target_column = st.selectbox('Select Target Column', data.columns)

                    html_file = create_network_graph(data, source_column, target_column)

                    # Create legend with colored patches
                    legend_html = """
                    <div style="margin-bottom: 2rem;">
                        <h6>Legend:</h6>
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="width: 12px; height: 12px; background-color: blue; margin-right: 0.5rem;"></div>
                            <span>Source Nodes</span>
                        </div>
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="width: 12px; height: 12px; background-color: green; margin-right: 0.5rem;"></div>
                            <span>Target Nodes</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 12px; height: 12px; background-color: grey; margin-right: 0.5rem;"></div>
                            <span>Default Nodes</span>
                        </div>
                    </div>
                    """

                    st.markdown(legend_html, unsafe_allow_html=True)
                    
                    # Display the network graph in Streamlit
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    
                    components.html(html_content, height=800)
                    # Clean up temporary file
                    os.remove(html_file)

                    # Provide a button for the user to download the HTML file
                    st.download_button(
                        label="Download Network Graph",
                        data=html_content,
                        file_name="network_graph.html",
                        mime="text/html"
                    )

                elif graph_type == 'Line':
                    st.header('Line Graph:')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_line_chart(clean_data, axes, index_column)

                elif graph_type == 'Bar':    
                    st.header('Bar Graph:')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_bar_chart(clean_data, axes, index_column)

            if app_mode == 'Timeline Analysis':
                
                graph_type = st.sidebar.selectbox('Select Visualization Graph Type', ['Line','Bar','Scatter'])
                
                if graph_type == 'Line':
                    st.header('Line Graph:')
                    try:
                        st.info('Select "Date Received" as the x-axis column to see the occurrence of events on the timestamp.')
                        x_axis = st.selectbox('Select X-Axis Column', data.columns)
                        y_axis = st.selectbox('Select Y-Axis Column', data.columns)
                        clean_data = data.dropna(subset=[x_axis, y_axis])

                        clean_data = clean_data.sort_values(by=[x_axis, y_axis])
                        fig = plot_line_chart(clean_data, x_axis, y_axis)
                        st.pyplot(fig)
                        buf = fig_to_buffer(fig)
                        st.download_button(label='Download Line Chart', data=buf, file_name='line_chart.png', mime='image/png')
                        try:
                            st.write("Table of Selected Columns:")
                            st.write(data[[x_axis, y_axis]])
                        except Exception as e:
                            st.error(f"An error occurred while displaying the table: {e}. The x-axis and y-axis must be different.")
                    except Exception as e:
                            st.error(f"An error occurred while generating the graph.")
        

                elif graph_type == 'Bar':   
                    st.header('Bar Graph:')
                    
                    try:
                        st.info('Bar graph is used to plot Counts in this tool.')
                        st.info('Select "Date Received" column to see the occurrence of events on the timestamp.')
                        # Select the x-axis column
                        x_axis = st.selectbox('Select a data column to be counted', data.columns)
                        # Drop rows with NaN values in the selected column
                        clean_data = data.dropna(subset=[x_axis])

                        # Count the occurrences of each unique value in the selected column
                        email_counts = clean_data[x_axis].value_counts().sort_index()

                        clean_data = clean_data.sort_values(by=[x_axis])
                        st.markdown('Counts Plots')  

                        st_bar_chart_counts(clean_data, x_axis)

                        # Display the counts as a table
                        st.markdown('Counts Table')
                        counts_table = pd.DataFrame(email_counts).reset_index()
                        counts_table.index += 1  # Add 1 to the index to start from 1
                        counts_table.columns = [x_axis, 'Counts']
                        st.table(counts_table)

                    except Exception as e:
                        st.error(f"An error occurred while generating the email counts plot: {e}")  
                
                elif graph_type == 'Scatter':
                    st.header('Scatter Chart:')
                    try:
                        st.info('Select "Date Received" as the index column to see the occurrence of events on the timestamp.')
                        axes = st.multiselect('Select Columns for Scatter Chart', data.columns)
                        index_column = st.selectbox('Select Index Column', data.columns)
                        # Ensure selected data columns are not the same as the index column
                        if index_column in axes:
                            st.warning("Selected data columns must not include the index column.")
                        else:
                            scatter_data = data.set_index(index_column)[axes]
                            st_scatter_chart(scatter_data)

                    except Exception as e:
                        st.error(f"An error occurred while generating the graph: {e}")          

            
            
        
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e} Please check the CSV file.")
        
