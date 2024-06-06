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
import altair as alt
import plotly.express as px

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

def plot_line_chart(data, x_axis, y_axis):
    if x_axis not in data.columns:
        st.error(f"Column '{x_axis}' not found in the data.")
        return
    if y_axis not in data.columns:
        st.error(f"Column '{y_axis}' not found in the data.")
        return
    if data.empty:
        st.error("The data is empty. Please provide valid data.")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data[x_axis], data[y_axis], marker='o')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'{y_axis} Over {x_axis}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    table_data = data[[x_axis, y_axis]].dropna().reset_index(drop=True)
    table_data.index += 1  # Start the table index from 1
    st.markdown('Table of Selected Columns:')
    st.write(table_data)

# Function to display line chart with markers and table
def st_line_chart_with_markers(data, axes, index_column):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        data.set_index(index_column, inplace=True)
        chart_data = data[axes].reset_index().melt(index_column, var_name='Column', value_name='Value')

        line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=index_column,
            y='Value',
            color='Column',
            tooltip=[index_column, 'Column', 'Value']
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)

        # Display a table based on the selected index column and y-axis columns
        table_data = data[axes].reset_index()  # Reset index to display the index column in the table
        table_data.index += 1  # Start the table index from 1
        st.markdown('Table of Selected Columns:')
        st.write(table_data)

def st_line_chart(data, axes, index_column):
    # Check if the selected columns are valid
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
        return

    # Check if the selected columns exist in the data
    missing_columns = [col for col in axes if col not in data.columns]
    if missing_columns:
        st.warning(f"The following selected columns are not found in the data: {', '.join(missing_columns)}")
        return

    # Check if the index column exists in the data
    if index_column not in data.columns:
        st.warning(f"The index column '{index_column}' is not found in the data.")
        return

    # Set the index and drop rows with missing values in the selected columns
    data.set_index(index_column, inplace=True)
    data.dropna(subset=axes, inplace=True)

    # Ensure the DataFrame is not empty after processing
    if data.empty:
        st.warning("The processed data is empty after removing rows with missing values.")
        return

    # Display the line chart using Streamlit
    st.line_chart(data[axes], marker='o')

    # Display a table based on the selected index column and y-axis columns
    table_data = data[axes].reset_index()  # Reset index to display the index column in the table
    table_data.index += 1  # Start the table index from 1
    st.markdown('Table of Selected Columns:')
    st.write(table_data)

def st_bar_chart(data, axes, index_column):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
    # Set the specified index column
        data.set_index(index_column, inplace=True)
        
        # Display the line chart using Streamlit
        st.bar_chart(data[axes])

        table_data = data[axes].reset_index()  # Reset index to display the index column in the table
        table_data.index += 1  # Start the table index from 1
        st.markdown('Table of Selected Columns:')
        st.write(table_data)

# Define the scatter chart function
def st_scatter_chart(data):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        st.scatter_chart(data)

        # Display a table based on the selected index column and y-axis columns
        table_data = data[axes].reset_index()  # Reset index to display the index column in the table
        table_data.index += 1  # Start the table index from 1
        st.markdown('Table of Selected Columns:')
        st.table(table_data)

def st_scatter_chart2(data):
    try:
        st.info('Select "Date Received" as the index column to see the occurrence of events on the timestamp.')
        # Allow multiple select for both axes and index_column
        axes = st.multiselect('Select Columns for Scatter Chart', data.columns)
        index_column = st.selectbox('Select Index Column', data.columns)
        
        if not axes or not index_column:
            st.warning("Please select at least one column for both axes and index.")
        else:
            scatter_data = data.set_index(index_column)[axes].reset_index()
            scatter_data['count'] = scatter_data.groupby(index_column).transform('count').iloc[:, 0]

            # Sort scatter_data based on index_column
            scatter_data.sort_values(by=index_column, inplace=True)

            fig = px.scatter(scatter_data, x=index_column, y=axes, hover_data={'count': True})
            st.plotly_chart(fig)
            
            # Display a table based on the selected index column and y-axis columns
            table_data = scatter_data
            table_data.index += 1  # Start the table index from 1
            st.markdown('Table of Selected Columns:')
            st.write(table_data)
    except Exception as e:
        st.error(f"An error occurred while generating the graph: {e}")

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
    # Convert source and target columns to strings to handle integer node identifiers
    data[source_column] = data[source_column].astype(str)
    data[target_column] = data[target_column].astype(str)
    
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
    
st.text('A tool designed to assist digital forensic analysis of email datasets.') 
st.text('Two analysis methods are provided in this tool: Timeline Analysis and Link Analysis.')    
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
            app_mode = st.sidebar.selectbox('Select Analysis Method', ['Timeline Analysis', 'Link Analysis'])
        

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
                        plot_line_chart(clean_data, x_axis, y_axis)
                        
                        buf = fig_to_buffer(plt.gcf())
                        st.download_button(label='Download Line Chart', data=buf, file_name='line_chart.png', mime='image/png')
                        
                    except Exception as e:
                        st.warning(f"Choose different columns for x-axis and y-axis to obtain meaningful table result: {e}.")
                
        

                elif graph_type == 'Bar':   
                    st.header('Bar Graph:')
                    
                    try:
                        st.write('Bar graph is used to plot Counts in this tool.')
                        st.info('Select "Date Received" column to see the occurrence of events on the timestamp. e.g. The plots will show that there are XX emails received on date XX-XX-XX')
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
            
                        st_scatter_chart2(data)

                    except Exception as e:
                        st.error(f"An error occurred while generating the graph: {e}")

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

                     # Display the data used for the network graph as a table
                    st.markdown('Table of Selected Columns:')
                    selected_data = data[[source_column, target_column]]
                    data.index += 1  # Start the table index from 1
                    st.write(selected_data)

                elif graph_type == 'Line':
                    st.header('Line Graph:')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_line_chart_with_markers(clean_data, axes, index_column)

                elif graph_type == 'Bar':    
                    st.header('Bar Graph:')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_bar_chart(clean_data, axes, index_column)          

            
            
        
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e} Please check the CSV file.")
        
