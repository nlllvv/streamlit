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
from dateutil import parser

@st.cache_data
#Function to read csv file
def load_data(file):
    return pd.read_csv(file)

def parse_date_column(data, column):
    if data[column].dtype == 'object':  # Check if the data type is string
        datetime_formats = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',                    # YYYY-MM-DD HH:MM:SS
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}',             # YYYY-MM-DD HH:MM:SS.SSS
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}',             # YYYY-MM-DD HH:MM:SS.SSSSSS
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{9}',             # YYYY-MM-DD HH:MM:SS.SSSSSSSSS
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d{2}:\d{2}',     # YYYY-MM-DD HH:MM:SS +HH:MM
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d{2}',           # YYYY-MM-DD HH:MM:SS +HH
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d{4}',           # YYYY-MM-DD HH:MM:SS +HHMM
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (AM|PM)',           # YYYY-MM-DD hh:mm:ss AM/PM
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}',                         # MM/DD/YYYY HH:MM
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',                   # MM/DD/YYYY HH:MM:SS
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}',                         # DD/MM/YYYY HH:MM
            r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{1,2}',                 # D/M/YYYY H:M
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+',               # YYYY-MM-DD HH:MM:SS TZ
        ]

        for fmt in datetime_formats:
            match = re.match(fmt, data[column].iloc[0])  # Check if the first value matches any format
            if match:
                try:
                    data[column] = data[column].apply(lambda x: parser.parse(x, fuzzy=True))
                    return data
                except Exception as e:
                    st.warning(f"Unable to parse column '{column}' as date. Displaying as original datatype.")
                    return data
        # If none of the formats match, return the original data
        return data
    elif pd.api.types.is_datetime64_any_dtype(data[column]):
        return data
    else:
        return data

def sanitize_text(text):
    sanitized = ''.join(e for e in text if e.isalnum() or e.isspace())
    return sanitized

#Function to extract emails and ISPs
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
    
    # Check if x_axis is a date column and parse it
    data = parse_date_column(data, x_axis)
    data = parse_date_column(data, y_axis)
    data = data.sort_values(by=[x_axis])
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data[x_axis], data[y_axis], marker='o')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'{y_axis} Over {x_axis}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    buf = fig_to_buffer(plt.gcf())
    st.download_button(label='Download Line Chart', data=buf, file_name='line_chart.png', mime='image/png')
                   
    table_data = data[[x_axis, y_axis]].dropna().reset_index(drop=True)
    table_data.index += 1  # Start the table index from 1
    st.markdown('Table of Selected Columns:')
    st.dataframe(table_data, width=1000, height=400)


# Function to display line chart with markers and table
def st_line_chart_with_markers(data, axes, index_column):
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        # Parse the date columns
        data = parse_date_column(data, index_column)
        for axis in axes:
            data = parse_date_column(data, axis)
        
        data = data.set_index(index_column)
        chart_data = data[axes].reset_index().melt(index_column, var_name='Column', value_name='Value')

        line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=index_column,
            y='Value',
            color='Column',
            tooltip=[index_column, 'Column', 'Value']
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
        st.write('*Hover over the plots to view details*')

        # Display a table based on the selected index column and y-axis columns
        table_data = data[axes].reset_index()  # Reset index to display the index column in the table
        table_data.index += 1  # Start the table index from 1
        st.markdown('Table of Selected Columns:')
        st.dataframe(table_data, width=1000, height=400)

def st_bar_chart(data, axes, index_column):
    # Check if index_column is a date column and parse it
    if not axes or index_column in axes:
        st.warning("Selected data columns must not include the index column.")
    else:
        # Parse the date columns
        data = parse_date_column(data, index_column)
        for axis in axes:
            data = parse_date_column(data, axis)
    # Set the specified index column
        data.set_index(index_column, inplace=True)
        
        # Display the line chart using Streamlit
        st.bar_chart(data[axes])

        st.write('*Hover over the bars to view details*')

        table_data = data[axes].reset_index()  # Reset index to display the index column in the table
        table_data.index += 1  # Start the table index from 1
        st.markdown('Table of Selected Columns:')
        st.dataframe(table_data, width=1000, height=400)

def st_scatter_chart(data):
    try:
        st.info('Select "Date Received" as the index column to see the occurrence of events on the timestamp.')
        
        # Allow multiple select for both axes and index_column
        axes = st.multiselect('Select Columns for Scatter Chart', data.columns)
        index_column = st.selectbox('Select Index Column', data.columns)
        
        if not axes or not index_column:
            st.warning("Please select at least one column for both axes and index.")
        else:
            # Check if index_column is a date column and parse it
            data = parse_date_column(data, index_column)
            
            scatter_data = data.set_index(index_column)[axes].reset_index()
            scatter_data['count'] = scatter_data.groupby(index_column).transform('count').iloc[:, 0]
    
            # Sort scatter_data based on index_column
            scatter_data.sort_values(by=index_column, inplace=True)
    
            fig = px.scatter(scatter_data, x=index_column, y=axes, hover_data={'count': True})
            st.plotly_chart(fig)
            st.write('*Hover over the plots to view counts*')
    
            # Display a table based on the selected index column and y-axis columns
            table_data = scatter_data
            table_data.index += 1  # Start the table index from 1
            st.markdown('Table of Selected Columns:')
            st.dataframe(table_data, width=1000, height=400)
    except Exception as e:
        st.error(f"An error occurred while generating the graph: {e}")

def st_bar_chart_counts(data, x_axis):
    # Parse the date column if necessary
    data = parse_date_column(data, x_axis)

    # Count the occurrences of each category in the y-axis column
    chart_data = data[x_axis].value_counts().reset_index()
    chart_data.columns = [x_axis, 'Counts']
    # Set the index to the x-axis column for plotting
    chart_data.set_index(x_axis, inplace=True)

    # Plot the bar chart using Streamlit
    st.bar_chart(chart_data)

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
col1, col2 = st.columns([1.2, 6])
with col1:
    st.image('logo.png')
with col2:
    st.title("Welcome to E-Mailyser")
    st.write('*A tool designed to assist digital forensic analysis of email datasets.*:sleuth_or_spy: :e-mail:') 
st.divider()

st.write('Two analysis methods are provided in this tool: ' ':mag: Timeline Analysis ' ':mag: Link Analysis')   

caseid = st.text_input("**Case ID**")
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
            with st.expander("Expand to view more details on dataset :arrow_down::arrow_down::arrow_down:"):
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
            st.sidebar.subheader('Case ID analysing:') 
            st.sidebar.header(caseid)
            st.sidebar.divider()
            
            st.sidebar.subheader('Analysis Module')
            st.sidebar.write('')
            app_mode = st.sidebar.selectbox('Select Analysis Method', ['Timeline Analysis', 'Link Analysis'])
        
            if app_mode == 'Timeline Analysis':
                
                graph_type = st.sidebar.selectbox('Select Visualization Graph Type', ['Line','Bar','Scatter'])
                
                if graph_type == 'Line':
                    st.header('Line Graph')
                    try:
                        st.write('*Line graph is used  in this tool to construct **timeline of events** chronologically.*')
                        st.info('Select "Date Received" as the x-axis column to see the occurrence of events on the timestamp.')
                        x_axis = st.selectbox('Select X-Axis Column', data.columns)
                        y_axis = st.selectbox('Select Y-Axis Column', data.columns)
                        clean_data = data.dropna(subset=[x_axis, y_axis])

                        clean_data = clean_data.sort_values(by=[x_axis, y_axis])
                        plot_line_chart(clean_data, x_axis, y_axis)
                             
                    except Exception as e:
                        st.warning(f"Choose different columns for x-axis and y-axis to obtain meaningful table result: {e}.")
                
        

                elif graph_type == 'Bar':   
                    st.header('Bar Graph')
                    
                    try:
                        st.write('*Bar graph is used in this tool to plot **counts**.*')
                        st.info('Select "Date Received" column to see the occurrence of events on the timestamp. e.g. The plots will show that there are XX emails received on date XX-XX-XX')
                        # Select the x-axis column
                        x_axis = st.selectbox('Select a data column to be counted', data.columns)
                        # Drop rows with NaN values in the selected column
                        clean_data = data.dropna(subset=[x_axis])

                        # Parse the date column if necessary
                        clean_data = parse_date_column(clean_data, x_axis)

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
                        st.dataframe(counts_table, width=1000, height=400)

                    except Exception as e:
                        st.error(f"An error occurred while generating the email counts plot: {e}")  
                
                elif graph_type == 'Scatter':
                    st.header('Scatter Chart:')
                    st.write('*Scatter graph is used  in this tool to connect **timeline of events** chronologically with **counts**.*')
                        
                    try:
            
                        st_scatter_chart(data)

                    except Exception as e:
                        st.error(f"An error occurred while generating the graph: {e}")

            if app_mode == 'Link Analysis':
                graph_type = st.sidebar.selectbox('Select Visualization Graph Type', ['Network','Line','Bar'])
                
                if graph_type == 'Network':
                    st.header('Network Graph')
                    st.write('*Network graph is used in this tool to find out the **relationship between different senders and receivers**.*')
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

                    st.divider()
                    # Provide a button for the user to download the HTML file
                    st.download_button(
                        label="Download Network Graph",
                        data=html_content,
                        file_name="network_graph.html",
                        mime="text/html"
                    )

                    try:
                        # Display the data used for the network graph as a table
                        st.markdown('Table of Selected Columns:')
                        selected_data = data[[source_column, target_column]]
                        data.index += 1  # Start the table index from 1
                        st.write(selected_data)
                    except Exception as e:
                        st.warning(f"Choose different columns for x-axis and y-axis to obtain meaningful table result: {e}")


                elif graph_type == 'Line':
                    st.header('Line Graph')
                    st.write('*Line graph can be used to find out the **relationship between different senders and receivers** on specific **timestamp**.*')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_line_chart_with_markers(clean_data, axes, index_column)

                elif graph_type == 'Bar':    
                    st.header('Bar Graph')
                    st.write('*Bar graph can be used to find out the **relationship between different senders and receivers** on specific **timestamp**.*')
                    axes = st.multiselect('Select Column(s)', data.columns)
                    index_column = st.selectbox('Select Index Column', data.columns)
                    clean_data = data.dropna(subset=axes)

                    st_bar_chart(clean_data, axes, index_column)          

            
            st.sidebar.divider()
            st.sidebar.header(':bulb: User Guide')
            st.sidebar.markdown(' :arrow_right:[Click Here](#)', unsafe_allow_html=True)


        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e} Please check the CSV file.")
        
