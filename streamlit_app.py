import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def creds_entered():
    if st.session_state["user"].strip() == st.secrets["user"]["Username"] and st.session_state["password"].strip() == st.secrets["user"]["Password"]:
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        st.error("Invalid Username/Password")


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label="Username :", value="", key="user")
        st.text_input(label="Password :", value="", key="password", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            st.text_input(label="Username :", value="", key="user")
            st.text_input(label="Password :", value="", key="password", type="password", on_change=creds_entered)
            return False


def load_data(query, engine):
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error executing query: {e}")


def format_data(data):
    # Convert 'Date' to datetime and format it as 'mm/dd/yyyy'
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%m/%d/%Y')
    
    # Convert 'Time' column to a string representation
    data['Time'] = data['Time'].apply(lambda x: str(x).split(' ')[2])
    
    return data


def create_gauge(label, value, min_value=240, max_value=3000):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [240, 3000]}},
        number={'valueformat': ".0f"}
    ), layout=dict(
        width=100, 
        height=160, 
        margin=dict(b=4, t=40),
    ))
    return fig


def get_value_for_label(data, label):
    # Filter the DataFrame for the given label and get the last value in the 'value' column
    value = data[data['label'] == label]['value'].iloc[-1] if not data[data['label'] == label].empty else 0
    return value


def create_line_chart(data, y_column, width=400, height=300):
    fig = px.line(data, x='Datetime', y=y_column, labels={y_column: 'Pressure'})
    fig.update_layout(
        margin=dict(t=40, b=40),  # Adjust top (t) margin to be minimal
        width=width,
        height=height
    )
    return fig


def display_trailer_info(trailer_number, recent_pressure, line_chart, last_time_full, current_burn_rate, remaining_fuel, status):
    main_columns = st.columns([1, 2, 1])  # Adjust the width of the columns

    # Gauge Column
    with main_columns[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)   
        st.plotly_chart(create_gauge(f'Trailer {trailer_number} Pressure', recent_pressure), use_container_width=True)
    
    # Line Chart Column
    with main_columns[1]:
        st.plotly_chart(line_chart, use_container_width=True)
    
    # Stats Column
    with main_columns[2]:
        st.markdown("<br>", unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True) 
        st.markdown(f"**Last time full:** {get_last_time_full(data, f'Trailer_{trailer_number}_Pressure')}")
        st.markdown(f"**Current burn rate (hrs):** {current_burn_rate}")
        st.markdown(f"**Remaining Fuel (hrs):** {remaining_fuel}")
        # Display the status with the appropriate color
        st.markdown(f"**Status:** <span style='color: {color};'>{status}</span>", unsafe_allow_html=True)


def get_last_time_full(data, label):
    # Ensure 'Date' is in datetime format and create a 'DateTime' column
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Time'] = pd.to_timedelta(data['Time'])
    data['DateTime'] = data['Date'] + data['Time']
    # Filter the DataFrame for rows where the pressure was >= 3000 and 'Offline' is 0 (Online)
    filtered_data = data[(data[label] >= 3000) & (data['Offline'] == 0)]

    if not filtered_data.empty:
        # Get the last row where the condition was met
        last_row = filtered_data.iloc[-1]
        
        # Format 'DateTime' as 'mm/dd/yyyy - HH:MM:SS' and return
        return last_row['DateTime'].strftime('%m/%d/%Y - %H:%M:%S')
    else:
        return "N/A"


def get_current_burn_rate(data, trailer_pressure_column):
    # Ensure 'Date' and 'Time' are in datetime format and handle parsing errors
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Time'] = pd.to_timedelta(data['Time'].astype(str), errors='coerce')
    
    # Merge 'Date' and 'Time' into a single 'DateTime' column
    data['DateTime'] = data['Date'] + data['Time']

    # Handle rows with NaT in 'DateTime'
    data = data.dropna(subset=['DateTime'])

    # Sort the DataFrame by DateTime in descending order
    data = data.sort_values(by='DateTime', ascending=False)

    # Find the most recent time when the pressure was <= 240 and > 0
    low_pressure_idx = data[(data[trailer_pressure_column] <= 240) & 
                            (data[trailer_pressure_column] != 0) & 
                            (data['Offline'] != 1)].index.min()

    # If no such instance is found, return np.nan
    if np.isnan(low_pressure_idx):
        return np.nan

    # Find the most recent time before low_pressure_idx when the pressure was >= 3000
    high_pressure_idx = data.loc[low_pressure_idx:][(data[trailer_pressure_column] >= 3000) & 
                                                     (data['Offline'] != 1)].index.max()

    # If no such instance is found, return np.nan
    if np.isnan(high_pressure_idx):
        return np.nan

    # Calculate the time difference between the two instances in hours
    time_diff = (data.loc[low_pressure_idx, 'DateTime'] - data.loc[high_pressure_idx, 'DateTime']) / pd.Timedelta(hours=1)

    # Ensure time_diff is non-zero before calculating burn_rate
    if time_diff == 0:
        return np.nan
    
    # Calculate the burn rate as the pressure drop per hour
    pressure_diff = data.loc[high_pressure_idx, trailer_pressure_column] - data.loc[low_pressure_idx, trailer_pressure_column]
    burn_rate = pressure_diff / time_diff
    
    return burn_rate


def get_avg_burn_rate(data, trailer_pressure_column):
    # Sort the DataFrame by Date in ascending order
    data = data.sort_values(by='Date')
    # Initialize a list to store individual burn rates
    burn_rates = []
    # Iterate over the rows of the DataFrame
    for i in range(1, len(data)):
        # If the current row has pressure >= 3000 and the next row has pressure <= 240
        # and 'Offline' is 0 for both rows
        if (data.iloc[i-1][trailer_pressure_column] >= 3000 and data.iloc[i][trailer_pressure_column] <= 240
            and data.iloc[i-1]['Offline'] == 0 and data.iloc[i]['Offline'] == 0):
            # Calculate the time difference between the two rows in hours
            time_diff = (data.iloc[i]['Date'] - data.iloc[i-1]['Date']).total_seconds() / 3600.0
            # Calculate the pressure difference between the two rows
            pressure_diff = data.iloc[i-1][trailer_pressure_column] - data.iloc[i][trailer_pressure_column]
            # Calculate the burn rate as the pressure drop per hour
            burn_rate = pressure_diff / time_diff
            burn_rates.append(burn_rate)
    # If no such instances are found, return a placeholder value or None
    return np.mean(burn_rates) if burn_rates else np.nan


def get_remaining_fuel(current_pressure, burn_rate):
    # If burn_rate is 'N/A' or 0, return 'N/A'
    if burn_rate == 'N/A' or burn_rate == 0:
        return 'N/A'
    
    # Calculate and return the remaining fuel in hours
    return current_pressure / burn_rate


def get_status(data):
    # Get the most recent 'Offline' value
    most_recent_offline_value = data['Offline'].iloc[-1] if not data.empty else None
    
    # Determine the status and color based on the most recent 'Offline' value
    if most_recent_offline_value is not None:
        status = 'Offline' if most_recent_offline_value == 1 else 'Online'
        color = 'red' if most_recent_offline_value == 1 else 'green'
        return status, color
    else:
        return 'Unknown', 'grey'
    

def get_latest_non_zero_pressure(data, trailer_pressure_column):
    for pressure in data[trailer_pressure_column][::-1]:  # Iterating backwards
        if pressure != 0:
            return pressure
    return None


# Setup
DATABASE_URL = st.secrets["database"]["DB_URL"]
engine = create_engine(DATABASE_URL)
st.set_page_config(page_title="SPL Trailer Pressure Readings", page_icon= ":bar_chart:", layout="wide")

# Load the data
query = "SELECT * FROM trailer_data"
data = load_data(query, engine)

data = format_data(data)


# Get the latest date and time
latest_date = data['Date'].max()
latest_time = data.loc[data['Date'] == latest_date, 'Time'].max()


if authenticate_user():

    st.title(st.secrets["user"]["Title"])
    st.write(f"**Last updated {latest_date} at {latest_time}**")
    st.write("*Data updated every 15 minutes*")

    if st.button('Refresh Data'):
        data = load_data(query, engine)
        st.success('Data refreshed successfully!')


    # Display the Data in Streamlit
    with st.expander("Data Log"):
        try:
            data = pd.read_sql(query, engine)
            # Format data
            data = format_data(data)
            st.dataframe(data.tail(20))
        except Exception as e:
            st.error(f"Error: {e}")
    #st.markdown("<div style='padding: 50px;'></div>", unsafe_allow_html=True)  # Adjust the padding as needed

    # Define the pages in the app
    pages = {
        "Dashboard 📊": "dashboard",
        "Analysis 📈": "analysis"
    }

    # Get the current selected page from the query params
    query_params = st.experimental_get_query_params()
    current_page = query_params.get("page", ["dashboard"])[0]

    # Display the tabs in the sidebar and let the user select
    selected_page = st.sidebar.radio("Choose a page:", options=list(pages.keys()), index=list(pages.values()).index(current_page))

    # Set the query param for the selected page
    if selected_page != current_page:
        st.experimental_set_query_params(page=pages[selected_page])

    # Display the selected page
    if selected_page == "Dashboard 📊":

        st.markdown("<div style='text-align: center;'><h1>Dashboard 📊</h1></div>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 10px;'></div>", unsafe_allow_html=True) #Padding under title


        # Get the most recent non-zero pressure reading
        recent_trailer_1_pressure = get_latest_non_zero_pressure(data, 'Trailer_1_Pressure')
        recent_trailer_2_pressure = get_latest_non_zero_pressure(data, 'Trailer_2_Pressure')
        recent_trailer_3_pressure = get_latest_non_zero_pressure(data, 'Trailer_3_Pressure')


        # Calculate 'last_time_full' for each trailer
        last_time_full_1 = get_last_time_full(data, 'Trailer_1_Pressure')
        last_time_full_2 = get_last_time_full(data, 'Trailer_2_Pressure')
        last_time_full_3 = get_last_time_full(data, 'Trailer_3_Pressure')

        # Calculate 'current_burn_rate' for each trailer
        current_burn_rate_1 = get_current_burn_rate(data, 'Trailer_1_Pressure')
        current_burn_rate_2 = get_current_burn_rate(data, 'Trailer_2_Pressure')
        current_burn_rate_3 = get_current_burn_rate(data, 'Trailer_3_Pressure')

        # Calculate 'remaining_fuel' for each trailer
        remaining_fuel_1 = get_remaining_fuel(recent_trailer_1_pressure, current_burn_rate_1)
        remaining_fuel_2 = get_remaining_fuel(recent_trailer_2_pressure, current_burn_rate_2)
        remaining_fuel_3 = get_remaining_fuel(recent_trailer_3_pressure, current_burn_rate_3)

        # Calculate 'status' and 'color' for each trailer
        status, color = get_status(data)

        # Convert 'Time' to a recognizable time format
        data['Time'] = data['Time'].apply(lambda x: str(x).split(' ')[2] if 'days' in str(x) else str(x))
        # Concatenate 'Date' and 'Time' and convert to 'Datetime'
        data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
        # Filter DataFrame where 'Offline' is '0' to not plot 'Offline' points
        filtered_data_1 = data[(data['Offline'] == 0) & (data["Trailer_1_Pressure"] != 0)]
        filtered_data_2 = data[(data['Offline'] == 0) & (data["Trailer_2_Pressure"] != 0)]
        filtered_data_3 = data[(data['Offline'] == 0) & (data["Trailer_3_Pressure"] != 0)]

        # Display info for each trailer
        st.title("Trailer 1")
        display_trailer_info(
            1, 
            recent_trailer_1_pressure, 
            create_line_chart(filtered_data_1, 'Trailer_1_Pressure'), 
            last_time_full_1, 
            current_burn_rate_1, 
            remaining_fuel_1, 
            status
        )

        st.title("Trailer 2")
        display_trailer_info(
            2, 
            recent_trailer_2_pressure, 
            create_line_chart(filtered_data_2, 'Trailer_2_Pressure'), 
            last_time_full_2, 
            current_burn_rate_2, 
            remaining_fuel_2, 
            status
        )

        st.title("Trailer 3")
        display_trailer_info(
            3, 
            recent_trailer_3_pressure, 
            create_line_chart(filtered_data_3, 'Trailer_3_Pressure'), 
            last_time_full_3, 
            current_burn_rate_3, 
            remaining_fuel_3, 
            status
        )

    elif selected_page == "Analysis 📈":
        st.markdown("<div style='text-align: center;'><h1>Analysis 📈</h1></div>", unsafe_allow_html=True)

        data = load_data(query, engine)
        data = format_data(data)

        col1, col2 = st.columns(2)

        # Convert Date and Time columns to string type if they are not already
        data['Date'] = data['Date'].astype(str)
        data['Time'] = data['Time'].astype(str) 
        # Combine Date and Time into a single datetime column
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

        
        def plot_avg_temp_over_time(data):
            fig = px.line(data, x='Date', y='Temperature', title='Temperature Over Time')
            fig.update_layout(xaxis_title='Date', yaxis_title='Average Temperature')
            return fig
    

        def get_all_burn_rates(data, trailer_pressure_column):
            # Ensure 'Date' is in datetime format
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

            # Sort the DataFrame by Date in ascending order
            data = data.sort_values(by='Date')
            
            all_burn_rates = []
            
            for i in range(len(data)):
                if (data.iloc[i][trailer_pressure_column] <= 240 and 
                    data.iloc[i][trailer_pressure_column] != 0 and
                    data.iloc[i]['Offline'] != 1):

                    high_pressure_idx = data.loc[:i][(data[trailer_pressure_column] >= 3000) & (data['Offline'] != 1)].index.max()
                    
                    if np.isnan(high_pressure_idx):
                        continue
                    
                    time_diff = (data.loc[i, 'DateTime'] - data.loc[high_pressure_idx, 'DateTime']).total_seconds() / 3600.0
                    
                    # Ensure time_diff is non-zero before calculating burn_rate
                    if time_diff == 0:
                        burn_rate = np.nan  # or handle as preferred
                    else:
                        pressure_diff = data.loc[high_pressure_idx, trailer_pressure_column] - data.loc[i, trailer_pressure_column]
                        burn_rate = pressure_diff / time_diff
                    
                    all_burn_rates.append((data.loc[i, 'Date'], burn_rate))
            

            return all_burn_rates

    
        def get_longest_offline_duration(data):
            # Ensure the DataFrame is sorted by Date and Time
            data = data.sort_values(by=['Date', 'Time'])
            
            # Initialize variables
            max_duration = pd.Timedelta(seconds=0)
            offline_start_time = None
            
            # Iterate over the rows of the DataFrame
            for i, row in data.iterrows():
                # Check if the current entry is offline
                if row['Offline'] == 1:
                    # If this is the first offline entry in a sequence, record the start time
                    if offline_start_time is None:
                        offline_start_time = pd.to_datetime(f"{row['Date']} {str(row['Time']).replace(' 0 days', '')}")
                # If the current entry is online and there was an offline period before, calculate the duration
                elif offline_start_time is not None:
                    offline_end_time = pd.to_datetime(f"{row['Date']} {str(row['Time']).replace(' 0 days', '')}")
                    current_duration = offline_end_time - offline_start_time
                    # Update max_duration if the current_duration is longer
                    if current_duration > max_duration:
                        max_duration = current_duration
                    # Reset the offline start time
                    offline_start_time = None
            
            # Convert max_duration to hours and return it
            return max_duration.total_seconds() / 3600.0
   

        def create_scatter_plot(merged_data):
            fig = px.scatter(merged_data, x='BurnRate', y='Temperature', color='Trailer', title='Burn Rate vs Average Temperature')
            fig.update_layout(xaxis_title='Burn Rate', yaxis_title='Average Temperature', showlegend = True)
            st.plotly_chart(fig, use_container_width = True)

        # Calculate average temperature
        avg_temp = data.groupby('Date')['Temperature'].mean().reset_index()

        # Initialize an empty DataFrame for merged data
        all_data = pd.DataFrame()

        # Combine Date and Time into a single datetime column
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

        with col1:
            for i, trailer_pressure_column in enumerate(['Trailer_1_Pressure', 'Trailer_2_Pressure', 'Trailer_3_Pressure'], 1):
                burn_rates = get_all_burn_rates(data, trailer_pressure_column)
                burn_rate_df = pd.DataFrame(burn_rates, columns=['Date', 'BurnRate'])
                # Ensure 'Date' is in datetime format
                burn_rate_df['Date'] = pd.to_datetime(burn_rate_df['Date'])
                avg_temp['Date'] = pd.to_datetime(avg_temp['Date'])  # Ensure avg_temp 'Date' is also in datetime format
                merged_data = pd.merge(burn_rate_df, avg_temp, on='Date', how='inner')
                merged_data['Trailer'] = f'Trailer {i}'
                all_data = pd.concat([all_data, merged_data], ignore_index=True)

            # Scatter Plot
            create_scatter_plot(all_data)

            # Line Graphs
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Burn Rate Over Time', 'Avg Temp Over Time'))

            # Line Graph (Burn Rate over Time)
            for trailer in all_data['Trailer'].unique():
                filtered_data = all_data[all_data['Trailer'] == trailer]
                fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['BurnRate'], name=trailer), row=1, col=1)

            # Line Graph (Avg Temperature Over Time)
            avg_temp_per_day = data.groupby('Date')['Temperature'].mean().reset_index()
            fig.add_trace(go.Scatter(x=avg_temp_per_day['Date'], y=avg_temp_per_day['Temperature'], name='Avg Temp'), row=2, col=1)

            fig.update_layout(height=600, width=800, title_text='Combined Graphs')
            st.plotly_chart(fig, use_container_width=True)  


        with col2:
            # Calculate the longest time spent offline
            longest_offline_duration = get_longest_offline_duration(data)

            # Display the summary stats
            st.markdown(f"<h3 style='text-align: center;'>Summary Stats</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Longest Time Spent Offline: {longest_offline_duration:.2f} hours</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # List of trailer pressure column names
            trailer_columns = ['Trailer_1_Pressure', 'Trailer_2_Pressure', 'Trailer_3_Pressure']

            # Loop through each trailer
            for trailer_column in trailer_columns:
                # Get all burn rates
                all_burn_rates = get_all_burn_rates(data, trailer_column)
                st.markdown(f"<h5 style='text-align: center;'>{trailer_column.replace('_', ' ')}</h4>", unsafe_allow_html=True)
                
                # Check if there are any burn rates to analyze
                if all_burn_rates:
                    # Find the minimum and maximum burn rates along with their corresponding dates
                    min_burn_rate, min_burn_rate_date = min(all_burn_rates, key=lambda x: x[1])
                    max_burn_rate, max_burn_rate_date = max(all_burn_rates, key=lambda x: x[1])
                    
                    # Convert potential string dates to datetime, if not already in datetime format
                    if not isinstance(min_burn_rate_date, pd.Timestamp):
                        min_burn_rate_date = pd.to_datetime(min_burn_rate_date, errors='coerce')
                    if not isinstance(max_burn_rate_date, pd.Timestamp):
                        max_burn_rate_date = pd.to_datetime(max_burn_rate_date, errors='coerce')
                    
                    # Generate and display formatted date strings or 'N/A' if invalid
                    min_date_str = min_burn_rate_date.strftime('%Y-%m-%d') if pd.notnull(min_burn_rate_date) else 'N/A'
                    max_date_str = max_burn_rate_date.strftime('%Y-%m-%d') if pd.notnull(max_burn_rate_date) else 'N/A'
                    
                    # Display the trailer name, minimum, and maximum burn rates in Streamlit
                    st.markdown(f"<p style='text-align: center;'>Minimum burn rate: {min_burn_rate:.2f} on {min_date_str}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>Maximum burn rate: {max_burn_rate:.2f} on {max_date_str}</p>", unsafe_allow_html=True)
                    #print("Min burn rate:", min_burn_rate, "on", min_burn_rate_date)
                    #print(min(all_burn_rates, key=lambda x: x[1]))
                    #print(max(all_burn_rates, key=lambda x: x[1]))


                else:
                    st.markdown(f"<p style='text-align: center;'>No burn rate data available for {trailer_column.replace('_', ' ')}.</p>", unsafe_allow_html=True)

                # Add a separator line between different trailers' data
                st.write("---")