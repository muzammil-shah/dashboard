import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide")

# Hide the default Streamlit header
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Add your custom header
st.markdown(
    """
    <h1 style="text-align: center;">Store Sales Analysis by Products</h1>
    """,
    unsafe_allow_html=True
)


def dashboard():
    # Load and parse CSV files
    sales_df = pd.read_csv('sample.csv')
    predicted_df = pd.read_csv('predicted_sales.csv')

    # Convert dates to datetime
    sales_df['Transaction Date'] = pd.to_datetime(sales_df['Transaction Date']).dt.date
    predicted_df['Transaction Date'] = pd.to_datetime(predicted_df[['Year', 'Month']].assign(DAY=10))

    # Sidebar filters for sales data
    st.sidebar.title("Sales Data Filters")
    categories = sales_df['Category'].unique()
    selected_category = st.sidebar.selectbox("Select Category", categories)
    products = sales_df[sales_df['Category'] == selected_category]['Product Name'].unique()
    selected_product = st.sidebar.selectbox("Select Product", products)

    # Date range slider for sales data
    min_date = sales_df['Transaction Date'].min()
    max_date = sales_df['Transaction Date'].max()
    start_date, end_date = st.slider(
        "Select Date Range to see Data",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter sales data
    filtered_sales = sales_df[
        (sales_df['Category'] == selected_category) &
        (sales_df['Product Name'] == selected_product) &
        (sales_df['Transaction Date'] >= start_date) &
        (sales_df['Transaction Date'] <= end_date)
    ]

    # Summary metrics for sales data
    total_sales = filtered_sales['Total Amount'].sum()
    total_quantity = filtered_sales['Quantity Sold'].sum()
    avg_sales_per_day = filtered_sales.groupby('Transaction Date')['Total Amount'].sum().mean()

    st.subheader(f"Sales Summary for {selected_product} in {selected_category}")
    st.metric("Total Sales", f"${total_sales:,.2f}")
    st.metric("Total Quantity Sold", f"{total_quantity:,}")
    st.metric("Average Sales Per Day", f"${avg_sales_per_day:,.2f}")

    # Sales over time line chart
    st.subheader("Sales Over Time")
    sales_over_time = filtered_sales.groupby('Transaction Date')['Total Amount'].sum()
    st.line_chart(sales_over_time)

    # Quantity sold over time line chart
    st.subheader("Quantity Sold Over Time")
    quantity_over_time = filtered_sales.groupby('Transaction Date')['Quantity Sold'].sum()
    st.line_chart(quantity_over_time)

    # Top selling days bar chart
    st.subheader("Top Selling Days")
    top_days = sales_over_time.nlargest(10)
    st.bar_chart(top_days)

    # Display filtered sales data
    st.subheader("Filtered Sales Data")
    st.write("You can scroll the table")
    st.write(filtered_sales)

    # Sidebar filters for predicted sales data
    st.sidebar.header('Sales Prediction Filters')
   
    product_name = st.sidebar.selectbox('Select Product Name for Prediction', predicted_df[predicted_df['Category'] == selected_category]['Product Name'].unique())
    
    st.header("Sales Prediction Data ")
    date_range = st.selectbox('Select Date Range for Prediction', ['1 month', '3 months', '6 months', '1 year'])

    # Filter predicted sales data based on user input
    filtered_predicted = predicted_df[
        (predicted_df['Category'] == selected_category) &
        (predicted_df['Product Name'] == product_name)
    ]

    # Date range filter for predicted data
    if date_range == '1 month':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
    elif date_range == '3 months':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=3)
    elif date_range == '6 months':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    else:  # 1 year
        start_date = pd.Timestamp.now() - pd.DateOffset(years=1)

    filtered_predicted = filtered_predicted[filtered_predicted['Transaction Date'] >= start_date]

    # Plotting predicted sales
    st.subheader(f'Predicted Sales for {product_name} in {selected_category}')
    fig, ax = plt.subplots()
    ax.plot(filtered_predicted['Transaction Date'], filtered_predicted['Predicted Sales'], marker='o', label='Predicted Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Sales')
    ax.set_title(f'Predicted Sales Over Time for {product_name}')
    ax.legend()
    st.pyplot(fig)

    # Display predicted sales data
    st.write(f'Predicted Sales Data for {product_name} in {selected_category}:')
    st.dataframe(filtered_predicted)


params = st.query_params


action = params.get('action')

if action == 'dashboard':
    dashboard()

else:
    st.write("Go Select the city from the map")