import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st

# Set up Streamlit
st.title('Stock Sage: A Stock Price Prediction Application')

# User input for stock symbol
stock = st.text_input('Enter stock symbol:', 'GOOG')
start = '2012-01-01'
end = '2022-04-09'

if st.button("Predict"):

    if stock:
        st.write(f"Fetching data for {stock} from {start} to {end}...")
    
        # Download data
        data = yf.download(stock, start, end)

        # Define feature and target columns
        features = ['Open', 'High', 'Low']
        target = 'Close'

        # Prepare training and testing data
        x = data[features]
        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        train=data.loc[data.index<'01-01-2021']
        test=data.loc[data.index>='31-12-2020']


        # Training RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True)
        rf.fit(x_train, y_train)
        
        # Display predictions plot
        st.subheader('Prediction vs Actual Price')
        
        test_ohl=test[features]
        y_pred = rf.predict(test_ohl)
        latest_data = data.iloc[-1][features].values.reshape(1, -1)  # Get latest data for the next day
        next_day_prediction = rf.predict(latest_data)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(test.index, test[target], color='red', label='Actual Price')
        ax.plot(test.index, y_pred, color='blue', label='Predicted Price')
        ax.set_title('Original Price vs Predicted Price')
        ax.set_xlabel('Days')
        ax.set_ylabel('Close Price')
        ax.legend()
        st.pyplot(fig)

