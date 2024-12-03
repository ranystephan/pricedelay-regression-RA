import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta, datetime


# set option to display precision to 10 decimal places
pd.set_option('display.precision', 10)

df_crypto = pd.read_csv('listoneprice22(1).csv')
df_index = pd.read_csv('crix_index(1).csv')


df_crypto['Date'] = pd.to_datetime(df_crypto['Date'])
df_index['date'] = pd.to_datetime(df_index['date'])



# Rename  columns date to match
df_index.rename(columns={'date': 'Date'}, inplace=True)

# merge the dataframces on the 'Date' column
df = pd.merge(df_crypto, df_index, on='Date', how='inner')

df.to_csv('merged.csv', index=False)


# rename columns to match code
df.rename(columns={'Date': 'Date', 'close': 'crypto_prices', 'price_index': 'market_prices'}, inplace=True)
df.to_csv('merged.csv', index=False)


"""
Description: This function calculates the compounded weekly returns for a given time series of daily prices.
Parameters:
    daily_prices: pd.Series - A time series of daily prices

Note:
    This function will take in the daily prices and calculate the daily returns.
    It will then resample the daily returns to weekly, with weeks ending on Wednesday.
    The weekly returns are then compounded to get the compounded weekly returns.
"""

def calculate_compounded_returns(daily_prices):
    # Calculate daily returns
    daily_returns = daily_prices.pct_change().add(1)

    # Resample to weekly, with weeks ending on Wednesday
    # Calculate the product of returns for each week to get compounded return
    weekly_compounded_returns = daily_returns.resample('W-WED').prod().sub(1)

    return weekly_compounded_returns


# Putting this into a function
# adding weekly returns
def add_weekly_returns(df):
    

    # Convert 'Date' to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Lists to store the results
    results = []
    
    # Loop through each crypto in the dataframe
    for code, group in df.groupby('code'):
        # Calculate compounded returns for each group
        group['crypto_weekly_returns'] = calculate_compounded_returns(group['crypto_prices'])
        group['market_weekly_returns'] = calculate_compounded_returns(group['market_prices'])
        
        results.append(group)
        
    # Concatenate the results and reset the index
    df = pd.concat(results)
    
    return df


# Add weekly returns
df = add_weekly_returns(df)

df.to_csv('test.csv', index=False)


# Drop na values
df.dropna(inplace=True)
df.to_csv('test.csv', index=False)


def calculate_lagged_returns(df):
    results = []

    # Group by 'code' and calculate lagged returns for each group
    for code, group in df.groupby('code'):
        # Calculate Lagged Returns
        for n in range(1, 5):
            group[f'market_returns_lag_{n}'] = group['market_weekly_returns'].shift(n)

        # Remove rows with NaN values which result from lagging
        group.dropna(inplace=True)

        # Append the group to the results list
        results.append(group)

    # Concatenate all the groups back into a single DataFrame
    df = pd.concat(results)

    # Save to CSV if needed
    df.to_csv('data_with_lags.csv')

    return df


# Calculate lagged returns
df = calculate_lagged_returns(df)


df.reset_index(drop=False, inplace=True)


# df = pd.DataFrame({'Date': dates, 'crypto_ID': crypto_ids, 'crypto_prices': crypto_prices, 
#                    'market_prices': market_prices, 'crypto_weekly_returns': crypto_returns, 
#                    'market_returns_lag_1': market_lag1, 'market_returns_lag_2': market_lag2, 
#                    'market_returns_lag_3': market_lag3, 'market_returns_lag_4': market_lag4})

def calculate_annual_delay_measures(df):
    results = []

    # Assuming df is your DataFrame and 'Date' is in a format that pandas can recognize as a date
    df['Date'] = pd.to_datetime(df['Date'])

    # Function to adjust the year
    def adjust_year(row):
        if row.month < 7:  # If month is before July
            return row.year - 1  # Consider it as part of the previous year
        else:
            return row.year

    # First group by cryptocurrency
    for code, crypto_group in df.groupby('code'):
        # Convert 'Date' to datetime and apply the year adjustment within each crypto group
        crypto_group['Date'] = pd.to_datetime(crypto_group['Date'])
        crypto_group['Year'] = crypto_group['Date'].apply(adjust_year)

        # Now group by year within each crypto group
        for year, year_group in crypto_group.groupby('Year'):
            
            # Skip adjusted years with less than 52 weeks
            if len(year_group) < 52:
                continue
            
            
            # Prepare regression variables
            X = sm.add_constant(year_group[['market_weekly_returns', 'market_returns_lag_1', 'market_returns_lag_2', 'market_returns_lag_3', 'market_returns_lag_4']])
            y = year_group['crypto_weekly_returns']

            # Full regression model
            full_model = sm.OLS(y, X).fit()
            r_squared_full = full_model.rsquared

            # Restricted regression model (only contemporaneous market return)
            X_restricted = sm.add_constant(year_group[['market_weekly_returns']])
            restricted_model = sm.OLS(y, X_restricted).fit()
            r_squared_restricted = restricted_model.rsquared

            # Calculate D1
            D1 = 1 - (r_squared_restricted / r_squared_full)

            # Append results for each year and crypto
            results.append({'code': code, 'Year': year, 'D1': D1})
            
    return pd.DataFrame(results)

# Usage example
annual_delay_measures = calculate_annual_delay_measures(df)
print(annual_delay_measures)

# export to csv

df.to_csv('final_df_after_computation.csv', index=False)

annual_delay_measures.to_csv('annual_delay_measures.csv', index=False)



