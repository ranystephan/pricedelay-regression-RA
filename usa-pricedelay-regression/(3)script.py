import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta, datetime

#################################
# FUNCTIONS
#################################
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


######################################################


"""
Description: This function calculates the compounded weekly returns for a given time series of daily prices.
Parameters:
    df: pd.DataFrame - A DataFrame containing daily prices for multiple stocks

Note:
    This function will calculate the compounded weekly returns for each stock in the DataFrame.
    It will group the DataFrame by 'code' and calculate the compounded weekly returns for each group.
    The weekly returns are then added to the original DataFrame.
"""
def add_weekly_returns(df):
    # Convert 'Date' to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Lists to store the results
    results = []
    
    # Loop through each stock in the dataframe
    for code, group in df.groupby('code'):
        # Calculate compounded returns for each group
        group['stock_weekly_returns'] = calculate_compounded_returns(group['stock_price'])
        group['market_weekly_returns'] = calculate_compounded_returns(group['sp500_price'])
        
        results.append(group)
        
    # Concatenate the results and reset the index
    df = pd.concat(results)
    
    return df


######################################################


"""
Description: This function calculates the lagged returns for each stock in the DataFrame.
Parameters:
    df: pd.DataFrame - A DataFrame containing weekly returns for multiple stocks

Note:
    This function will calculate the lagged returns for each stock in the DataFrame.
    It will group the DataFrame by 'code' and calculate the lagged returns for each group.
    The lagged returns are then added to the original DataFrame.
    The resulting DataFrame is then saved to a CSV file.
"""
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
    df.to_csv(f'(3)/data_process/chunk_{chunk}_data_with_lags.csv')

    return df

######################################################


"""
Description: This function calculates the annual delay measures for each stock in the DataFrame.
Parameters:
    df: pd.DataFrame - A DataFrame containing weekly returns and lagged returns for multiple stocks
    
Returns:
    pd.DataFrame - A DataFrame containing the annual delay measures for each stock
    
Note:
    This function calculates the annual delay measures for each stock in the DataFrame.
    It groups the DataFrame by 'code' and calculates the annual delay measures for each group.
    The annual delay measures are then returned as a DataFrame.
"""
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
    for code, stock_group in df.groupby('code'):
        # Convert 'Date' to datetime and apply the year adjustment within each crypto group
        stock_group['Date'] = pd.to_datetime(stock_group['Date'])
        stock_group['Year'] = stock_group['Date'].apply(adjust_year)

        # Now group by year within each crypto group
        for year, year_group in stock_group.groupby('Year'):
            
            # Skip adjusted years with less than 52 weeks
            if len(year_group) < 24:
                continue
            
            
            # Prepare regression variables
            X = sm.add_constant(year_group[['market_weekly_returns', 'market_returns_lag_1', 'market_returns_lag_2', 'market_returns_lag_3', 'market_returns_lag_4']])
            y = year_group['stock_weekly_returns']

            # Full regression model
            full_model = sm.OLS(y, X).fit()
            r_squared_full = full_model.rsquared

            # Restricted regression model (only contemporaneous market return)
            X_restricted = sm.add_constant(year_group[['market_weekly_returns']])
            restricted_model = sm.OLS(y, X_restricted).fit()
            r_squared_restricted = restricted_model.rsquared

            # Calculate D1
            D1 = 1 - (r_squared_restricted / r_squared_full)

            # Append results for each year and stock, also include the ticker, company name, and year
            results.append({
                'code': code,
                'ticker': year_group['ticker'].iloc[0],
                'company_name': year_group['company_name'].iloc[0],
                'year': year,
                'D1': D1
            })
            
            
    return pd.DataFrame(results)

#################################
# MAIN
#################################

CHUNKS = 38

for chunk in range(CHUNKS + 1):

  df = pd.read_csv(f'(3)/data_chunks/(3)data_{chunk}.csv')

  # only keep the columns we need
  df = df[['GVKEY', 'LPERMNO', 'tic', 'conm', 'datadate', 'prccd']]

  # make sure the dates are in the right format
  df['datadate'] = pd.to_datetime(df['datadate'])

  # sort by GVKEY and datadate
  df = df.sort_values(['GVKEY', 'datadate'])

  # reset the index
  df = df.reset_index(drop=True)

  spx = pd.read_csv('sp500_daily_prices.csv')

  spx = spx[['Date', 'Close']]

  # make sure the dates are in the right format
  spx['Date'] = pd.to_datetime(spx['Date'])

  # merge df and spx on datadate and Date
  merged = pd.merge(df, spx, how='left', left_on='datadate', right_on='Date')

  # merge the columns GVKEY and LPERMNO in the format 'GVKEY_LPERMNO' as 'id' 
  merged['id'] = merged['GVKEY'].astype(str) + '_' + merged['LPERMNO'].astype(str)

  # delete the columns GVKEY and LPERMNO
  merged = merged.drop(['GVKEY', 'LPERMNO'], axis=1)

  merged = merged.drop(['datadate'], axis=1)

  # rename the columns
  merged = merged.rename(columns={'id': 'code', 'tic': 'ticker', 'conm': 'company_name', 'prccd': 'stock_price', 'Close': 'sp500_price'})


  df_temp = add_weekly_returns(merged)
  # drop na values
  df_temp.dropna(inplace=True)
  df_temp.to_csv(f'(3)/data_process/chunk_{chunk}_data_weekly_returns.csv', index=False)


  # Calculate lagged returns
  df_temp = calculate_lagged_returns(df_temp)


  df_temp.reset_index(drop=False, inplace=True)



  # Usage example
  annual_delay_measures = calculate_annual_delay_measures(df_temp)


  annual_delay_measures.to_csv(f'(3)/data_process/chunk_{chunk}_annual_delay_measures.csv', index=False)
  
  print(f'Chunk {chunk} done!')