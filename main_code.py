import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_file(path):
    """This function will take path as a parameter
       and reads the csv file and returns two data frames one with 
       country as columns and other witgh year as columns"""
    ds_raw = pd.read_csv(filepath_or_buffer=path, sep=',',
                         encoding='cp1252', skip_blank_lines=True)
    #Deleting rows with null values
    ds_raw = ds_raw.dropna()
    pd_countries = pd.DataFrame(ds_raw)
    #calling yearly function inorder to get the years as column
    pd_years = yearly_data(pd_countries)
    out = [pd_countries, pd_years]
    return out


def df_info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary"""
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))

def yearly_data(df):
    """This function takes data frame with years as columns and \
        converts country as columns and years as rows"""
    #slicing the df into only years columns
    y = df.loc[:,'2006':'2014']
    #changing values type from object to float
    y = y.astype(float)
    y['countries'] = df['Country Name']
    #transposing the data frame.
    y = y.T
    y.rename(columns = y.iloc[-1], inplace = True)
    y = y.drop(y.index[-1])
    #reset the index and making index as year column
    y = y.reset_index()
    y = y.rename(columns={'index':'year'})
    return y
