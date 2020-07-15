import pandas as pd
import numpy as np

def missing_calculator(df,data_type = None):
    """
    The following function takes a data frame and returns a dataframe with the following statistics:
        data_type takes a string or list of strings referenceing data type
        missing_count: Number of missing values per column 
        missing_%: Returns the % of total values missing from the column
    The function also returns a list of columns that containg
    """
    if data_type == None:
        missing = pd.DataFrame(index=df.isna().sum().index)
        missing['count'] = df.isna().sum()
        missing['percentage'] = (missing['count']/len(df)) * 100
        missing = missing[missing['percentage'] > 0].sort_values(by = 'percentage', ascending = False)
       
    else:
        missing = pd.DataFrame(index=df.select_dtypes(data_type).isna().sum().index)
        missing['count'] = df.isna().sum()
        missing['percentage'] = (missing['count']/len(df)) * 100
        missing = missing[missing['percentage'] > 0].sort_values(by = 'percentage', ascending = False)
    return(missing)