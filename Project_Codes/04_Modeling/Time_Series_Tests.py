#Import Libraries
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import pandas as pd
import numpy as np

def ADF_KPSS_test(series, periods=0):
    """
    Function
        - Runs ADF and KPSS test on a stationary series and prints the results of each test
    
    Arguments
        - Series: Time series data you would like to test for stationrity
        - Periods: If your data has NA's, you can specify which rows to slice series(like so series[periods:])
    """
    #Convert series to values and subset to remove NA values
    X = series.values[periods:]
    
    #Implement ADF and KPSS
    adf_results = adfuller(X)
    kpss_results = kpss(X)
    
    #Print results
    print('ADF Statistic: {}'.format(adf_results[0]))
    print('P-Value: {}'.format(adf_results[1]))
    for k,v in adf_results[4].items():
        print('Critical Values {} : {}'.format(k,round(v,4)))
    print('===============================================')
    print('KPSS Statistic: {}'.format(kpss_results[0]))
    print('P-Value: {}'.format(kpss_results[1]))
    for k,v in kpss_results[3].items():
        print('Critical Values {} : {}'.format(k,round(v,4)))
    print('===============================================\n')
    
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag = 12):    
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. 
    
    Arguments
    data: pandas dataframe containing the time series variables
    variables: list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def cointegration_test(df, alpha = .05):
    """
    Implement cointegration test on multivariate time series
    """
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary stats
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

    




