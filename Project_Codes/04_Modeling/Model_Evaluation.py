import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def regression_metrics(name,y_hat, y,return_dict = True):
    """
    Function:
        - Prints regression metrics given y_hat and y arguments
        - Will return a dictionary of metrics if return_dict = True
    Arguments:
        - name: model name as a string
        - y_hat: predictions
        - y: actual
        - return_dict: flag if you would like metrics returned as a dictionary
    """
    mae = mean_absolute_error(y, y_hat)
    mse = np.mean(np.square((y - y_hat)))
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    print('{} Evaluation Metrics:'.format(name))
    print('R2: {}\nMAE: {} \nMSE: {}\nRMSE: {}'.format(r2,mae,mse,rmse))
    print('==================================================')
    if return_dict == True:
        metrics_dict = {}
        for metric in ('r2','mae','mse','rmse'):
            metrics_dict[metric] = locals()[metric]
        return metrics_dict

def prediction_error_df(preds, actual):
    """
    Function returns:
        - Dataframe containing calculated error and error %'s'
    Arguments:
        - preds: machine learning predictions
        - actual: target
    
    """
    error_df = pd.DataFrame({'predictions':preds, 'actual':actual})
    error_df['error'] = error_df['predictions'] - error_df['actual']
    error_df['error_%'] = np.abs(error_df['error']/error_df['actual']) * 100
    return error_df


def prediction_fit_eval(prediction_error_df, figsize = (14,7), bins = 50, model_name = None, rmse=None):
    """
    Function returns:
        - Returns error distribution and prediction vs fit plot
    Arguments:
        - prediction_error_df: output of prediction_error_df 
        - figsize: figsize for plot
        - bins: bins for error histogram
        - model_name: Name of model for plot title
        - rmse: RMSE score of model for plot title
    """
    #Create figure
    f, ax = plt.subplots(1,2,figsize = figsize)
    
    #Plot title
    plt.suptitle('{} Error Evaluation'.format(model_name), y =1.02, fontsize = 18,)
    plt.figtext(s = 'Root Mean Squared Error:{}'.format(round(rmse, 2)),
                x=0.5, y = .98, fontsize = 12, ha="center", va="top")

    #Error Distribution
    prediction_error_df.error.hist(ax=ax[0], bins = bins, color = 'r', alpha = .4)
    
    #Set ax[0] aesthetics
    ax[0].set_title('Prediction Error Distribution',fontstyle = 'italic', fontsize=16,)
    ax[0].set_xlabel('Error',fontsize = 12)
    ax[0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:}".format(int(x))))
    ax[0].set_ylabel('Count',fontsize = 12)
    ax[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    #Prediction vs fit plot
    ax[1].scatter(x = prediction_error_df['actual'], y = prediction_error_df['predictions']
                  , alpha = .1)
    X=list(np.arange(0,max(prediction_error_df['actual'])))
    ax[1].plot(X,X, color = 'black', linestyle = '--', alpha = .7)
    
    #Set ax[1] aesthetics
    ax[1].set_title('Prediction vs Actual Fit',fontstyle = 'italic', fontsize=16,)
    ax[1].set_xlabel('Actual',fontsize = 12)
    ax[1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:}".format(int(x))))
    ax[1].set_ylabel('Predictions', fontsize = 12)
    ax[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:}".format(int(x))))
    
    return ax