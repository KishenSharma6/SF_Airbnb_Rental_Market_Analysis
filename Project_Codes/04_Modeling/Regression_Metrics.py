import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def regression_metrics(name,y_hat, y,return_dict = True):
    """
    Prints regression metrics given:
    name: model name as a string
    y_hat: predictions
    y = actual
    
    Also returns dictionary of metrics if return_dict = True
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