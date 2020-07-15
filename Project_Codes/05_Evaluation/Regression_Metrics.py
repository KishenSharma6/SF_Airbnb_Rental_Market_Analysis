import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def regression_metrics(name,training_data, y_hat,y):
    mae = mean_absolute_error(y, y_hat)
    mse = np.mean(np.square((y - y_hat)))
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1-r2)*(len(y_hat)-1)/(len(y_hat)-training_data.shape[1]-1)
    print('{} Evaluation Metrics:'.format(name))
    print('Adjusted R2: {}\nMAE: {} \nMSE: {}\nRMSE: {}'.format(adj_r2,mae,mse,rmse))
    print('==================================================')