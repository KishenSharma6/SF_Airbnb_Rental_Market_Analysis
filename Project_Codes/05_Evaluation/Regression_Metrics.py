import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def regression_metrics(name,y_hat,y):
    mae = mean_absolute_error(y, y_hat)
    mse = np.mean(np.square((y - y_hat)))
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    print('{} Evaluation Metrics:'.format(name))
    print('Adjusted R2: {}\nMAE: {} \nMSE: {}\nRMSE: {}'.format(r2,mae,mse,rmse))
    print('==================================================')