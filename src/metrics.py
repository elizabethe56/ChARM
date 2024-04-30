import numpy as np
from sklearn.metrics import confusion_matrix as cm

def accuracy_score(ytest : np.ndarray,
                   yhat : np.ndarray
                   ) -> float:
    '''
    Calculates the accuracy score between two lists. Divides the number of correctly labeled data by the total number of data. Usually used to calculate the 'correctness' of a classification model.

    Parameters:
        ytest: the true data labels
        yhat: the predicted data labels
    
    Returns:
        float: the accuracy score of ytest and yhat
    '''
    nz = np.flatnonzero(ytest == yhat)
    return len(nz)/len(ytest)

def confusion_matrix(ytest : np.ndarray,
                     yhat : np.ndarray
                     ) -> np.ndarray:
    return cm(ytest, yhat)