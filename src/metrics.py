import numpy as np
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt

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

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()