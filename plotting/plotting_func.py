"""
Plotting functions
"""

import logger
log = logger.get_logger(__name__)
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_history(obs: str, history: tf.keras.callbacks.History, saveas: str=None) -> None:
    """
    Function to plot loss vs epochs for a history object and saves it to file
    args:
        obs: str - Name of observable being trained on
        history: tf.keras.callbacks.History - Tensorflow history object returned by fit() method
        saveas: str (default=None) - filepath to save to. Will create plots/history_plots/{obs}_history.png if 
        not given
    returns:
        None
    """
    
    fig, ax = plt.subplots();
    ax.plot(history.history['loss'], label='Train');
    ax.plot(history.history['val_loss'], label='Val');
    ax.set_xlabel("Epochs");
    ax.set_ylabel("Loss");
    ax.set_title(obs)
    ax.legend();
    if saveas is None:
        os.makedirs(os.path.join('plots', 'history'), exist_ok=True)
        saveas = os.path.join('plots', 'history', f'{obs}_history.png')
    plt.savefig(saveas)