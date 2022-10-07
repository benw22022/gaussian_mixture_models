"""
Callbacks
___________________________________________________
Functions to configure callbacks for training
"""

import logger
log = logger.get_logger(__name__)
import os
import time
import datetime
import keras
import tensorflow as tf
from omegaconf import DictConfig
from typing import List

class LoggingCallback(keras.callbacks.Callback):
    """
    Simple callback to log log train / val metrics to logging
    Regular tensorflow printouts are not logged
    """

    time_start = time.time()
    time_epoch_start = 0

    def on_train_begin(self, logs=None):
        log.info("Starting training")

    def on_train_end(self, logs=None):
        log.info("Training stopped")

    def on_epoch_begin(self, epoch, logs=None):
        log.info(f"Start epoch {epoch + 1}")
        self.time_epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print("\n") # print newline to prevent logging on same line as pbar
        time_epoch_taken = time.time() - self.time_epoch_start
        time_taken = time.time() - self.time_start
        time_epoch_taken = datetime.timedelta(seconds=time_epoch_taken)
        time_taken = datetime.timedelta(seconds=time_taken)

        log.info(f"Epoch {epoch + 1} took {time_epoch_taken}")
        log.info(f"End epoch {epoch + 1}: Time elapased so far = {time_taken}")
        for key, value in logs.items():
            log.info(f"{key}: {value:2.4f}")


def configure_callbacks(config: DictConfig, **kwargs) -> List[keras.callbacks.Callback]:
    """
    Parses config files to configure callbacks
    args:
        config: DictConfig - Hydra config object
    returns:
        List[keras.callbacks.Callback] - A list of tensorflow callbacks
    """
    callbacks = []
    
    if config.callbacks.early_stopping.enabled:
        min_delta = config.callbacks.early_stopping.min_delta
        patience=config.callbacks.early_stopping.patience

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=min_delta,
            patience=patience, verbose=0, restore_best_weights=True)

        log.info("Enabling early stopping")
        callbacks.append(early_stopping)

    if config.callbacks.model_checkpoint.enabled:
        weights_save_dir = kwargs.get("weights_save_dir", 'network_weights')
        os.makedirs(weights_save_dir, exist_ok=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(weights_save_dir, 'weights-{epoch:02d}.h5'),
                                                    monitor="val_loss", save_best_only=True, save_weights_only=True)
        log.info("Enabling model checkpointing")                                                    
        callbacks.append(model_checkpoint)                                                

    if config.callbacks.lr_schedd.enabled:
        factor = config.callbacks.lr_schedd.factor
        patience = config.callbacks.lr_schedd.patience
        min_lr = config.callbacks.lr_schedd.min_lr

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)
        log.info("Enabling learing rate decay")
        callbacks.append(reduce_lr)

    if config.callbacks.tensorboard.enabled:
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq = 1)
        log_dir = os.path.join(os.getcwd(), 'logs')
        log.info(f"Enabling tensorboard, to start run: tensorboard --logdir={log_dir}")
        callbacks.append(tensorboard_callback)

    if config.callbacks.logging.enabled:
        log.info("Enabling training logging")
        callbacks.append(LoggingCallback())

    return callbacks