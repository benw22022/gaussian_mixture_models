"""
Models
__________________________________________________________
Gaussian Mixture models and related functions
"""

import logger
log = logger.get_logger(__name__)
import os
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from keras.activations import softplus
from keras.layers import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate, Lambda, Reshape, Softmax
from keras.models import Model
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import Callback, EarlyStopping
import keras.backend as K
from collections import OrderedDict
from typing import Tuple
from models.callbacks import configure_callbacks
from source.stats import transform_onto_latent_space

def custom_weight_init_initial(shape: Tuple, dtype: 'str'=None, g: float=0.2) -> tf.Tensor:
    """Keras: custom weight initialiser function for leaky relu layer with gradient g (first layer in network)"""
    limit = 4. / np.sqrt(shape[0]) / (1 + g)
    return K.random_uniform(shape, -limit, limit, dtype=dtype)

def custom_weight_init_hidden (shape, dtype=None, g=0.2) :
    """Keras: custom weight initialiser function for leaky relu layer with gradient g (not first layer in network)"""
    limit = 3. / np.sqrt(shape[0]) / (1 + g)
    return K.random_uniform(shape, -limit, limit, dtype=dtype)

def add_gauss_mean_offsets (x, num_gauss, offset_min, offset_max) :
    """TF method: for input x of size [?, num_gauss], add evenly spaced offsets between [offset_min, offset_max]"""
    c = tf.convert_to_tensor([offset_min + (offset_max-offset_min)*i/(num_gauss-1.) for i in range(num_gauss)])
    return x + c    

def set_initial_gauss_sigmas (x, num_gauss, offset_min, offset_max, gauss_width_factor) :
    """TF method: for input x of size [?, num_gauss], add a constant factor which sets initial Gaussian widths as gauss_width_factor * (offset_max-offset_min) / num_gauss
       - to be applied before a Softmax function, so offset addition is performed in a logarithmic basis"""
    target_width = gauss_width_factor * float(offset_max - offset_min) / num_gauss
    offset       = float(np.log(np.exp(target_width) - 1))
    c = tf.convert_to_tensor([offset for i in range(num_gauss)])
    return x + c

def add_epsilon_to_gauss_sigmas (x, num_gauss, epsilon=1e-4) :
    """TF method: for input x of size [?, num_gauss], add epsilon to every value"""
    c = tf.convert_to_tensor([float(epsilon) for i in range(num_gauss)])
    return x + c

def add_gauss_fraction_offsets (x, num_gauss, const_frac=0.2) :
    """TF method: for input x of size [?, num_gauss], where x is a multinomial(num_gauss) probability distribution, add a constant term to prevent probabilities going to 0"""
    c = tf.convert_to_tensor([1./num_gauss for i in range(num_gauss)])
    return (1.-const_frac)*x + const_frac*c

def K_gauss_prob (x, mean, sigma) :
    """return the Gaussian probability density for datapoints x"""
    prob = K.exp(-0.5*(x - mean)*(x - mean)/(sigma*sigma)) / K.sqrt(2*np.pi*sigma*sigma)
    return prob

def K_datapoint_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    """Keras: return the probability density for datapoints x as described by the Gaussian mixture model"""
    prob = 0.
    x = x[:,0]
    for i in range(num_gauss) :
        prob = prob + gauss_fracs[:,i] * K_gauss_prob(x, gauss_means[:,i], gauss_sigmas[:,i])
    return prob

def K_datapoint_log_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    """Keras: return the log probability density for datapoints x as described by the Gaussian mixture model"""
    return K.log(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))

def K_dataset_log_likelihood (x, params, num_gauss) :
    """Keras: return the log probability density for datapoints x as described by the Gaussian mixture model"""
    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]
    return K_datapoint_log_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas)

def GMM(conf: OmegaConf, n_external_parameters, n_observables) -> tf.keras.Model:
    
    N1 = conf.A1 + conf.A2 * n_external_parameters
    N2 = conf.B1 + conf.B2 * n_observables if n_observables > 0 else 0
    
    conditions_input  = Input((n_external_parameters,))
    model_conditions  = conditions_input
    model_conditions  = Dense(N1, kernel_initializer=custom_weight_init_initial, bias_initializer='zeros', activation='relu')(model_conditions) 
    if conf.use_leaky_relu : model_conditions = LeakyReLU (0.2)      (model_conditions)
    if conf.batch_norm     : model_conditions = BatchNormalization() (model_conditions)
    if conf.dropout > 0.   : model_conditions = Dropout(conf.dropout)     (model_conditions)
    
    #
    #  If they exist, create an input layer for other input observables
    #  -  if configured, add a layer which transforms these inputs onto the given domain
    #  -  add a layer to process just these inputs
    #  -  concatenate the resulting hidden layer with that from the external parameter dependence
    #  If they don't exist, skip this step
    #
    
    if n_observables > 0 :
        observables_input = Input((n_observables,))
        model_observables = observables_input
        model_observables = Dense(N2, kernel_initializer=custom_weight_init_initial, bias_initializer='zeros', activation='relu')(model_observables)    
        if conf.use_leaky_relu : model_observables = LeakyReLU(0.2)       (model_observables)
        if conf.batch_norm     : model_observables = BatchNormalization() (model_observables)
        if conf.dropout > 0.   : model_observables = Dropout(conf.dropout)     (model_observables)
        model             = Concatenate()([model_conditions, model_observables])
    else :
        model = model_conditions
        
    for c in range(conf.C) :
        model = Dense (N1 + N2, kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation='relu')(model)
        if conf.use_leaky_relu : model = LeakyReLU (0.2)      (model)
        if conf.batch_norm     : model = BatchNormalization() (model)
        if conf.dropout > 0.   : model = Dropout(conf.dropout)     (model)
    
    # If target distribution is continuous and not discrete
    if not conf.is_discrete:
    
        gauss_means     = Dense (conf.D2 * conf.num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation='relu')(model)
        if conf.use_leaky_relu : gauss_means = LeakyReLU (0.2)      (gauss_means)
        if conf.batch_norm     : gauss_means = BatchNormalization() (gauss_means)
        if conf.dropout > 0.   : gauss_means = Dropout(conf.dropout)     (gauss_means)
        gauss_means = Dense (conf.num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation="linear")(gauss_means)
        gauss_means = Lambda(lambda x : conf.gauss_mean_scale * x)(gauss_means)
        gauss_means = Lambda(lambda x : add_gauss_mean_offsets(x, conf.num_gaussians, conf.range_min, conf.range_max))(gauss_means)
        
        gauss_sigmas       = Dense (conf.D2 * conf.num_gaussians   , kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation='relu')(model)
        if conf.use_leaky_relu : gauss_sigmas = LeakyReLU (0.2)      (gauss_sigmas)
        if conf.batch_norm     : gauss_sigmas = BatchNormalization() (gauss_sigmas)
        if conf.dropout > 0.   : gauss_sigmas = Dropout(conf.dropout)     (gauss_sigmas)
        gauss_sigmas = Dense (conf.num_gaussians     , kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation='relu')(gauss_sigmas)
        gauss_sigmas = Lambda (lambda x : conf.gauss_sigma_scale * x )                                                                (gauss_sigmas)
        gauss_sigmas = Lambda (lambda x : set_initial_gauss_sigmas(x, conf.num_gaussians, conf.range_min, conf.range_max, conf.gauss_width_factor))(gauss_sigmas)
        gauss_sigmas = Lambda (lambda x : K.log(1. + K.exp(x)))                                                                (gauss_sigmas)
        gauss_sigmas = Lambda (lambda x : add_epsilon_to_gauss_sigmas(x, conf.num_gaussians))                                       (gauss_sigmas)
        
        
        gauss_fractions = Dense(conf.D2 * conf.num_gaussians , kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation='relu')(model)
        if conf.use_leaky_relu : gauss_fractions = LeakyReLU (0.2)      (gauss_fractions)
        if conf.batch_norm     : gauss_fractions = BatchNormalization() (gauss_fractions)
        if conf.dropout > 0.   : gauss_fractions = Dropout(conf.dropout)     (gauss_fractions)
        gauss_fractions = Dense(conf.num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer='zeros', activation="linear")(gauss_fractions)
        gauss_fractions = Lambda(lambda x : conf.gauss_frac_scale * x)                                                    (gauss_fractions)
        gauss_fractions = Softmax()                                                                                (gauss_fractions)
        gauss_fractions = Lambda(lambda x : add_gauss_fraction_offsets(x, conf.num_gaussians, conf.min_gauss_amplitude_frac))(gauss_fractions)
        
        
        model = Concatenate()([gauss_fractions, gauss_means, gauss_sigmas])

    if n_observables > 0 : model = Model ([conditions_input, observables_input], model, name='GMM')
    else                      : model = Model (conditions_input, model, name='GMM')
    
    
    loss_function = "categorical_crossentropy"
    if not conf.is_discrete:
        loss_function = lambda y_true, y_pred : -1. * K_dataset_log_likelihood(y_true, y_pred, conf.num_gaussians)
        
    
    if   conf.optimiser.lower() == "sgd"      : model.compile(loss=loss_function, optimizer=SGD     (learning_rate=conf.learning_rate), weighted_metrics=[])    
    elif conf.optimiser.lower() == "adadelta" : model.compile(loss=loss_function, optimizer=Adadelta(learning_rate=conf.learning_rate), weighted_metrics=[])    
    elif conf.optimiser.lower() == "adam"     : model.compile(loss=loss_function, optimizer=Adam    (learning_rate=conf.learning_rate), weighted_metrics=[])   
    else : raise ValueError(f"Optimiser '{conf.optimiser}' not recognised") 
    
    return model, (n_external_parameters, n_observables, conf.num_gaussians)


class MultiDimGMM:
    
    def __init__(self, conf: OmegaConf, run_build: bool=True) -> None:
        self.conf = conf
        self.gmm_models = OrderedDict()
        self.obs_confs = OrderedDict()
        self.model_histories = OrderedDict()
        self.latent_transforms = OrderedDict()
        self.callbacks = []
        if run_build:
            self.build()
        self._run_build = run_build
        
    def build(self) -> None:
        
        for i, obs in enumerate(self.conf.features):
            
            obs_conf = OmegaConf.load(os.path.join(self.conf.features_conf_dir, f"{obs}.yaml"))    
            self.obs_confs[obs] = obs_conf
            self.gmm_models[obs] = GMM(obs_conf, n_observables=i)
            log.info(f"Built GMM for observable: {obs}")
        
        self.callbacks = configure_callbacks(self.conf)
    
    def fit(self, data) -> None:
        
        latent_data = data.copy()
        
        for obs in self.conf.features:
            x_to_latent_space, latent_space_to_x = transform_onto_latent_space(data[obs], self.conf, weights=data[self.conf.weight])
            latent_data[obs] = x_to_latent_space(latent_data[obs])
            self.latent_transforms[obs] = (x_to_latent_space, latent_space_to_x)
        
        for i, (obs, model) in enumerate(self.gmm_models.items()):
            
            x_external_params = data[self.conf.external_params]
            
            if i > 0:
                x_obs_features = self.conf.features[: i]
                x_train = (x_external_params, data[x_obs_features])
            else:
                x_train = x_external_params
                
            log.debug(f"{obs} is trained on {x_obs_features}")
            
            conf = self.obs_confs[obs]
            
            log.info(f"Training GMM for observable: {obs}")
            history = model[0].fit(x_train, data[obs], sample_weight=data[self.conf.weight], validation_split=conf.validation_split, epochs=conf.epochs, batch_size=conf.batch_size, shuffle=True, callbacks=self.callbacks)
            self.model_histories[obs] = history