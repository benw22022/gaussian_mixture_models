# Imports
import logger
log = logger.get_logger(__name__)
import os
import pandas as pd
import numpy as np
import uproot
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from source.stats import transform_onto_latent_space
from gc import callbacks
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy
from matplotlib import cm
from hydra.utils import get_original_cwd, to_absolute_path
from models import GMM, configure_callbacks



def train_model(model_conf, dataset, obs, n_external_parameters=1, n_observables=0):

    model, params = GMM(model_conf, n_external_parameters, n_observables)
    weights = dataset['weight']
    
    x_to_latent_space, latent_space_to_x = transform_onto_latent_space(dataset, )

    y = x_to_latent_space(dataset[obs])

    x = dataset[model_conf.external_params]

    callbacks = configure_callbacks(model_conf, weights_save_dir=f'network_weights_{obs}')
        
    history = model.fit(x, y, sample_weight=weights, validation_split=0.1, epochs=5, batch_size=model_conf.batch_size, shuffle=True, callbacks=callbacks)

    return model, history

def plot_history(history, obs):
    # Plot loss vs epochs
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Val')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(obs)
    ax.legend()
    
def histo_to_line (bins, values, errors=None) :
    X, Z, EZ = [], [], []
    for i in range(len(bins)-1) :
        X .append(bins[i])
        X .append(bins[i+1])
    for zp in values :
        Z .append(zp)
        Z .append(zp)
    if type(errors) is type(None) :
        return np.array(X), np.array(Z)
    for ezp in errors :
        EZ.append(ezp)
        EZ.append(ezp)
    return np.array(X), np.array(Z), np.array(EZ)


def plot_gmm(model, model_conf, feature, x, y):
    
    params = model.predict(x[:1])
    num_gauss = model_conf.num_gaussians

    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]

    x_range = np.linspace(-5, 5, 501)
    sum_gauss = 0

    newcolors = cm.get_cmap('brg', num_gauss)(np.linspace(0, 1, num_gauss))

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    for i, (frac, mean, sigma) in enumerate(zip(gauss_fracs[0], gauss_means[0], gauss_sigmas[0])):
        gauss_y = frac * scipy.stats.norm.pdf(x_range, mean, sigma)
        sum_gauss += gauss_y
        ax1.plot(x_range, gauss_y, c=newcolors[i], linewidth=0.6, linestyle="-")
        
    ax1.plot(x_range, sum_gauss, color='black', label=r"$p_\phi(u|{M_{N}, M_{W_{R}}})$")
    ax1.hist(y, bins=100, density=True, histtype='step', label='MG5')
    ax1.set_xlabel(r"{0} Latent".format(model_conf.latex_name))
    ax1.legend()
    
    gauss_hist, bins = np.histogram(x_range, bins=np.linspace(-5, 5, 25), density=True, weights=sum_gauss)
    nom_hist, bins = np.histogram(y, bins=np.linspace(-5, 5, 25), density=True)
    
    residual = gauss_hist / nom_hist
    
    ax2.plot(bins, np.ones_like(bins), color='black')
    ax2.stairs(residual, bins, color='red', baseline=1)
    plt.tight_layout()
    os.makedirs(os.path.join("plots", "model_output"), exist_ok=True)
    saveas = os.path.join("plots", "model_output", f'{feature}_gmm.png')
    plt.savefig(saveas, dpi=300)
    log.info(f"Plotted {saveas}")

def simple_1D_training(conf: OmegaConf):
    
    dataset = uproot.concatenate(conf.dataset_path, library='pd')
    weights = dataset[conf.weight]
    log.info(f"Loaded data from {conf.dataset_path}")
    
    os.makedirs(os.path.join('plots', 'transformed_inputs'), exist_ok=True)
    
    for feature, feature_conf in conf.features.items():
        
        log.info(f"Training network for {feature}")
        
        # obs_conf = OmegaConf.load(os.path.join(get_original_cwd(), f'config/features/{feature}.yaml'))    
        # conf = OmegaConf.merge(conf, obs_conf)

        obs_data = dataset[feature]
        
        x_to_latent_space, latent_space_to_x = transform_onto_latent_space(dataset, conf, feature)

        obs_transformed = x_to_latent_space(obs_data)
        
        # Check closure
        obs_trans_back = latent_space_to_x(obs_transformed)

        ratio = (obs_trans_back / obs_data).transpose()
        log.info(f"Observable {feature} closure is {np.mean(ratio):.5f} +- {np.std(ratio):.5f}     (should be 1 +- 0)")
        
        # Plot transformation
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        ax1.hist(obs_data, 24, histtype='step', density=False, color='blue')
        ax2.hist(obs_transformed, 25, histtype='step', density=False, color='orange')
        ax1.set_xlabel(feature_conf.latex_name)
        ax2.set_xlabel(feature_conf.latex_name)
        ax1.set_title("Data")
        ax2.set_title(f"Latent: f = {feature_conf.data_frac_constant:.2f}")
        plt.tight_layout()
        saveas = os.path.join('plots', 'transformed_inputs', f'{feature}.png')
        plt.savefig(saveas, dpi=300)
    
    # Begin Training
    results = {}

    for feature, feature_conf in conf.features.items():
        
        print(f"Training {feature}")
        
        obs_data = dataset[feature]
        x_to_latent_space, latent_space_to_x = transform_onto_latent_space(dataset, conf, feature)
        
        model, params = GMM(feature_conf, len(conf.external_params), 0)
        
        weights = dataset['weight']
        
        y = x_to_latent_space(dataset[feature])

        x = dataset[conf.external_params]

        print(x)
        
        callbacks = configure_callbacks(conf, weights_save_dir=f'network_weights_{feature}')
            
        history = model.fit(x, y, sample_weight=weights, validation_split=0.1, epochs=1, batch_size=feature_conf.batch_size, shuffle=True, callbacks=callbacks)

        plot_history(history, feature)
        plot_gmm(model, feature_conf, feature, x, y)
        
        results[feature] = [model, history]
        
    
    

