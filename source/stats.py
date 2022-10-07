"""
Statistical functions
"""

import numpy as np
from omegaconf import OmegaConf

# Define transformation onto latent space
def smoothing_function(u: np.ndarray, alpha: float=4, beta: float=3, gamma: float=1) -> np.ndarray:
    
    q = 1 / (1 + np.exp(alpha * (u - beta) - gamma)) * 1 / (1 + np.exp(-alpha * (u + beta) - gamma))
    
    return q

def get_cumalative_distribution(data: np.ndarray, weights: np.ndarray=None, npoints: int=50, lower_lim: float=None, upper_lim: float=None) -> np.ndarray:
    
    if lower_lim is None:
        lower_lim = np.amin(data)
    if upper_lim is None:
        upper_lim = np.amax(data)
    if weights is None:
        weights = np.ones_like(data, weights, npoints, lower_lim, upper_lim)
    
    ax_scan_points = np.linspace(lower_lim, upper_lim, npoints + 1)
    
    data_cdf = []
    for point in ax_scan_points :
        data_cdf.append(np.sum([w for x,w in zip(data, weights) if x < point]))
        
    data_cdf = np.array(data_cdf)
    
    return data_cdf

def transform_onto_latent_space(dataset, conf, feature):
    
    data = dataset[feature]
    weights = dataset[conf.weight]
    feature_conf = conf.features[feature]
    
    weights = weights / np.sum(weights)
    xmin = np.amin(data)
    xmax = np.amax(data)
    x_scan_points = np.linspace(xmin, xmax, 1 + conf.n_axis_points)

    umin = conf.smooth_space_limits[0]
    umax = conf.smooth_space_limits[1]
    udiv = conf.smooth_space_division
    
    data_cdf = get_cumalative_distribution(data, npoints=conf.n_axis_points, weights=weights)
    constant_cdf = (x_scan_points - xmin) / (xmax - xmin)
    combined_cdf = feature_conf.data_frac_constant * constant_cdf + (1 - feature_conf.data_frac_constant) * data_cdf
    
    latent_space_x = np.linspace(umin, umax, 1 + int(2 * umax / udiv))
    smooth_space_y = smoothing_function(latent_space_x, conf.alpha, conf.beta, conf.gamma)
    
    smooth_space_cdf = np.array([np.sum(smooth_space_y[:i+1]) for i in range(len(smooth_space_y))])
    smooth_space_cdf /= smooth_space_cdf[-1]
    smooth_space_cdf[0] = 0.
    
    constant_cdf    = (latent_space_x + umax) / (2 * umax)
    latent_space_cdf = feature_conf.gauss_frac_constant * constant_cdf + (1 - feature_conf.gauss_frac_constant) * smooth_space_cdf

    A_to_z = lambda A : np.interp(A, x_scan_points, combined_cdf)     # x -> x_CDF
    z_to_A = lambda z : np.interp(z, combined_cdf  , x_scan_points)   # x_CDF -> x

    z_to_g = lambda z : np.interp(z, latent_space_cdf, latent_space_x  )  # u_CDF -> u
    g_to_z = lambda g : np.interp(g, latent_space_x  , latent_space_cdf)  # u -> u_CDF

    A_to_g = lambda A : z_to_g(A_to_z(A))  # x -> u
    g_to_A = lambda g : z_to_A(g_to_z(g))  # u -> x
    
    return A_to_g, g_to_A
    