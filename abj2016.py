# Getting distances from parallaxes using the simple Bayes formula by Astraatmadja & Bailer-Jones (2016)
# Author: F. Anders (AIP)
# Last modified: 12.04.2018

import numpy as np

# Isotropic Priors (Table 1 in Astraatmadja&Bailer-Jones 2016)
def uniform_distance_prior(d, rlim=30.):
    """
    Uniform distance prior
    
    Input:
        d: distance (typically an array)
    Optional:
        rlim: Maximum allowed distance (default: 30 kpc)
    Output:
        Uniform distance prior
    """
    return np.piecewise(d, [d < 0, (d >= 0)*(d<=rlim), d>rlim], [0, 1/rlim, 0])

def uniform_density_prior(d, rlim=30.):
    """
    Uniform space density prior
    
    Input:
        d: distance (typically an array)
    Optional:
        rlim: Maximum allowed distance (default: 30 kpc)
    Output:
        Uniform density prior
    """
    return np.piecewise(d, [d < 0, (d >= 0)*(d<=rlim), d>rlim], [0, 1/(rlim**3.) * d**2., 0])

def exp_prior(d, L=1.):
    """
    Exponentially decreasing space density prior
    
    Input:
        d: distance (typically an array)
    Optional:
        L: Scale of the exponentially decreasing density (default: 1 kpc)
    Output:
        Exponentially decreasing space density prior
    """
    return np.piecewise(d, [d < 0, d >= 0], [0, 1/(2*L**3.) * d**2. * np.exp(- d / L )])

# Likelihood
def likelihood(pi, d, sigma_pi):
    """
    Gaussian likelihood of parallax given distance and parallax uncertainty
        
    Input:
        pi:       parallax (array or scalar)
        d:        distance (typically an array)
        sigma_pi: parallax_uncertainty (array or scalar)
    Output:
        Likelihood of parallax given distance and parallax uncertainty (formula 1 of Astraatmadja&Bailer-Jones 2016)
    """
    if np.isscalar(pi):
        return 1/np.sqrt(2*np.pi*sigma_pi**2.) * np.exp(-1/(2*sigma_pi**2.) * (pi - 1./d)**2. )
    else:
        return 1/np.sqrt(2*np.pi*sigma_pi**2.) * np.exp(-1/(2*sigma_pi**2.) * (pi[np.newaxis, :] - 1./d[:, np.newaxis])**2. )

def posterior(distarray, pi, sigma_pi, prior="exponential" ):
    """
    Posterior distance distribution.
        
    Input:
        distarray:distance array on which to calculate the posterior PDF
        pi:       parallax (array or scalar)
        sigma_pi: parallax_uncertainty (array or scalar)
    Optional:
        prior:    String. Decides which prior to use (at present either "exponential", "uniform_density", "uniform_distance")
    Output:
        Posterior distance PDF (up to a factor), given parallax and parallax uncertainty (formula 2 of Astraatmadja&Bailer-Jones 2016)
    """
    if prior == "exponential":
        prior = exp_prior
    elif prior == "uniform_density":
        prior = uniform_density_prior
    elif prior == "uniform_distance":
        prior = uniform_distance_prior
    else:
        raise ValueError("Prior keyword does not exist")
    
    if np.isscalar(pi):
        return prior(distarray) * likelihood(pi, distarray, sigma_pi)
    else:
        return prior(distarray)[:, np.newaxis] * likelihood(pi, distarray, sigma_pi)


def distpdf(pi, sigma_pi, min_dist=0., max_dist=30., resolution=10000, **kwargs):
    """
    Returns a distance array and the corresponding distance PDF
        
    Input:
        pi:        parallax (array or scalar)
        sigma_pi:  parallax_uncertainty (array or scalar)
    Optional:
        min_dist:  minimum allowed distance
        max_dist:  maximum allowed distance
        resolution:resolution of the distance PDF
    Output:
        distance array, posterior distance PDF (1D if pi&sigma_pi are scalar, 2D if not)
    """
    distarray = np.linspace(min_dist, max_dist, resolution)
    distpdf   = posterior(distarray, pi, sigma_pi, **kwargs)
    distpdf   = distpdf / np.sum(distpdf, axis=0) 
    return distarray, distpdf

def meandist(pi, sigma_pi, **kwargs):
    """
    Get the mean posterior distance given parallax and parallax uncertainty
        
    Input:
        pi:        parallax (array or scalar)
        sigma_pi:  parallax_uncertainty (array or scalar)
    Optional:
        keyword arguments (min_dist, max_dist, resolution, prior)
    Output:
        mean posterior distance (array or scalar)
    """
    dists, pdf = distpdf(pi, sigma_pi, **kwargs)
    if np.isscalar(pi):
        return np.average( dists, axis=0, weights=pdf )
    else:
        return np.sum(pdf*dists[:, np.newaxis]/np.sum(pdf,axis=0), axis=0)

def diststd(pi, sigma_pi, **kwargs):
    """
    Get the posterior distance standard deviation given parallax and parallax uncertainty
        
    Input:
        pi:        parallax (array or scalar)
        sigma_pi:  parallax_uncertainty (array or scalar)
    Optional:
        keyword arguments (min_dist, max_dist, resolution, prior)
    Output:
        posterior distance standard deviation (array or scalar)
    """
    dists, pdf = distpdf(pi, sigma_pi)
    if np.isscalar(pi):
        return np.sqrt( np.average( (dists-meandist(pi, sigma_pi))**2,  axis=0, weights=pdf ) )
    else:
        return np.sqrt( np.sum( pdf * (dists[:, np.newaxis] - meandist(pi, sigma_pi)[np.newaxis,:] / np.sum(pdf,axis=0) )**2., axis=0 ))

def modedist(pi, sigma_pi, **kwargs):
    """
    Get the posterior distance mode given parallax and parallax uncertainty
        
    Input:
        pi:        parallax (array or scalar)
        sigma_pi:  parallax_uncertainty (array or scalar)
    Optional:
        keyword arguments (min_dist, max_dist, resolution, prior)
    Output:
        mode posterior distance (array or scalar)
    """
    dists, pdf = distpdf(pi, sigma_pi, **kwargs)
    if np.isscalar(pi):
        return dists[np.argmax(pdf)]
    else:
        return dists[np.argmax(pdf, axis=0)]

