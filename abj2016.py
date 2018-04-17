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

def exp_prior(d, L=1.35):
    """
    Exponentially decreasing space density prior
    
    Input:
        d: distance (typically an array)
    Optional:
        L: Scale of the exponentially decreasing density (default: 1.35 kpc)
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

class distpdf(object):
    """
    Class for posterior distance PDF given parallax and parallax uncertainty
    """
    def __init__(self, pi, sigma_pi, min_dist=0., max_dist=30., resolution=10000, **kwargs):
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
            (none)
        Object properties:
            distarray, distpdf - distance array & corresponding posterior 
                                 distance PDF (1D if pi&sigma_pi are scalar, 
                                 2D if not)
            meandist, diststd, modedist - statistics of the distance PDF
        """
        self.distarray = np.linspace(min_dist, max_dist, resolution)
        distpdf   = posterior(self.distarray, pi, sigma_pi, **kwargs)
        self.distpdf   = distpdf / np.sum(distpdf, axis=0) 
        
        # Compute some basic statistics: Mean, standard deviation, and mode 
        if np.isscalar(pi):
            self.meandist = np.average( self.distarray, axis=0, weights=self.distpdf )
            self.diststd  = np.sqrt( np.average( (self.distarray - self.meandist)**2, 
                                                  axis=0, weights=self.distpdf ) )
            self.modedist = self.distarray[np.argmax(self.distpdf)]
        else:
            self.meandist = np.sum(self.distpdf*self.distarray[:, np.newaxis]/np.sum(self.distpdf,axis=0), axis=0)
            self.diststd  = np.sqrt( np.sum( self.distpdf * 
                                            (self.distarray[:, np.newaxis] - self.meandist[np.newaxis,:] / np.sum(self.distpdf,axis=0) )**2., axis=0 ))
            self.modedist = self.distarray[np.argmax(self.distpdf, axis=0)]


