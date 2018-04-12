# abj2016
**Converting parallaxes to distances like Astraatmadja &amp; Bailer-Jones (2016)**

This simple python module lets you infer stellar distances from measured parallaxes. 
It basically implements the formalism of [Astraatmadja &amp; Bailer-Jones (2016)](http://adsabs.harvard.edu/abs/2016ApJ...832..137A "ABJ2016 Paper").
This is how it works:

~~~~
import abj2016
import numpy 
# Give me an array of measured parallaxes (in milli-arcseconds) and the associated uncertainties
pi = 1./np.linspace(0.05, 4., 100)
e_pi = 0.04 * np.ones(100)
~~~~
