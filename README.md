# abj2016 - Convert parallaxes to distances like Astraatmadja &amp; Bailer-Jones (2016)

## HOWTO

This simple python module lets you infer stellar distances from measured parallaxes. 
It basically implements the formalism of [Astraatmadja &amp; Bailer-Jones (2016)](http://adsabs.harvard.edu/abs/2016ApJ...832..137A "ABJ2016 Paper").
This is how it works. Easiest thing is to download the `abj2016.py` and copy it into your project. Then
~~~~
import abj2016
import numpy as np
~~~~
Now, let's imagine an array of stellar distances (in kiloparsecs), between 0.05 and 4 kpc. We measure the parallax for each star with an uncertainty of about 0.04 milliarcsec (vamos, a typical Gaia DR2 value). Then we have 
~~~~
true_dists = np.linspace(0.05, 4., 100)
measured_parallaxes = 1. / true_dists + np.random.normal(loc=0.0, scale=0.04, size=100)
~~~~
If we, in turn, want to infer the stellar distances from the measured parallaxes, that's not as easy, as explained in e.g. 
[Bailer-Jones (2015)](http://adsabs.harvard.edu/abs/2015PASP..127..994B "BJ2015 Paper"). Especially in the presence of large fractional parallax uncertainties, you shouldn't just do `d = 1/measured_parallax`... 
Instead, you want to compute statistics of the posterior distance PDF (given parallax and parallax uncertainty):
~~~~
distpdf = abj2016.distpdf(measured_parallaxes, 0.04)   # Initiates the distpdf object
modedists = distpdf.modedist
meandists = distpdf.meandist
sigdists  = distpdf.diststd
~~~~
This also works if the parallax uncertainties are given as an array. Or if the measured parallax is a scalar.

You can also specify the space density prior by adding e.g. `priors="uniform_density"`. Currently, only the three isotropic density priors presented by Astraatmadja &amp; Bailer-Jones (2016) are supported (default:`priors="exponential"`). You can also specify the resolution of the distance posterior PDF (default: `resolution=10000`), and the minimum and maximum allowed distances in kiloparsec (default: `min_dist=0` and `max_dist=30`).

You can also get the posterior PDF itself, via e.g.:
~~~~
pi, sigma_pi = 0.3, 0.1 # measured parallax and uncertainty in mas 
distpdf = abj2016.distpdf(pi, sigma_pi, min_dist=0., max_dist=10., resolution=100000, priors="uniform_distance")
array, pdf = distpdf.distarray, distpdf.distpdf
~~~~
Have fun and give a shout if you find a bug or have a question: `fanders*ät*aip*dot*de`.

