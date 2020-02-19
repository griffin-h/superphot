.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Superphot Documentation
=======================

Superphot is an open-source Python package for photometric classification of supernovae, based on the method of Villar et al. (`2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract>`_). To apply this method, you must have the full *griz* light curve and redshift (e.g., from the host galaxy) for each transient. As such, Superphot is designed for use by time-domain surveys to retrospectively classify transients that were not observed spectroscopically, not for real-time classification.

In Villar et al. (`2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract>`_) and Dauphin et al. (`2020 <https://ui.adsabs.harvard.edu/abs/2020AAS...23527618D/abstract>`_), we trained the classifier on a sample of ~500 spectroscopically classified supernovae from the `Pan-STARRS1 <https://panstarrs.stsci.edu>`_ Medium Deep Survey, and then applied it to an additional ~3000 transients with host galaxy redshifts. This trained classifier is included in the package, but users will most likely want to train on their own survey data for better performance.

.. toctree::
   :maxdepth: 2

   installation
   usage
   api