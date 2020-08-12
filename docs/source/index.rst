.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
Superphot Documentation
=======================

Superphot is an open-source Python package for photometric classification of supernovae, based on the method of Villar et al. (`2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract>`_). To apply this method, you must have the redshift (e.g., from the host galaxy) and Milky Way extinction for each transient. Superphot is optimized for use by time-domain surveys to retrospectively classify transients that were not observed spectroscopically, not for real-time classification.

In Hosseinzadeh et al. (2020, submitted), we trained the classifier on a sample of 557 spectroscopically classified supernovae from the `Pan-STARRS1 <https://panstarrs.stsci.edu>`_ Medium Deep Survey, and then applied it to an additional 2315 transients with host galaxy redshifts.

.. toctree::
   :maxdepth: 2

   installation
   usage
   api
   release-history