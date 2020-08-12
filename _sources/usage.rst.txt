=====
Usage
=====

Superphot runs in five steps:

1. For each light curve, *fit* a model to the observations. This is the slowest step, but it can be done in parallel for a sample of light curves.
2. *Compile* the parameters from all the model fits into a single table. This is a separate step in case you did step 1 on a different computer (as we did).
3. *Extract* features from those models to be used for classification.
4. Initialize the classifier and *train* it on the training set.
5. Use the trained classifier to *classify* the test set.
6. (Optional) Cross-*validate* the classifier on the training set.
7. (Optional) *Optimize* the hyperparameters of the classifier.

-----------------------------
Running from the Command Line
-----------------------------

For basic functionality, Superphot can be run from the command line. For example:

.. code-block:: bash

    superphot-fit light_curves/*.dat --output-dir stored_models/  # this is parallelizable
    superphot-compile stored_models/ --output params
    superphot-extract train_input.txt params.npz --output train_data
    superphot-extract test_input.txt params.npz --output test_data --pcas pca.pickle  # use the same PCA
    superphot-train train_data.txt --output pipeline.pickle
    superphot-classify pipeline.pickle test_data.txt
    superphot-validate pipeline.pickle train_data.txt
    superphot-optimize param_dist.json pipeline.pickle train_data.txt

To see additional command-line arguments for any of these scripts, give the ``-h`` argument (e.g., ``superphot-fit -h``).

---------------------
Scripting with Python
---------------------

For more advanced use cases, you can import the module and use some version of the following workflow:

.. code-block:: python

    from superphot import fit, extract, classify
    import numpy as np

    # Fit the model to the data. Do this for each file.
    light_curve = fit.read_light_curve('light_curves/PSc000001.dat')  # may need custom parser
    fit.two_iteration_mcmc(light_curve, 'stored_models/PSc000001{}')

    # Compile parameters
    param_table = extract.compile_parameters('stored_models/', 'griz')
    np.savez_compressed('params.npz', **param_table, **param_table.meta)

    # Extract training features
    train_params = extract.load_data('train_input.txt', 'params.npz')
    train_data = extract.extract_features(train_params)

    # Extract test features
    test_params = extract.load_data('test_input.txt', 'params.npz')
    test_data = extract.extract_features(test_params, stored_pcas='pca.pickle')

    # Initialize and train the pipeline (can adjust hyperparameters here)
    pipeline = classify.Pipeline([
        ('scaler', classify.StandardScaler()),
        ('sampler', classify.MultivariateGaussian(sampling_strategy=1000)),
        ('classifier', classify.RandomForestClassifier(criterion='entropy', max_features=5)),
    ])
    classify.train_classifier(pipeline, train_data)

    # Do the classification
    results = classify.classify(pipeline, test_data)

    # Validate the classifier (optional)
    results_validate = classify.validate_classifier(pipeline, train_data)
    classify.make_confusion_matrix(results_validate, pipeline.classes_)

Most of these functions have optional inputs that are not shown here. See the :ref:`api_documentation`.

------------------------
Light Curve Data Formats
------------------------

The `light curve data <https://zenodo.org/record/3974950>`_ we used in developing this package were stored in `SNANA <https://github.com/RickKessler/SNANA>`_ text format.
Here is an example::

    SURVEY: PS1MD
    SNID:  PSc000001
    IAUC:    UNKNOWN
    RA:        52.4530625  deg
    DECL:       -29.0749750  deg
    MWEBV: 0.0075 +- 0.0003 MW E(B-V)
    REDSHIFT_FINAL:  0.1260 +- 0.0010  (CMB)
    SEARCH_PEAKMJD: 55207.0
    FILTERS:    griz

    # ======================================
    # TERSE LIGHT CURVE OUTPUT
    #
    NOBS: 306
    NVAR:   7
    VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR

    OBS: 55086.6 g NULL  -243.440 231.478 nan -1.032
    OBS: 55089.6 g NULL  -62.931 13.480 nan -0.233
    OBS: 55095.6 g NULL  -15.102 16.238 nan -1.167
    OBS: 55098.6 g NULL  -94.646 13.910 nan -0.160
    OBS: 55104.6 g NULL  -28.093 12.441 nan -0.481
    OBS: 55191.3 g NULL  -27.414 10.304 nan -0.408
    OBS: 55203.3 g NULL  1381.526 18.142 -12.851 0.014
    OBS: 55446.6 g NULL  -3.432 9.291 nan -2.939
    OBS: 55449.6 g NULL  9.291 10.095 -7.420 1.180
    OBS: 55452.6 g NULL  -2.915 10.422 nan -3.881
    ...

Superphot includes a function that can parse data in this format (:func:`superphot.fit.read_light_curve`).
It should also be able to recognize a simple text format like this::

    PHASE FLT FLUXCAL FLUXCALERR
    -120.4 g -243.44 231.478
    -117.4 g -62.931 13.48
    -111.4 g -15.102 16.238
    -108.4 g -94.646 13.91
    -102.4 g -28.093 12.441
    -15.7 g -27.414 10.304
    -3.7 g 1381.526 18.142
    239.6 g -3.432 9.291
    242.6 g 9.291 10.095
    245.6 g -2.915 10.422
    ...

If your data are in an unrecognizable format, you will have to write your own parser.
The data need to end up as an Astropy table with (at least) the following columns and metadata:

* ``PHASE`` is the date of the observation in days relative to discovery (``SEARCH_PEAKMJD`` in our case)
* ``FLT`` is the filter
* ``FLUXCAL`` and ``FLUXCALERR`` are the flux and its uncertainty

--------------------------
Input/Output Table Formats
--------------------------

Superphot writes all its outputs in Astropy's ``ascii.fixed_width_two_line`` format, but it can read any plain text format guessable by Astropy.

The files called ``train_input.txt`` and ``test_input.txt`` should have the following columns:

* ``filename``: the name of the light curve data file, without the extension;
* ``redshift``: the redshift of the transient, used to calculate the luminosity distance and cosmological :math:`K`-correction;
* ``MWEBV``: the Milky Way selective extinction :math:`E(B-V)`, used to correct the fluxes; and
* ``type`` (optional): the supernova spectroscopic classification, used to train the classifier.

The ``filename`` column is used as the supernova identifier, so each filename must be unique (even if they are in different directories).

Superphot's feature extraction step saves the features in two separate files with the same base name (e.g., ``test_data`` above) but different extensions.
The ``test_data.txt`` file includes all the supernova metadata, which will be identical to ``test_input.txt`` unless stored model parameters are missing for any input supernovae.
The ``test_data.npz`` file includes the features themselves, stored as a compressed multidimensional binary array.

The classification and validation results are also written to text files by :func:`superphot.classify.write_results`.
The tables include the same metadata as the feature extraction step plus the photometric classification, the classification confidence, and probabilities for each possible classification.

----------------------------
Other Command Line Utilities
----------------------------
In addition to the main commands listed above, Superphot includes four utilities to help produce tables and figures for publications:

* ``superphot-confuse validation_results.txt`` plots a confusion matrix from saved cross-validation results.
* ``superphot-bar validation_results.txt test_results.txt`` plots stacked bar plots showing the class fractions of the training and test sets.
* ``superphot-latex test_results.txt`` converts the plain text results table into a nicely formatted AASTeX deluxetable.
* ``superphot-hyperparameters hyperparameters.txt`` plots 3D scatter plots of various performance metrics vs. the classifier hyperparameters.
