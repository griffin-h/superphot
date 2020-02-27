=====
Usage
=====

Superphot runs in three steps:

1. For each light curve, fit a model to the observations.
2. Extract features from those models to be used for classification.
3. Using those features, train the classifier, classify the light curves, and validate the classifier performance.

For basic functionality, Superphot can be run from the command line. For example:

.. code-block:: bash

    superphot-fit light_curve_file.dat --output-dir stored_models/
    superphot-extract input_table.txt stored_models/ --output test_data
    superphot-classify test_data

For more advanced use cases, you can import the module and use some version of the following workflow:

.. code-block:: python

    from superphot import util, extract_features, classify
    from glob import glob
    from os import path

    # Fit the model to the data. Do this for each file.
    for filename in glob('light_curves/*.dat'):
        outfile = path.join('stored_models/', path.basename(filename).split('.')[0] + '_{}')
        light_curve = util.read_light_curve(filename)  # may need to write your own parser
        fit_model.two_iteration_mcmc(light_curve, outfile)

    # Extract features
    data_table = extract_features.compile_data_table('input_table.txt')
    test_data = extract_features.extract_features(data_table, 'stored_models/')
    train_data = util.select_labeled_events(test_data)

    # Initialize the classifier and resampler (can adjust hyperparameters here)
    clf = classify.RandomForestClassifier(criterion='entropy', max_features=5)
    sampler = classify.MultivariateGaussian(sampling_strategy=1000)

    # Do the classification
    test_data['probabilities'] = classify.fit_predict(clf, sampler, train_data, test_data)
    results = classify.aggregate_probabilities(test_data)

    # Validate the classifier
    train_data['probabilities'] = classify.validate_classifier(clf, sampler, train_data)
    results_validate = classify.aggregate_probabilities(train_data)
    classify.make_confusion_matrix(results_validate, clf.classes_)

------------------------
Light Curve Data Formats
------------------------

The light curve data we used in developing this package were stored in `SNANA <https://github.com/RickKessler/SNANA>`_ text format.
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

Superphot includes a function that can parse data in this format (:func:`superphot.util.read_light_curve`).
It should also be able to recognize a simple text format like this::

    # MWEBV: 0.0075
    # REDSHIFT: 0.1260
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
* ``MWEBV`` is the Milky Way selective extinction :math:`E(B-V)` (used to correct the fluxes)
* ``REDSHIFT`` is the redshift (used to calculate the luminosity distance and cosmological :math:`K`-correction)

Alternatively, if your light curve files include no metadata, you can give the metadata in the input table (see below).

--------------------------
Input/Output Table Formats
--------------------------

Superphot writes all its outputs in Astropy's ``ascii.fixed_width_two_line`` format, but it can read any plain text format guessable by Astropy.

The file called ``input_table.txt`` above must have at least two columns: ``filename`` (referring to the light curve data file) and ``type`` (referring to the supernova classification).
The ``filename`` column is used as the supernova identifier, so each filename must be unique (even if they are in different directories).
The ``type`` column is used to train the classifier and can be left blank for supernovae not in the training set.
If the required metadata are not in the light curve files, you must also include the columns ``id``, ``A_V``, and ``redshift``.

Superphot's feature extraction step saves the features in two separate files with the same base name (``test_data`` above) but different extensions.
The ``test_data.txt`` file includes all the supernova metadata, which will be identical to ``input_table.txt`` unless stored model parameters are missing for any input supernovae.
The ``test_data.npz`` file includes the features themselves, stored as a compressed multidimensional binary array.

The classification and validation results are also written to text files by :func:`superphot.classify.write_results`.
The tables include the same metadata as the feature extraction step plus columns of probabilities for each possible classification.
In addition, the validation results can be used to create and plot a confusion matrix using :func:`superphot.classify.make_confusion_matrix`.
You can also plot a confusion matrix from stored validation data on the command line with ``superphot-confuse validation.txt``.