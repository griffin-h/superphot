.. _reproducing_our_paper:

=====================
Reproducing Our Paper
=====================
In Hosseinzadeh et al. (`2020 <https://ui.adsabs.harvard.edu/abs/2020arXiv200804912H/abstract>`_), we trained the classifier on a sample of 557 spectroscopically classified supernovae from the `Pan-STARRS1 <https://panstarrs.stsci.edu>`_ Medium Deep Survey, and then applied it to an additional 2315 transients with host galaxy redshifts.

To reproduce our work, first download the light curve data from `Zenodo <https://zenodo.org/record/3974950>`_. Unzip the file ``ps1_sne_zenodo.zip``.

For each light curve file, run ``superphot-fit``.
You can either run these in series, by using a glob pattern (``*``) as below, or in parallel, by giving only one filename at a time.
Decide where you want to store your models. Here I am using a directory called ``stored_models/``.
These will take a while to run.

.. code-block:: bash

    superphot-fit ps1_sne_zenodo/*.dat --output-dir stored_models/

When all the fits are finished, compile the resulting parameters into a single file. (I'm calling this ``params.npz``.)
To reproduce our results exactly, use 0 as the seed for the random number generator.

.. code-block:: bash

    superphot-compile stored_models/ --output params --random-state 0

Next, download the machine-readable input tables from our paper.
The test set is in `Table 3 <https://arxiv.org/src/2008.04912v1/anc/t3.txt>`_.
The training/validation set is in `Table 4 <https://arxiv.org/src/2008.04912v1/anc/t4.txt>`_.
These tables include the results. You can either delete these columns or just ignore them.
Extract features for each of these sets.
In our paper, we used the median parameters for training, and 10 draws from the posterior (the default) for validation and testing.
Make sure to use the same PCA in the test and validation sets as in the training set. (This is saved automatically.)
These commands will produce six output files (in addition to ``pca.pickle``):

- ``train_data.txt``
- ``train_data.npz``
- ``validation_data.txt``
- ``validation_data.npz``
- ``test_data.txt``
- ``test_data.npz``

.. code-block:: bash

    superphot-extract t4.txt params.npz --use-median --output train_data
    superphot-extract t4.txt params.npz --pcas pca.pickle --output validation_data
    superphot-extract t3.txt params.npz --pcas pca.pickle --output test_data

Now that the data are ready, you can initialize and train the classifier.
Here I'm saving it to a Python pickle file ``pipeline.pickle``.
Again, if you want to reproduce our results exactly, use a random seed of 0.

.. code-block:: bash

    superphot-train train_data.txt --output pipeline.pickle --random-state 0

Finally, you can use the pipeline to classify the test set.
Here the results will be saved to ``test_data_results.txt``.

.. code-block:: bash

    superphot-classify pipeline.pickle test_data.txt --output test_data

Note that in our paper, we also had a third set of "rare transients" in `Table 5 <https://arxiv.org/src/2008.04912v1/anc/t5.txt>`_.
To reproduce our classifications for these, repeat the above procedure using ``t5.txt`` instead of ``t3.txt``.

.. code-block:: bash

    superphot-extract t5.txt params.npz --pcas pca.pickle --output other_data
    superphot-classify pipeline.pickle other_data.txt --output other_data

To reproduce our confusion matrices, you can run the leave-one-out cross-validation.

.. code-block:: bash

    superphot-validate pipeline.pickle validation_data.txt --train-data train_data.txt

Finally, if you want to reproduce the hyperparameter optimization in Appendix D, copy and paste the following into a JSON file (here I'm calling it ``param_dist.json``):

.. code-block:: json

    {"classifier__n_estimators": [500, 200, 100, 50, 20, 10],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [30, 25, 20, 15, 10, 5],
    "classifier__max_features": [25, 20, 15, 10, 5]}

Then run the optimizer.

.. code-block:: bash

    superphot-optimize param_dist.json pipeline.pickle validation_data.txt --train-data train_data.txt
