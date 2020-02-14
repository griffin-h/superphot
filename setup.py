import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="superphot",
    version="0.0.1",
    author="Frederick Dauphin & Griffin Hosseinzadeh",
    author_email="griffin.hosseinzadeh@cfa.harvard.edu",
    description="Photometric classification of supernovae",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FDauphin/2019_REU_CfA",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'astropy',
        'pymc3',
        'scikit-learn',
        'imbalanced-learn',
        'arviz',
    ],
    entry_points={
        'console_scripts': [
            'superphot-fit = superphot.fit_model:main',
            'superphot-diagnostics = superphot.fit_model:plot_diagnostics',
            'superphot-extract = superphot.extract_features:main',
            'superphot-classify = superphot.classify:main',
            'superphot-confuse = superphot.classify:plot_confusion_matrix_from_file',
        ],
    },
)
