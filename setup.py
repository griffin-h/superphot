from os import path
import setuptools
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
superphot does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*sys.version_info[:2], *min_version)
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setuptools.setup(
    name="superphot",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Griffin Hosseinzadeh & Frederick Dauphin",
    author_email="griffin0@arizona.edu",
    description="Photometric classification of supernovae",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://griffin-h.github.io/superphot",
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    license="GNU General Public License v2 (GPLv2)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
    ],
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'superphot-fit = superphot.fit:_main',
            'superphot-compile = superphot.extract:_compile_parameters',
            'superphot-extract = superphot.extract:_main',
            'superphot-train = superphot.classify:_train',
            'superphot-classify = superphot.classify:_classify',
            'superphot-validate = superphot.classify:_validate',
            'superphot-confuse = superphot.classify:_plot_confusion_matrix_from_file',
            'superphot-bar = superphot.classify:_bar_plot_from_file',
            'superphot-latex = superphot.classify:_latex',
            'superphot-optimize = superphot.optimize:_main',
            'superphot-hyperparameters = superphot.optimize:_plot_hyperparameters_from_file',
        ],
    },
)
