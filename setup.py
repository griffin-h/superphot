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

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setuptools.setup(
    name="superphot",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Frederick Dauphin & Griffin Hosseinzadeh",
    author_email="griffin.hosseinzadeh@cfa.harvard.edu",
    description="Photometric classification of supernovae",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FDauphin/2019_REU_CfA",
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    license="GNU General Public License v2 (GPLv2)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
    ],
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    install_requires=requirements,
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
