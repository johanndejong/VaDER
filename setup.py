from setuptools import setup

setup(
    name='VaDER',
    version='0.1.0',
    author='Johann de Jong',
    author_email='johanndejong@gmail.com',
    packages=['VaDER'],
    #    package_dir={'': '.'},
    #    scripts=['bin/script1', 'bin/script2'],
    url='',
    license='LICENSE',
    description='Clustering multivariate time series with missing values',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.19.5",
        "scikit_learn >= 1.0.2",
        "scipy >= 1.7.3",
        "setuptools >= 57.0.0",
        "tensorflow >= 2.7.1",
        "tensorflow_addons >= 0.16.1"
    ]
)