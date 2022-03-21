from setuptools import setup

setup(
    name='hdf5io',
    version='0.1',
    packages=['hdf5io'],
    install_requires=['numpy','tables','future'],
    url='',
    license='MIT',
    author='Daniel Hillier',
    author_email='hillier_dani@yahoo.fr',
    description='Save/load data from a Hdf5io class into hdf5 file similar to how Matlab saves/loads data',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
