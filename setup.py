from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='convae3d',
    version='0.1.0',
    author='Murray Cutforth',
    author_email='mcc4@stanford.edu',
    description='A simple 3D convolutional autoencoder repository',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)