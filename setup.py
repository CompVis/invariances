from setuptools import setup, find_packages

setup(
    name='invariances',
    version='0.0.1',
    url='https://compvis.github.io/invariances/',
    description='Interpreting deep representations and their invariances.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'requests',
        'tqdm',
    ],
)
