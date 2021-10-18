from setuptools import setup, find_packages

setup(
    name='profiler',
    version='0.1.0',
    packages=find_packages(include=['profiler', 'profiler.*']),
    install_requires=[
        'tensorflow',
        'scikit-learn',
    ]
)