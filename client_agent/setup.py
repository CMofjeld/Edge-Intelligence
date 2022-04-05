from setuptools import setup, find_packages

setup(
    name='client_agent',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'httpx',
        'opencv-python',
        'pytest',
        'pytest-httpx'
    ]
)