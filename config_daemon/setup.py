from setuptools import setup, find_packages

setup(
    name='config_daemon',
    version='0.1.0',
    packages=find_packages(include=['config_daemon', 'config_daemon.*']),
    install_requires=[
        'fastapi==0.68.0',
        'uvicorn==0.15.0',
        'protobuf==3.17.3',
        'requests==2.26.0',
    ]
)