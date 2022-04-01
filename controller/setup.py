from setuptools import setup, find_packages

setup(
    name='controller',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'fastapi',
        'fastapi-utils',
        'httpx',
        'pytest',
        'pytest-asyncio',
        'pytest-httpx',
        'requests',
        'sortedcollections',
        'uvicorn'
    ]
)