from setuptools import setup, find_packages

setup(
    name='worker_agent',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'fastapi',
        'httpx',
        'numpy',
        'pillow',
        'pytest',
        'pytest-httpx',
        'tritonclient[all]',
        'uvicorn'
    ]
)