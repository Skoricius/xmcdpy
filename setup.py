from setuptools import find_packages, setup
import os


# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup(
    name='XMCDpy',
    version='1.2.2',
    packages=find_packages(include=['xmcdpy']),
    description='Library for processing XMCD images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luka Skoric',
    license='GNU GENERAL PUBLIC LICENSE',
    install_requires=[
        'numpy>=1.20.2',
        'matplotlib>=3.4.1',
        'scikit-image>=0.18.1',
        'scipy>=1.6.2',
        "opencv-python>=4.4.0.44",
        "PyQt5>=5.15.4"
    ]
)
