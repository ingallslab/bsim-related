#from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req:
    install_requires = req.read().splitlines()

setuptools.setup(
    name="bsim_related", 
    version="0.0.1",
    author='',
    author_email='',
    description="post-processing scripts for bsim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ingallslab/bsim-related.git",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires = install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
