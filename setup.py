"""
Setup script for WirelessContrastiveMaskedLearning package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="wireless-contrastive-masked-learning",
    version="0.1.0",
    author="Berkay Guler",
    author_email="gulerb@uci.edu",
    description="Official implementation of 'A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning' (arXiv:2505.09160)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,


    include_package_data=True,
    package_data={
        "wimae": ["configs/*.yaml"],
    },
    zip_safe=False,
) 