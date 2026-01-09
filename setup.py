"""
Setup script for ContraWiMAE package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Core requirements (minimal dependencies)
core_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "numpy>=1.21.0",
    "PyYAML>=6.0",
    "tqdm>=4.64.0",
    "tensorboard>=2.10.0",
    "scipy>=1.9.0",
]

# Optional extras
docs_requirements = [
    "sphinx>=7.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]

dev_requirements = [
    "pytest>=7.0.0",
] + docs_requirements

setup(
    name="ContraWiMAE",
    version="0.1.0",
    author="Berkay Guler",
    author_email="gulerb@uci.edu",
    description="Official implementation of 'A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning'",
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
    install_requires=core_requirements,
    extras_require={
        "docs": docs_requirements,
        "dev": dev_requirements,
        "all": dev_requirements,
    },
    include_package_data=True,
    zip_safe=False,
) 