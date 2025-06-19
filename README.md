# RF Device Fingerprinter

A machine learning framework for radio frequency signal analysis using deep learning techniques.

## Overview

This project implements neural network architectures for various RF signal analysis tasks including classification, modulation recognition, device fingerprinting, and signal generation. Built to work with the RadioML dataset and other RF signal sources.

## Features

- **Flexible Architecture**: CNN-LSTM hybrid models for spatial and temporal feature extraction
- **Multiple Tasks**: Supports classification, fingerprinting, generation, and custom analysis
- **RadioML Integration**: Built for RadioML 2016.10A and 2018.01A datasets
- **Signal Processing Pipeline**: Comprehensive preprocessing and augmentation capabilities
- **Scalable Training**: GPU/cloud computing support for large-scale experiments

## Project Structure# rf-signal-classifier

rf-signal-classifier/
├── src/                 # Source code
├── data/                # Datasets (not tracked in git)
├── notebooks/           # Jupyter notebooks for exploration
├── configs/             # Configuration files
├── docs/                # Documentation
├── tests/               # Unit tests
└── README.md           # This file

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, SciPy for signal processing
- GPU recommended for training

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/rf-signal-classifier.git
cd rf-signal-classifier

Install dependencies:
pip install -r requirements.txt

Download RadioML dataset and place files in the data directory.

## Development Status

This project is currently in early development phase.

Current focus areas:
- Data pipeline implementation
- Model architecture design
- Experimental framework setup
- Task definition and requirements gathering

## Research Foundation

This project builds on established research in automatic modulation classification, deep learning for RF signal analysis, CNN-LSTM architectures for time-series data, and RadioML benchmarking standards.

## Contributing

This is an active research project. Please coordinate with the development team before making changes.

## License

To be determined.

## Contact

Contact information to be added.
