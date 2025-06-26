#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy matplotlib seaborn scikit-learn
python3 cifar10_cnn.py