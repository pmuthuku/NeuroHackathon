#!/usr/bin/env/sh

ln -s ../Data_Processing/testing_data
ln -s ../Data_Processing/training_data
ln -s ../Data_Processing/load_data.py
mkdir .keras ; cd .keras ; git clone https://github.com/wolet/keras.git
ln -s .keras/keras
