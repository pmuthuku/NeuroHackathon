#!/usr/bin/env/sh

mname=$1
echo "python train_cnn.py --prefix WITHPRE/$mname --pretrained GRID/$mname"
python train_cnn.py --prefix WITHPRE/$mname --pretrained GRID/$mname
