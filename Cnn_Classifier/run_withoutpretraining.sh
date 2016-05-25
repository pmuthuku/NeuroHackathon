#!/usr/bin/env/sh

mname=$1


hidden=`echo $mname | cut -d "N" -f1 | cut -d"H" -f2`
nf=`echo $mname | cut -d "F" -f2`
fl=`echo $mname | cut -d "L" -f2`
echo "python train_cnn.py --prefix WITHOUTPRE/$mname --hidden $hidden --n-filter $nf --filter-length $fl"
python train_cnn.py --prefix WITHOUTPRE/$mname --hidden $hidden --n-filter $nf --filter-length $fl
