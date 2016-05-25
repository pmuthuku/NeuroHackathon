#!/usr/bin/env/sh
prefix=$1
filter_length=$2

mkdir -p $prefix
for hidden in 256
do
    for n_filter in 8 16 32 64
    do
	echo "THEANO_FLAGS=device=gpu1 python pretrain_cnn.py --prefix ${prefix}/H${hidden}NF${n_filter}FL${filter_length} --hidden $hidden --n-filter ${n_filter} --filter-length ${filter_length}"
	THEANO_FLAGS=device=gpu1 python pretrain_cnn.py --prefix ${prefix}/H${hidden}NF${n_filter}FL${filter_length} --hidden $hidden --n-filter ${n_filter} --filter-length ${filter_length}
    done
done
