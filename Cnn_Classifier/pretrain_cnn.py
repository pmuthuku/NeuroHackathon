from __future__ import print_function
import numpy as np
import theano
np.random.seed(1337)

from keras.optimizers import RMSprop
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import sys,json
import cPickle as pickle
from utils import get_parser
from load_data import load_unlabeled
parser = get_parser()
p = parser.parse_args()

(X_train, Y_train, _) = load_unlabeled()

X_train = np.asarray(X_train, dtype = theano.config.floatX).reshape((X_train.shape[0],X_train.shape[1],1))
Y_train = np.asarray(Y_train, dtype = bool)

index = range(X_train.shape[0])
np.random.shuffle(index)
X_train = X_train[index]
Y_train = Y_train[index]

Ydim = Y_train.shape[1]
Xdim = X_train.shape[1]

print('Build model...')
model = Graph()
model.add_input(name =  'input', input_shape = (Xdim,1))
model.add_node(Convolution1D( nb_filter= p.nb_filter,filter_length = p.filter_length, border_mode='valid',activation='relu', subsample_length=1), name='conv1', input='input')
model.add_node(MaxPooling1D(pool_length=2), name='pool1', input='conv1')
model.add_node(Flatten(), name='flatten', input = 'pool1')
model.add_node(Dense(p.hidden, activation = 'relu'), name = 'dense', input = 'flatten')
model.add_node(Dense(Ydim, activation = 'softmax'), name = 'softmax', input = 'dense')
model.add_output(name = 'output', input = 'softmax')
optimizer = RMSprop()
model.compile(loss = { 'output' : 'categorical_crossentropy'}, optimizer= optimizer)

PREFIX = p.prefix
EPOCH = p.n_epochs
best_val = float('inf')
pat = 0
PATIENCE = p.patience
BATCH_SIZE = p.batch_size
train_history = {'loss' : [], 'val_loss' : []}

for iteration in xrange(EPOCH):
	train_history['loss'] += [0]

	epoch_history = model.fit({'input' : X_train, 'output' : Y_train}, batch_size=BATCH_SIZE, nb_epoch=1, validation_split = 0.2)
	train_history['loss'] += epoch_history.history['loss']
	train_history['val_loss'] += epoch_history.history['val_loss']

	print("iteration {}/{} : VAL {:10.4f} best VAL {:10.4f} no improvement in {}".format(iteration+1,EPOCH,train_history['val_loss'][-1],best_val,pat))

	if train_history['val_loss'][-1] >= best_val:
		 pat += 1
	else:
		 pat = 0
		 best_val = train_history['val_loss'][-1]
		 model.save_weights(PREFIX + '.model',overwrite = True)
	if pat == PATIENCE:
		 break

with open( PREFIX + '.arch', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights(PREFIX + '.model',overwrite = True)

outfile = open( PREFIX + '.output','w')
pickle.dump({'train_history' : train_history},open(PREFIX + '.meta', 'w'))
print('DONE!')
