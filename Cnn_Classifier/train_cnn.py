from __future__ import print_function
import numpy as np
import theano
np.random.seed(1337)  # for reproducibility

from keras.optimizers import RMSprop
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import sys, json
from keras.models import model_from_json
import cPickle as pickle

from utils import get_parser
from load_data import load_dev, load_test
from collections import defaultdict

parser = get_parser()
p = parser.parse_args()

def sample_training(label_split, n):
	sample_x = []
	sample_y = []
	for label in label_split:
		indexes = np.random.randint(0, len(label_split[label]), n)
		for i in indexes:
			x,y = label_split[label][i]
			sample_x += [x]
			sample_y += [y]

	return np.stack(sample_x, axis = 0).reshape((len(label_split)*n,sample_x[0].shape[0],1)), np.stack(sample_y, axis = 0)

(X_train, Y_train, _) = load_dev()
(X_test, Y_test, _) = load_test()

X_train = np.asarray(X_train, dtype = theano.config.floatX).reshape((X_train.shape[0],X_train.shape[1],1))
Y_train = np.asarray(Y_train, dtype = bool)

label_split = defaultdict(list)
for i,label in enumerate(np.argmax(Y_train, axis = 1)):
	label_split[label] += [(X_train[i], Y_train[i])]

X_test = np.asarray(X_test, dtype = theano.config.floatX).reshape((X_test.shape[0],X_test.shape[1],1))
Y_test = np.asarray(Y_test, dtype = bool)

Ydim = Y_train.shape[1]
Xdim = X_train.shape[1]

print('Build model...')
model = Graph()

if p.pretrained != '':                              ### Load pre-trained model
	params = p.pretrained.strip().split('/')[1]
	hidden = int(params.split('H')[1].split('NF')[0])
	nf = int(params.split('NF')[1].split('FL')[0])
	fl = int(params.split('FL')[1])
else:
	hidden = p.hidden
	nf = p.nb_filter
	fl = p.filter_length

model.add_input(name =  'input', input_shape = (Xdim,1))
model.add_node(Convolution1D( nb_filter= nf ,filter_length = fl , border_mode='valid',activation='relu', subsample_length=1), name='conv1', input='input')
model.add_node(MaxPooling1D(pool_length=2), name='pool1', input='conv1')
model.add_node(Flatten(), name='flatten', input = 'pool1')
model.add_node(Dense(hidden, activation = 'relu'), name = 'dense', input = 'flatten')
model.add_node(Dense(p.hidden, activation = 'relu'), name = 'dense1', input = 'dense')
model.add_node(Dense(Ydim, activation = 'softmax'), name = 'softmax', input = 'dense1')
model.add_output(name = 'output', input = 'softmax')

if p.pretrained != '':                              ### Load pre-trained model
	print('Initializing from a pretrained model {}...'.format(p.pretrained))

	arch = p.pretrained  + '.arch'
	model_filename = p.pretrained + '.model'

	with open(arch) as json_file:
		architecture = json.load(json_file)

	pretrained_model = model_from_json(architecture)
	pretrained_model.load_weights(model_filename)

	for node in pretrained_model.nodes:
		if node == 'softmax':
			break
		model.nodes[node].set_weights(pretrained_model.nodes[node].get_weights())

	get_pretrained = theano.function([pretrained_model.inputs['input'].get_input(train = False)], pretrained_model.nodes['dense'].get_output(train=False), allow_input_downcast=True)
	get_new = theano.function([model.inputs['input'].get_input(train = False)], model.nodes['dense'].get_output(train=False), allow_input_downcast=True)

	### Check if we initialized the weight correctly
	assert( np.array_equal(get_pretrained(X_train[:10]),get_new(X_train[:10])))

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

	eh = {'loss' : [], 'val_loss' : []}
	for j in xrange(20):
		x, y = sample_training(label_split, 32)
		epoch_history = model.fit({'input' : x, 'output' : y}, batch_size = 96, nb_epoch=1, validation_split = 0.2, class_weight = {'output': {0:0.00000001, 1:0.99, 2:0.99}}, verbose = 0)
		for l in ['loss','val_loss']:
			eh[l] +=  epoch_history.history[l]

	train_history['loss'] += [np.mean(eh['loss'])]
	train_history['val_loss'] += [np.mean(eh['val_loss'])]

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

print('testing...')
model.load_weights(PREFIX + '.model')


prediction_train = model.predict({'input' : X_train, 'output' : Y_train}, batch_size = BATCH_SIZE, verbose = 0)
prediction_test = model.predict({'input' : X_test, 'output' : Y_test}, batch_size = BATCH_SIZE, verbose = 0)

pred_classes = np.argmax(prediction_test['output'], axis = 1)
gold_classes = np.argmax(Y_test, axis = 1)

accuracy = np.sum(np.equal( pred_classes , gold_classes)) * 1.0 / Y_test.shape[0]

correct = defaultdict(int)
length = [0]*3

for i in xrange(X_test.shape[0]):
	if pred_classes[i] == gold_classes[i]:
		correct[pred_classes[i]] += 1
	length[pred_classes[i]] += 1

for l in correct:
	print("label index {} # of predictions {} accuracy {:10.4f}".format(l, length[l], correct[l]*1.0 / length[l]))

pickle.dump({'pred_classes' : pred_classes, 'length' : length, 'correct' : correct, 'accuracy' : accuracy, 'train_history' : train_history, 'X_test' : X_test, 'Y_test' : Y_test},open(PREFIX + '.meta', 'w'))

pickle.dump({'val' : np.concatenate([prediction_train['output'], np.argmax(Y_train, axis = 1).reshape((prediction_train['output'].shape[0],1))], axis = 1) , 'test' : np.concatenate([prediction_test['output'], gold_classes.reshape((gold_classes.shape[0],1))], axis = 1)}, open( PREFIX + '.output','w'))

print("Test accuracy for frame prediction {:10.4f}".format(accuracy))
print('DONE!')
