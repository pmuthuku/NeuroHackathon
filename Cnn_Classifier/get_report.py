import sys
import cPickle as pickle
from load_data import load_dev, load_test
import numpy as np

prefix = sys.argv[1]
details = int(sys.argv[2])
(X_test, Y_test, _ ) = load_test()

meta_dict = pickle.load(open(prefix  + '.meta'))

accuracy = meta_dict['accuracy']
gold_classes = np.argmax(Y_test, axis = 1)

print accuracy
if details:
	pred_classes = meta_dict['pred_classes']
	length = meta_dict['length']
	correct = meta_dict['correct']

	for l in correct:
		print("label index {} # of predictions {} accuracy {:10.4f}".format(l, length[l], correct[l]*1.0 / length[l]))

