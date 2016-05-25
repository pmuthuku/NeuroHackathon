import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 512',type=int, default = 512)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 100',type=int, default = 100)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 10', type=int, default = 10)

	parser.add_argument('--prefix', action='store', dest='prefix',help='prefix of the model, default = DUMMY', default = 'DUMMY')

	parser.add_argument('--pretrained', action='store', dest='pretrained',help='pretrained file path', default = '')


	parser.add_argument('--hidden', action='store', dest='hidden',help='hidden size of relu layer, default = 256', type=int, default = 256)

	parser.add_argument('--n-filter', action='store', dest='nb_filter',help='number of filter, default = 16', type=int, default = 16)

	parser.add_argument('--filter-length', action='store', dest='filter_length',help='filter length, default = 50', type=int, default = 50)

	return parser
