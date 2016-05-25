import scipy.io
import numpy as np
import os

def load_unlabeled():
  filenames = []
  x_y_pairs = []
  unlabeled_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testing_data')
  for filename in os.listdir(unlabeled_subdir):
    if not filename.endswith('.mat'): continue
    filenames.append(filename)
    frames = scipy.io.loadmat(os.path.join(unlabeled_subdir, filename))
    frames = frames['frames']
    for frame_index in range(len(frames)):
      x_y_pairs.append( (frames[frame_index], filename) )

  frames_count = len(x_y_pairs)
  frame_length = len(x_y_pairs[0][0])
  labels_count = len(filenames)

  x, y = [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[ filenames.index(x_y_pairs[0][1]) ] = 1
    y.extend(label)

  x, y = np.matrix(x).reshape( (frames_count, frame_length) ), np.matrix(y).reshape( (frames_count, labels_count) )

  print 'x.shape = ', x.shape
  print 'y.shape = ', y.shape

  return (x, y)

load_unlabeled()
