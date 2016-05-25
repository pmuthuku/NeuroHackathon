import scipy.io
import numpy as np
import os

# given frame-level predictions of a neuron, pedict one label for the neuron
def aggregate_frame_predictions(frame_predictions):
  # there must be at least one frame for each neuron
  assert(len(frame_predictions) > 0)
  # gold annotations of all frames of the same neuron should match
  gold_label = frame_predictions[0][-1]
  # sum up the probabilities across all frames
  aggregate_probabilities = [0.0 for i in xrange(len(frame_predictions[0])-1)]
  for frame_prediction in frame_predictions:
    assert frame_prediction[-1] == gold_label
    label_distribution = frame_prediction[:-1]
    for i in xrange(len(label_distribution)):
      aggregate_probabilities[i] += label_distribution[i]
  # find the label with largest aggregate probability
  best_label = (0, aggregate_probabilities[0])
  for i in xrange(1, len(aggregate_probabilies)):
    if aggregate_probabilities[i] > best_label[1]:
      best_label = (i, aggregate_probabilities[i])
  return best_label[0]

# returns a tuple (x, y, z). 
# x is a features matrix, where each row represents features of a frame
# y is a label matrix, where each row represents a one-hot encoding of the label. Since we don't have labels, we use the neuron ID as a proxy task.
# z is a list of neurons (filenames) which correspond to each row in x and y
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

  x, y, z = [], [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[ 0][filenames.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)
    z.append(x_y_pairs[i][1])

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)

  return (x, y, z)

# returns a tuple (x, y, z). 
# x is a features matrix, where each row represents features of a frame
# y is a label matrix, where each row represents a one-hot encoding of the gold label
# z is a list of neurons (filenames) which correspond to each row in x and y
def load_dev():
  dev_filenames = []
  unique_labels = set()
  x_y_pairs = []
  unlabeled_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data')
  files_counter = 0
  for filename in sorted(os.listdir(unlabeled_subdir)):
    # skip non .mat files
    if not filename.endswith('.mat'): continue
    # skip even files
    files_counter += 1
    if files_counter % 2 == 0: 
      continue
    
    # use filename prefix as label (e.g., PV, Pyr, SST)
    dev_filenames.append(filename)
    label = filename.split('_')[0]
    unique_labels.add(label)

    # extract frames of this neuron
    frames = scipy.io.loadmat(os.path.join(unlabeled_subdir, filename))
    frames = frames['frames']
    for frame_index in range(len(frames)):
      x_y_pairs.append( (frames[frame_index], label, filename) )

  frames_count = len(x_y_pairs)
  assert(len(x_y_pairs) > 0)
  frame_length = len(x_y_pairs[0][0])
  unique_labels = list(unique_labels)
  labels_count = len(unique_labels)

  x, y, z = [], [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)
    z.append(x_y_pairs[i][2])

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)
  
  print 'dev files: ', ' '.join(dev_filenames)
  print 'x.shape=', x.shape, ', y.shape=', y.shape, ', len(z)=', len(z)
  return (x, y, z)

# returns a tuple (x, y, z). 
# x is a features matrix, where each row represents features of a frame
# y is a label matrix, where each row represents a one-hot encoding of the gold label
# z is a list of neurons (filenames) which correspond to each row in x and y
def load_test():
  test_filenames = []
  unique_labels = set()
  x_y_pairs = []
  unlabeled_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data')
  files_counter = 0
  for filename in sorted(os.listdir(unlabeled_subdir)):
    # skip non .mat files
    if not filename.endswith('.mat'): continue
    # skip even files
    files_counter += 1
    if files_counter % 2 == 1:
      continue
    
    # use filename prefix as label (e.g., PV, Pyr, SST)
    test_filenames.append(filename)
    label = filename.split('_')[0]
    unique_labels.add(label)

    # extract frames of this neuron
    frames = scipy.io.loadmat(os.path.join(unlabeled_subdir, filename))
    frames = frames['frames']
    for frame_index in range(len(frames)):
      x_y_pairs.append( (frames[frame_index], label, filename) )

  frames_count = len(x_y_pairs)
  assert(len(x_y_pairs) > 0)
  frame_length = len(x_y_pairs[0][0])
  unique_labels = list(unique_labels)
  labels_count = len(unique_labels)

  x, y, z = [], [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)
    z.append(x_y_pairs[i][2])

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)
  
  print 'test files: ', ' '.join(test_filenames)
  print 'x.shape=', x.shape, ', y.shape=', y.shape, ', len(z)=', len(z)
  return (x, y, z)

