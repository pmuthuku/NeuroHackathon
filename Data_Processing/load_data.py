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
  aggregate_probabilites = [0.0 for i in xrange(len(frame_predictions[0])-1)]
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

# load labeled examples for train/dev
def load_dev2():
  x_y_pairs = []
  labeled_frames_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data', 'all_frames.mat')
  frames = scipy.io.loadmat(labeled_frames_filename)
  unique_labels = ['PV_frames', 'Pyr_frames', 'SST_frames']
  for label in unique_labels:
    for frame_index in xrange(len(frames[label])):
      x_y_pairs.append( (frames[label][frame_index], label,) )

  frames_count = len(x_y_pairs)
  assert(len(x_y_pairs) > 0)
  frame_length = len(x_y_pairs[0][0])
  labels_count = len(unique_labels)

  x, y = [], []
  effective_frames_count = 0 
  for i in xrange(frames_count):
    if i%4 == 0 or i %4 == 1: continue
    effective_frames_count += 1
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  x, y = np.matrix(x).reshape( (effective_frames_count, frame_length, 1) ), np.matrix(y).reshape( (effective_frames_count, labels_count, 1) )

  return (x, y)

# load labeled examples for test
def load_test2():
  x_y_pairs = []
  labeled_frames_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data', 'all_frames.mat')
  frames = scipy.io.loadmat(labeled_frames_filename)
  unique_labels = ['PV_frames', 'Pyr_frames', 'SST_frames']
  for label in unique_labels:
    for frame_index in xrange(len(frames[label])):
      x_y_pairs.append( (frames[label][frame_index], label,) )

  frames_count = len(x_y_pairs)
  frame_length = len(x_y_pairs[0][0])
  labels_count = len(unique_labels)

  x, y = [], []
  effective_frames_count = 0 
  for i in xrange(frames_count):
    if i%4 == 2 or i %4 == 3: continue
    effective_frames_count += 1
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[ 0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  x, y = np.matrix(x).reshape( (effective_frames_count, frame_length) ), np.matrix(y).reshape( (effective_frames_count, labels_count) )

  return (x, y)

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
    label[ 0][filenames.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)

  return (x, y)


def load_dev():
  dev_filenames = []
  unique_labels = set()
  x_y_pairs = []
  unlabeled_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data')
  files_counter = 0
  for filename in os.listdir(unlabeled_subdir):
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

  x, y = [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)
  
  print 'dev files: ', ' '.join(dev_filenames)
  return (x, y)

def load_test():
  test_filenames = []
  unique_labels = set()
  x_y_pairs = []
  unlabeled_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training_data')
  files_counter = 0
  for filename in os.listdir(unlabeled_subdir):
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

  x, y = [], []
  for i in xrange(frames_count):
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  x, y = np.asarray(np.matrix(x).reshape( (frames_count, frame_length) ), dtype=float), np.asarray(np.matrix(y).reshape( (frames_count, labels_count) ), dtype=bool)
  
  print 'test files: ', ' '.join(test_filenames)
  return (x, y)

load_test()
