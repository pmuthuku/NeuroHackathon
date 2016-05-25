import scipy.io
import numpy as np
import os

def load_dev():
  x_y_pairs = []
  labeled_frames_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'all_frames.mat')
  frames = scipy.io.loadmat(labeled_frames_filename)
  #print frames.keys()
  unique_labels = ['PV_frames', 'Pyr_frames', 'SST_frames']
  for label in unique_labels:
    #print label, len(frames[label])
    for frame_index in xrange(len(frames[label])):
      x_y_pairs.append( (frames[label][frame_index], label,) )

  frames_count = len(x_y_pairs)
  frame_length = len(x_y_pairs[0][0])
  labels_count = len(unique_labels)

  #print 'frames_count=', frames_count
  #print 'frame_length= ', frame_length
  #print 'labels_count=', labels_count

  x, y = [], []
  effective_frames_count = 0 
  for i in xrange(frames_count):
    if i%4 == 0 or i %4 == 1: continue
    effective_frames_count += 1
    x.extend(x_y_pairs[i][0])
    #if i % 100 == 0: print i, x_y_pairs[i][1], unique_labels.index(x_y_pairs[i][1])
    label = np.zeros( (1,labels_count) )
    label[0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  #print 'len(x)=', len(x)
  #print 'len(y)=', len(y)
  x, y = np.matrix(x).reshape( (effective_frames_count, frame_length, 1) ), np.matrix(y).reshape( (effective_frames_count, labels_count, 1) )

  #print 'x.shape = ', x.shape
  #print 'y.shape = ', y.shape

  return (x, y)

def load_test():
  x_y_pairs = []
  labeled_frames_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'all_frames.mat')
  frames = scipy.io.loadmat(labeled_frames_filename)
  unique_labels = ['PV_frames', 'Pyr_frames', 'SST_frames']
  for label in unique_labels:
    for frame_index in xrange(len(frames[label])):
      x_y_pairs.append( (frames[label][frame_index], label,) )

  frames_count = len(x_y_pairs)
  frame_length = len(x_y_pairs[0][0])
  labels_count = len(unique_labels)

  #print 'frames_count=', frames_count
  #print 'frame_length= ', frame_length
  #print 'labels_count=', labels_count

  x, y = [], []
  effective_frames_count = 0 
  for i in xrange(frames_count):
    if i%4 == 2 or i %4 == 3: continue
    effective_frames_count += 1
    x.extend(x_y_pairs[i][0])
    label = np.zeros( (1,labels_count) )
    label[ 0][unique_labels.index(x_y_pairs[i][1]) ] = 1
    y.extend(label)

  #print 'len(x)=', len(x)
  #print 'len(y)=', len(y)
  x, y = np.matrix(x).reshape( (effective_frames_count, frame_length) ), np.matrix(y).reshape( (effective_frames_count, labels_count) )

  #print 'x.shape = ', x.shape
  #print 'y.shape = ', y.shape

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

  #print 'x.shape = ', x.shape
  #print 'y.shape = ', y.shape

  return (x, y)

#load_unlabeled()
#load_test()
#x, y =load_dev()

#for i in xrange(len(y)):
#  print y[i]

#print 'y[0]=', y[0]
#print 'y[',len(y)-1,']=', y[len(y)-1]
#print 'y[',len(y)/2,']=', y[len(y)/2]
