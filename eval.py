import scipy.io
import numpy as np
import os
import argparse
from collections import defaultdict
import pickle
from load_data import load_unlabeled, load_dev, load_test, aggregate_frame_predictions

# parse/validate arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--predictions", type=str, help='frame-level predictions for dev/test sets')
args = argparser.parse_args()

print args.predictions
predictions = pickle.load(open(args.predictions))
print predictions.keys()

dev_predictions, test_predictions = predictions['val'], predictions['test']

for split in ['test', 'dev']:

  # read model predictions
  frame_predictions = dev_predictions if split == 'dev' else test_predictions
  # read neuron ids of each frame
  _, one_hot_label, neuron_ids = load_dev() if split == 'dev' else load_test()
  print 'split=', split
  print 'len(frame_predictions) = ', len(frame_predictions)
  print 'len(neuron_ids) = ', len(neuron_ids)
  assert(len(frame_predictions) == len(neuron_ids))

  # group frame predictions of the same neuron to make a neuron-level prediction
  prev_neuron_id = ''
  neuron_prediction = {}
  neuron_gold = {}
  current_neuron_frames = []
  correct_predictions, all_predictions = 0.0, 0.0
  for i in xrange(len(frame_predictions)):
    if i >= len(neuron_ids): break
    frame_prediction = frame_predictions[i]
    neuron_id = neuron_ids[i]
    if prev_neuron_id != '' and prev_neuron_id != neuron_id:
      # lookup the gold label from the previous frame
      neuron_gold[neuron_id] = frame_predictions[i-1][-1]
      # aggregate frame level predictions into one label for the neuron
      assert(len(current_neuron_frames) > 0)
      neuron_prediction[neuron_id] = aggregate_frame_predictions(current_neuron_frames)
      # reset the buffer
      current_neuron_frames = []
      # evaluate this instance
      all_predictions += 1
      if neuron_prediction[neuron_id] == neuron_gold[neuron_id]: correct_predictions += 1
    prev_neuron_id = neuron_id
    current_neuron_frames.append(frame_prediction)
  
  # print all gold/prediction pairs for all neurons
  for neuron_id in neuron_prediction.keys():
    print 'neuron_id={}, gold={}, predicted={}'.format(neuron_id, neuron_gold[neuron_id], neuron_prediction[neuron_id])

  # report accuracy
  print 'accuracy: {}% ({} correct out of {})'.format(100 * correct_predictions / all_predictions, correct_predictions, all_predictions)
