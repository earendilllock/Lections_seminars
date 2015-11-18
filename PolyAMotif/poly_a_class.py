from SpectralHomogeneousHMMsparse import *
import utils
from liblinearutil import *
import numpy as np


def ExpandSymbol(concatenated_symbol, state_to_index_map):
  expanded_symbol = 0
  base = len(state_to_index_map)
  for symbol in concatenated_symbol:
    expanded_symbol = expanded_symbol * base + state_to_index_map[symbol]
  return expanded_symbol


def GetExpandedSymbolSet(state_set, d=2):
  original_state_number = len(state_set)
  expanded_set = range(original_state_number ** d)
  return expanded_set


def TransformOneSequence(sequence, state_set, d=2):
  state_to_index_map = dict()
  for index, state in enumerate(state_set):
    state_to_index_map[state] = index

  new_sequence = list()
  for i in range(len(sequence)-d+1):
    concatenated_symbol = sequence[i:i+d]
    expanded_symbol = ExpandSymbol(
        concatenated_symbol, state_to_index_map)
    new_sequence.append(expanded_symbol)

  return new_sequence


def TransformSequences(sequences, state_set, d=2):
  new_sequences = list()
  for sequence in sequences:
    new_sequences.append(
        TransformOneSequence(sequence,
                             state_set,
                             d))
  return new_sequences


def GenerateFeatures(data, model):
  feature = list()
  for i, sequence in enumerate(data):
    belief_list = SpectralHMMBeliefs(sequence, model)
    feature.append(np.hstack(belief_list).tolist())

  return feature


def GenerateFeaturesFromModelList(data, model_list):
  feature_model_list = list()
  for model in model_list:
    feature_model_list.append(
        np.array(GenerateFeatures(data, model)))

  return np.hstack(feature_model_list).tolist()


def TransformDNA(dna_sequences, d):
  original_state_set = ['A', 'C', 'G', 'T']
  num_states = 4 ** d

  sequence = TransformSequences(dna_sequences,
                                original_state_set,
                                d)
  return sequence


def BuildHMMModels(train_data, train_lab, d,  k):
  num_states = 4 ** d
  pos_train_data, pos_train_idx = utils.PosData(train_data, train_lab)
  neg_train_data, neg_train_idx = utils.NegData(train_data, train_lab)

  pos_model = SpectralHMMTrain(pos_train_data, num_states, k)
  neg_model = SpectralHMMTrain(neg_train_data, num_states, k)
  hmm_model_list = [pos_model, neg_model]

  return hmm_model_list


def TrainHMMFeatureClassifier(train_data_, train_lab, d, k, C=1):
  """Extract features using HMM and train an SVM classifier from them.

  Params
  -----------
  train_data_:  list of list
    a list of training sequences, each sequence is a list of 'A', 'C', 'G', or
    'T' strings.

  train_lab:  1 dimension numpy.ndarray
    an array of labels for training sequences, 1.0 for positive sequences and
    -1.0 for negative sequences.

  d:  scalar
    number of observations to combine into mega-state.

  k:  scalar
    number of left singular vectors to keep, also similar to number of latetn
    variable states.

  C:  a scalar, optioinal
    the parameter for linear SVM.

  Returns
  -----------
  svm_model:  a svm model class instance
    the trained SVM model.

  hmm_model_list: a list of HMM models
    HMM models trained on positive and negative sequences.
  """
  train_data = TransformDNA(train_data_, d)

  hmm_model_list = BuildHMMModels(train_data, train_lab, d, k)

  # Generate features from HMM models.
  x_train = GenerateFeaturesFromModelList(train_data, hmm_model_list)
  y_train = train_lab.tolist()

  svm_cmd = '-s 2 -c %f' % C
  svm_model = train(y_train, x_train, svm_cmd)

  # Calculate training error.
  train_predict, _, _ = predict(y_train, x_train, svm_model)

  train_acc = np.mean(train_predict == train_lab)
  print 'Training accuracy'
  print train_acc

  return svm_model, hmm_model_list


def PredictHMMClassifier(test_data_, test_lab, d, k,
                         svm_model, hmm_model_list):
  """Predict the labels of testig sequences.

  Params
  -----------
  test_data_:  list of list
    a list of testing sequences, each sequence is a list of 'A', 'C', 'G', or
    'T' strings.

  test_lab:  1 dimension numpy.ndarray
    an array of labels for testing sequences, 1.0 for positive sequences and
    -1.0 for negative sequences. It is only used to calcualte test accuracy.
    If unknown, just supply any random value or all 1's.

  d:  scalar
    number of observations to combine into mega-state.

  k:  scalar
    number of left singular vectors to keep, also similar to number of latetn
    variable states.

  svm_model:  a svm model class instance
    the trained SVM model.

  hmm_model_list: a list of HMM models
    HMM models trained on positive and negative sequences.

  Returns
  -----------
  test_predict:  1-D numpy.ndarray
    The predicted labels for test sequences.

  test_acc:  scalar
    Accurary of predicted test labels using test_lab.

  predict_vals:  1-D numpy.ndarray
    The decision values, aka margins, for test sequences.


  See also
  -----------
  TrainHMMFeatureClassifier
  """
  test_data = TransformDNA(test_data_, d)

  # Generate features from HMM models.
  x_test = GenerateFeaturesFromModelList(test_data, hmm_model_list)
  y_test = test_lab.tolist()

  test_predict_, test_acc, predict_vals_ = predict(y_test, x_test, svm_model)
  test_predict = np.array(test_predict_)
  predict_vals = np.array(predict_vals_)

  return test_predict, test_acc, predict_vals
