import numpy as np


def PosData(data, lab):
  pos_data = [data[i] for i in range(len(lab)) if lab[i] > 0]
  pos_idx = [i for i in range(len(lab)) if lab[i] > 0]
  return pos_data, pos_idx


def NegData(data, lab):
  neg_data = [data[i] for i in range(len(lab)) if lab[i] < 0]
  neg_idx = [i for i in range(len(lab)) if lab[i] < 0]
  return neg_data, neg_idx


def EvaluateResults(predict, gnd):
  true_predict = predict == gnd
  true_pos = np.logical_and(true_predict, predict == 1)
  true_neg = np.logical_and(true_predict, predict == -1)
  gnd_pos = gnd == 1
  gnd_neg = gnd == -1

  accuracy = np.mean(true_predict)
  sensitivity = np.sum(true_pos) / float(np.sum(gnd_pos))
  specificity = np.sum(true_neg) / float(np.sum(gnd_neg))

  return accuracy, sensitivity, specificity
