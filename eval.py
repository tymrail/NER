import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import json

from BiLSTM_CRF import BiLSTM_CRF
from CleanData import get_data_large, get_data_toy

import time

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_toy()

model_loaded = torch.load("saved_models/toy_model_8.pkl")
model_loaded.eval()
toy_training_data = [training_data[0]]

with torch.no_grad():
    recall_train_denominator = 0
    precision_train_denominator = 0
    acc_train_numerator = 0
    # for sentence, tags in training_data:
    for sentence, tags in toy_training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        real_tagsets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).tolist()
        predicted_tagsets = model_loaded(sentence_in)[1]
        print(real_tagsets)
        print(predicted_tagsets)
        for real_tag, predicted_tag in zip(real_tagsets, predicted_tagsets):
            if int(real_tag) != 7:
                recall_train_denominator += 1
                if int(real_tag) == int(predicted_tag):
                    acc_train_numerator += 1
            if int(predicted_tag) != 7:
                precision_train_denominator += 1
    recall = acc_train_numerator / recall_train_denominator
    precision = acc_train_numerator / precision_train_denominator
    F1_score = 2 * recall * precision / (recall + precision)

    print(recall)
    print(precision)
    print(F1_score)
    
