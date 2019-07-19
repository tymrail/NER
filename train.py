import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import json

from BiLSTM_CRF import BiLSTM_CRF
from CleanData import get_data_large, get_data_toy

torch.manual_seed(1)

gpu_available = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if gpu_available:
    torch.cuda.manual_seed(1)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_large()
TAG_SET_CoNLL, tag_to_ix, word_to_ix, training_data, testa_data, testb_data = get_data_toy()

print(training_data)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

if gpu_available:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    if gpu_available:
        precheck_sent = precheck_sent.cuda()
        precheck_tags = precheck_tags.cuda()
    print(model(precheck_sent))

for epoch in range(20):
    global_loss = 0
    index_t = 0
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        if gpu_available:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()

        loss = model.neg_log_likelihood(sentence_in, targets)

        if gpu_available:
            loss = loss.cuda()

        loss.backward()
        optimizer.step()

        global_loss += loss.item()

        index_t += 1
        if index_t % 100 == 0:
            print("Epoch:" + str(epoch) + ", progess: " + str(index_t / len(training_data) * 100) + "%, avg_loss:" + str(global_loss / index_t))

    print("Epoch:" + str(epoch) + ", avg_loss:" + str(global_loss / len(training_data)))

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
