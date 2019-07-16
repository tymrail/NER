import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import json

torch.manual_seed(1)

gpu_available = torch.cuda.is_available()

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

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        if gpu_available:
            self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size).cuda())
        else:
            self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if gpu_available:
            return (torch.randn(2, 1, self.hidden_dim // 2).cuda(), torch.randn(2, 1, self.hidden_dim // 2).cuda())
        else:
            return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        if gpu_available:
            init_alphas = init_alphas.cuda()

        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        if gpu_available:
            lstm_feats = lstm_feats.cuda()
        
        return lstm_feats

    def _score_sentence(self, feats, tags):
        if gpu_available:
            score = torch.zeros(1).cuda()
            tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tags])
        else:
            score = torch.zeros(1)
            tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        if gpu_available:
            init_vvars = init_vvars.cuda()
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def read_data(path):
    with open(path, 'r') as file:
        json_data = json.load(file)
    data_prepare = []
    for data in json_data:
        data_prepare.append((
            data["sentence"],
            data["tag_ner"]
        ))
    return data_prepare

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 10
HIDDEN_DIM = 8

TAG_SET_CoNLL = ["B-LOC", "I-PER", "B-MISC", "I-LOC", "O", "B-ORG", "I-ORG", "I-MISC"]

training_data = read_data("data/CoNLL-2003/train.json")
testa_data = read_data("data/CoNLL-2003/testa.json")
testb_data = read_data("data/CoNLL-2003/testb.json")

word_to_ix = {}
all_data = training_data + testa_data + testb_data
print(training_data[0])
for sentence, tags in all_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {}
for index, tag in enumerate(TAG_SET_CoNLL):
    tag_to_ix[tag] = index
tag_to_ix[START_TAG] = len(TAG_SET_CoNLL)
tag_to_ix[STOP_TAG] = len(TAG_SET_CoNLL) + 1

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

for epoch in range(300):
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

        print("Epoch:" + str(epoch) + ", loss:" + str(loss[0].item()))

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
