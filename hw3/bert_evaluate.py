'''
Embedding evaluation script for COMS W4705 Spring 2021 HW3.

Modified from evaluate.py

You can run this script from command-line on one file at a time using

    $ python evaluate.py (name-of-bert-model) ;

'''
import argparse
import numpy as np
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from numpy.linalg import norm

import random
import os
import re

from process import load_model, load_msr

import torch
from transformers import BertTokenizer, BertModel

model_name = 'bert-base-uncased'
# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# Load pre-trained model, set output_hidden_states to true
model = BertModel.from_pretrained(model_name, output_hidden_states = True,)
# Put the model in "evaluation" mode.
model.eval()


def sentence_embedding(sents):
    # adding CLS and SEP
    text = ' '.join(sents)
    marked_text = "[CLS] " + text + " [SEP]"

    # tokenize sentence
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token to indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # add the sentence belonging
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # get the hidden_states of the text
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    # average the second to last hiden layer
    token_vecs = hidden_states[-2][0]
    embedding = torch.mean(token_vecs, dim=0)

    return np.array(embedding)


def bat_model(dir):
    vocab = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(".txt"):
                for line in open(os.path.join(root, name), 'r').readlines():
                    splits = re.split(r'(\\|\/|\t|\s)\s*', line.strip())
                    for item in splits:
                        if item not in vocab:
                            vocab.append(item)
                continue
            else:
                continue

    indices = {}
    for i in range(len(vocab)): indices[vocab[i]] = i

    matrix = []
    for w in vocab:
        matrix.append(sentence_embedding(w))
    return np.array(matrix), vocab, indices


def eval_wordsim(f='data/wordsim353/combined.tab'):
    '''
        Evaluates a trained embedding model on WordSim353 using cosine
        similarity and Spearman's rho. Returns a tuple containing
        (correlation, p-value).
    '''
    sim = []
    pred = []

    for line in open(f, 'r').readlines()[1:]:
        splits = line.split('\t')
        w1 = splits[0]
        w2 = splits[1]
        sim.append(float(splits[2]))
        v1 = sentence_embedding(w1)
        v2 = sentence_embedding(w2)
        pred.append(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)


def eval_bats_file(matrix, vocab, indices, f, repeat=False,
                   multi=0):
    '''
        Evaluates a trained embedding model on a single BATS file using either
        3CosAdd (the classic vector offset cosine method) or 3CosAvg (held-out
        averaging).

        If multi is set to zero or None, this function will usee 3CosAdd;
        otherwise it will use 3CosAvg, holding out (multi) samples at a time.

        Default behavior is to use 3CosAdd.
    '''
    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs]
    pairs = [[p[0], [w for w in p[1]]] for p in pairs]
    pairs = [p for p in pairs if len(p[1]) > 0]
    if len(pairs) <= 1: return None

    transposed = np.transpose(np.array([x / norm(x) for x in matrix]))

    if not multi:
        qa = []
        qb = []
        qc = []
        targets = []
        exclude = []
        groups = []

        for i in range(len(pairs)):
            j = random.randint(0, len(pairs) - 2)
            if j >= i: j += 1
            a = sentence_embedding(pairs[i][0])
            c = sentence_embedding(pairs[j][0])
            for bw in pairs[i][1]:
                qa.append(a)
                qb.append(sentence_embedding(bw))
                qc.append(c)
                groups.append(i)
                targets.append(pairs[j][1])
                exclude.append([pairs[i][0], bw, pairs[j][0]])

        for queries in [qa, qb, qc]:
            queries = np.array([x / norm(x) for x in queries])

        sa = np.matmul(qa, transposed) + .0001
        sb = np.matmul(qb, transposed)
        sc = np.matmul(qc, transposed)
        sims = sb + sc - sa

        # exclude original query words from candidates
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    else:
        offsets = []
        exclude = []
        preds = []
        targets = []
        groups = []

        for i in range(len(pairs) // multi):
            qa = [pairs[j][0] for j in range(len(pairs)) if j - i not in range(multi)]
            qb = [[w for w in pairs[j][1]] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            a = np.mean([sentence_embedding(w) for w in qa], axis=0)
            b = np.mean([np.mean([sentence_embedding(w) for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                c = sentence_embedding(pairs[i + k][0])
                c = c / norm(c)
                offset = b + c - a
                offsets.append(offset / norm(offset))
                targets.append(pairs[i + k][1])
                exclude.append(qa + qbs + [pairs[i + k][0]])
                groups.append(len(groups))

        print(np.shape(transposed))

        sims = np.matmul(np.array(offsets), transposed)
        print(np.shape(sims))
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    preds = [vocab[np.argmax(x)] for x in sims]
    accs = [1 if preds[i].lower() in targets[i] else 0 for i in range(len(preds))]
    regrouped = np.zeros(np.max(groups) + 1)
    for a, g in zip(accs, groups):
        regrouped[g] = max(a, regrouped[g])
    return np.mean(regrouped)

def eval_bats(matrix, vocab, indices):
    '''
        Evaluates a trained embedding model on BATS.

        Returns a dictionary containing
        { category : accuracy score over the category }, where "category" can
        be
            - any of the low-level category names (i.e. the prefix of any of
              the individual data files)
            - one of the four top-level categories ("inflectional_morphology",
              "derivational_morphology", "encyclopedic_semantics",
              "lexicographic_semantics")
            - "total", for the overall score on the entire corpus
    '''
    accs = {}
    base = 'data/BATS'
    for dr in os.listdir('data/BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            dk = dr.split('_', 1)[1].lower()
            accs[dk] = []
            for f in os.listdir(os.path.join(base, dr)):
                accs[f.split('.')[0]] = eval_bats_file(matrix, vocab, indices, os.path.join(base, dr, f))
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs


def eval_msr():
    '''
        Evaluates a trained embedding model on the MSR paraphrase task using
        logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train = [[sentence_embedding(ss[0]), sentence_embedding(ss[1])] for ss in X_tr]
    test = [[sentence_embedding(ss[0]), sentence_embedding(ss[1])] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)


if __name__ == "__main__":

    print('[evaluate] WordSim353 correlation:')
    ws = eval_wordsim()
    print(ws)

    print('[evaluate] Collecting bat matrix...')
    matrix, vocab, indices = bat_model("data/BATS")

    print('[evaluate] BATS accuracies:')
    bats = eval_bats(matrix, vocab, indices)
    print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr()
    print(msr)
