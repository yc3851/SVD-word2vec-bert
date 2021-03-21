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

    return embedding



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

    print('[evaluate] MSR accuracy:')
    msr = eval_msr()
    print(msr)
