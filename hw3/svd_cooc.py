import os
import pickle

import numpy as np
import scipy.sparse as sps
from math import log

# create model directory
if not os.path.exists('models/svd'):
    os.makedirs('models/svd')


def read_corpus(input_file):
    corpus = []

    with open(input_file, 'r') as fp:
        for line in fp.readlines():
            corpus.append(line.strip('\n').lower().split())
    return corpus

def create_index(corpus):
    w2i, i2w = {}, {}
    index = 0

    for line in corpus:
        for word in line:
            if word not in w2i:
                w2i[word] = index
                i2w[index] = word
                index += 1
    return w2i, i2w

def get_context(line, win=2):
    word_ctx_pairs = []
    l = len(line)

    for i, word in enumerate(line):
        left_ctx, right_ctx = [], []
        if i > 0:
            left_ctx = line[max(0,i-win):i]
        if i < l-1:
            right_ctx = line[i+1:min(i+win+1,l)]
        ctx = left_ctx + right_ctx
        word_ctx_pairs.append((word, ctx))
    return word_ctx_pairs

def get_cooc_matrix(corpus, w2i, win):
    vocab_len = len(w2i)
    cooc_matrix = np.zeros((vocab_len, vocab_len))

    for line in corpus:
        word_ctx_pairs = get_context(line, win=win)
        for w, ctx in word_ctx_pairs:
            for c in ctx:
                cooc_matrix[w2i[w]][w2i[c]] += 1
    return sps.lil_matrix(cooc_matrix)	

def get_ppmi_matrix(cooc_matrix):
    # for unigram counts
    row_sum = cooc_matrix.sum(axis=1)
    # for non-contiguous bigrams
    total_sum = int(sum(row_sum)) 

    # calculate ppmi for nonzero entries
    rows, cols = cooc_matrix.nonzero()
    for row, col in zip(rows, cols): 
        cooc_matrix[row, col] = max(0, log(cooc_matrix[row, col] * total_sum / int(row_sum[row] * row_sum[col]), 2))

    return cooc_matrix


if __name__ == '__main__':
    data_file = 'data/brown.txt'
    corpus = read_corpus(data_file)
    w2i, i2w = create_index(corpus)
    pickle.dump(w2i, open('models/svd/w2i.pkl', 'wb'))
    pickle.dump(i2w, open('models/svd/i2w.pkl', 'wb'))

    print('Index created for corpus')

    windows = [2, 5, 10]

    for win in windows:
        print(f'Creating cooccurrence matrix for window = {win}')
        cooc_matrix = get_cooc_matrix(corpus, w2i, win)
        print('Converting to PPMI matrix')
        ppmi_matrix = get_ppmi_matrix(cooc_matrix)

        # save matrix and vocab
        print('Saving PPMI matrix')
        sps.save_npz(f'models/svd/ppmi_matrix_win_{win}.npz', sps.coo_matrix(ppmi_matrix))
