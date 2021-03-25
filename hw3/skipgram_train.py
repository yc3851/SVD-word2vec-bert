import os
from itertools import product
from gensim.models import Word2Vec


if not os.path.exists('models/skipgram'):
    os.makedirs('models/skipgram')

def read_corpus(fname):
    corpus = []
    
    with open(fname, 'r') as fp:
        for line in fp.readlines():
            corpus.append(line.strip('\n').lower().split())
    return corpus


train_file = 'data/brown.txt'
corpus = read_corpus(train_file)

sizes = [50, 100, 300]
windows = [2, 5, 10]
negatives = [1, 5, 15]

for size, window, negative in product(sizes, windows, negatives):
    print(f'Training for the setting: dimension = {size}, content window = {window}, negative samples = {negative}')

    # train model
    model = Word2Vec(sentences=corpus, iter=20, sg=1, size=size, window=window, negative=negative) 

    # save model
    save_dir = 'models/skipgram'
    outfile = f'skipgram_size_{size}_win_{window}_neg_{negative}.model' 
    model.wv.save(os.path.join(save_dir, outfile))
