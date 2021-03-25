import os
import pickle
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import svds


if not os.path.exists('models/svd'):
    os.makedirs('models/svd')

def get_svd_embeddings(ppmi_matrix, size=50):
    # truncated SVD 
    u, s, vt = svds(ppmi_matrix, k=size)

    # take product to get word vectors
    w = u.dot(np.diag(np.sqrt(s)))
    return w

def save_svd_embeddings(w, size, win):
    i2w = pickle.load(open('models/svd/i2w.pkl', 'rb'))
    outfile = f'svd_size_{size}_win_{win}.txt'

    with open(os.path.join('models/svd', outfile), 'w') as fp:
        for idx, word in i2w.items():
            vec = ' '.join([f'{item}' for item in w[idx]])
            fp.write(f'{word} {vec}\n')
    print(f'SVD embeddings saved to {outfile}')
    return


if __name__ == '__main__':
    windows = [2, 5, 10]
    sizes = [50, 100, 300]

    for win in windows:
        print('Load PPMI matrix')
        matrix_file = f'ppmi_matrix_win_{win}.npz'
        ppmi_matrix = sps.lil_matrix(sps.load_npz(os.path.join('models/svd', matrix_file)))

        for size in sizes:
            print(f'Perform truncated SVD for setting: size = {size}, window = {win}')
            w = get_svd_embeddings(ppmi_matrix, size)
            # save embeddings
            save_svd_embeddings(w, size, win)
            print('==========================')
