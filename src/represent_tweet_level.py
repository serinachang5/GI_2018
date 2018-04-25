import gensim
import numpy as np
import pickle

class TweetLevel:
    def __init__(self, embs, emb_type = 'pkl'):
        assert(emb_type == 'pkl' or emb_type == 'w2v' or emb_type == 'direct')
        if emb_type == 'pkl':
            idx2emb = pickle.load(open(embs, 'rb'))
            print('Number of embeddings in {}: {}'.format(embs, len(idx2emb)))
            self.idx2emb = idx2emb
        elif emb_type == 'w2v':
            wv = gensim.models.KeyedVectors.load_word2vec_format(embs, binary=True)
            print('Number of embeddings in {}: {}'.format(embs, len(wv.vocab)))
            self.idx2emb = dict((idx, wv.vocab[idx]) for idx in wv.vocab)
        else:  # direct
            self.idx2emb = embs

    def get_representation(self, seq, mode = 'avg'):
        if len(seq) == 0:
            return 0
        if type(seq[0]) is int:
            seq = [str(idx) for idx in seq]

        found_embeddings = []
        for idx in seq:
            if idx in self.idx2emb:
                found_embeddings.append(self.idx2emb[idx])
        if len(found_embeddings) == 0:
            return 0

        valid_modes = ['avg', 'sum', 'max', 'min', 'para']
        assert(mode in valid_modes)
        if mode == 'avg':
            return self._get_average(found_embeddings)
        elif mode == 'sum':
            return self._get_sum(found_embeddings)
        elif mode == 'max':
            return self._get_max(found_embeddings)
        elif mode == 'min':
            return self._get_min(found_embeddings)
        else:  # mode == 'para'
            return self._get_average(found_embeddings)

    # average of all embeddings
    def _get_average(self, elist):
        return np.mean(elist, axis=0)

    # sum of all embeddings
    def _get_sum(self, elist):
        return np.sum(elist, axis=0)

    # max per dimension
    def _get_max(self, elist):
        embs = np.array(elist)
        embs_by_dim = embs.T  # one dim per row
        max_per_dim = np.max(embs_by_dim, axis=1)
        return max_per_dim

    # min per dimension
    def _get_min(self, elist):
        embs = np.array(elist)
        embs_by_dim = embs.T  # one dim per row
        min_per_dim = np.min(embs_by_dim, axis=1)
        return min_per_dim


if __name__ == '__main__':
    example_dict = {'1':[1,3,1], '2':[2,2,2]}
    tl = TweetLevel(example_dict, emb_type='direct')
    print(tl.idx2emb)
    seq = [1,2,2]
    for mode in ['avg', 'sum', 'max', 'min']:
        print(mode, tl.get_representation(seq, mode = mode))