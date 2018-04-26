"""
===================
represent_tweet_level.py
===================
Authors: Serina Chang
Date: 04/25/2018
Generate and write tweet-level embeddings to file.
"""

from data_loader import Data_loader
from gensim.models import KeyedVectors, Doc2Vec
import numpy as np
import pickle


class TweetLevel:
    def __init__(self, embs, file_type = 'pkl', d2v_model = None):
        print('Initializing TweetLevel...')

        assert(file_type == 'pkl' or file_type == 'w2v' or file_type == 'direct')
        if file_type == 'pkl':
            idx2emb = pickle.load(open(embs, 'rb'))
            print('Number of embeddings in {}: {}'.format(embs, len(idx2emb)))
            self.idx2emb = idx2emb
        elif file_type == 'w2v':
            wv = KeyedVectors.load_word2vec_format(embs, binary=True)
            print('Number of embeddings in {}: {}'.format(embs, len(wv.vocab)))
            self.idx2emb = dict((idx, wv[idx]) for idx in wv.vocab)
        else:  # direct
            print('Number of embeddings provided:', len(embs))
            self.idx2emb = embs

        if d2v_model is not None:
            print('Loading Doc2Vec model:', d2v_model)
            self.d2v_model = Doc2Vec.load(d2v_model)
        else:
            self.d2v_model = None

        # dictionary of tweet_id to tweet_properties
        self.tweet_dict = pickle.load(open('../data/data.pkl', 'rb'))['data']
        print('Sample tweet_dict item:', next(iter(self.tweet_dict.items())))
        print('Number of tweets:', len(self.tweet_dict))

    def get_representation(self, tweet_id, mode = 'avg'):
        if type(tweet_id) is str:
            tweet_id = int(tweet_id)

        assert(tweet_id in self.tweet_dict)
        seq = self.tweet_dict[tweet_id]['word_int_arr']
        seq = [str(idx) for idx in seq]

        valid_modes = ['avg', 'sum', 'max', 'min', 'd2v']
        assert(mode in valid_modes)

        if len(seq) == 0:
            return self.idx2emb['1']  # neutral

        if mode == 'd2v':
            return self._get_docvec(seq)

        found_embeddings = []
        for idx in seq:
            if idx in self.idx2emb:
                found_embeddings.append(self.idx2emb[idx])
        if len(found_embeddings) == 0:
            return self.idx2emb['1']  # neutral

        if mode == 'avg':
            return self._get_average(found_embeddings)
        elif mode == 'sum':
            return self._get_sum(found_embeddings)
        elif mode == 'max':
            return self._get_max(found_embeddings)
        else:  # mode == 'min'
            return self._get_min(found_embeddings)

    # yield tweet-level reps for all tweets in data_loader
    def get_all_representations(self, mode = 'avg'):
        for tweet_id in self.tweet_dict:
            yield tweet_id, self.get_representation(tweet_id, mode=mode)

    # inferred document embedding from doc2vec model
    def _get_docvec(self, seq):
        if self.d2v_model is None:
            print('No d2v_model; cannot get docvec')
        return self.d2v_model.infer_vector(seq)

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


def write_reps_to_file(emb_type, rep_modes):
    assert(emb_type == 'w2v' or emb_type == 'splex')
    if emb_type == 'w2v':
        tl = TweetLevel('w2v_word_s300_w5_mc5_it20.bin', file_type='w2v')
    else:
        tl = TweetLevel('splex_normalized_svd_word_s300_seeds_hc.pkl', file_type='pkl')

    for rm in rep_modes:
        fname = '../reps/' + emb_type + '_' + rm + '.txt'
        print('\nWriting embeddings to', fname)
        with open(fname, 'w') as f:
            count = 0
            for id,rep in tl.get_all_representations(mode=rm):
                if count % 50000 == 0: print(count)
                f.write(str(id) + '\t')
                rep = [str(x) for x in rep]
                f.write(','.join(rep) + '\n')
                count += 1
        print('Done. Wrote {} embeddings'.format(count))

# check first 100 written embeddings
def check_written_embeddings(emb_type, rep_mode):
    assert(emb_type == 'w2v' or emb_type == 'splex')
    if emb_type == 'w2v':
        tl = TweetLevel('w2v_word_s300_w5_mc5_it20.bin', file_type='w2v')
    else:
        tl = TweetLevel('splex_normalized_svd_word_s300_seeds_hc.pkl', file_type='pkl')

    fname = '../reps/' + emb_type + '_' + rep_mode + '.txt'
    print('Checking', fname)
    with open(fname, 'r') as f:
        count = 0
        for line in f:
            id, written_emb = line.split('\t')
            written_emb = [float(x) for x in written_emb.split(',')]
            real_emb = tl.get_representation(id, mode=rep_mode)
            assert(np.allclose(written_emb, real_emb))
            count += 1
            if count == 100:
                return


if __name__ == '__main__':
    modes = ['avg', 'sum', 'max', 'min']
    write_reps_to_file(emb_type='w2v', rep_modes=modes)
    write_reps_to_file(emb_type='splex', rep_modes=modes)
    # check_written_embeddings('w2v_word_s300_w5_mc5_it20.bin', file_type='w2v', emb_type='w2v', mode='avg')