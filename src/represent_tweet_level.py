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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

class TweetLevel:
    def __init__(self, word_level = None, wl_file_type = None, d2v_model = None):
        print('Initializing TweetLevel...')

        # store word-level embeddings
        if word_level is not None:
            assert(wl_file_type == 'pkl' or wl_file_type == 'w2v')
            if wl_file_type == 'pkl':
                self.idx2emb = pickle.load(open(word_level, 'rb'))
            else:  # wl_file_type == 'w2v'
                wv = KeyedVectors.load_word2vec_format(word_level, binary=True)
                self.idx2emb = dict((idx, wv[idx]) for idx in wv.vocab)
            print('Number of embeddings in {}: {}'.format(word_level, len(self.idx2emb)))
        else:
            self.idx2emb = None

        # store doc2vec model
        if d2v_model is not None:
            self.d2v_model = Doc2Vec.load(d2v_model)
            print('Number of docvecs in {}: {}'.format(d2v_model, len(self.d2v_model.docvecs)))
        else:
            self.d2v_model = None

        assert(self.idx2emb is not None or self.d2v_model is not None)

        # dictionary of tweet_id to idx_seq
        complete_dict = pickle.load(open('../data/data.pkl', 'rb'))['data']
        self.tweet_dict = dict((tweet_id, complete_dict[tweet_id]['word_int_arr']) for tweet_id in complete_dict)
        print('Sample tweet_dict item:', next(iter(self.tweet_dict.items())))
        print('Size of tweet_dict:', len(self.tweet_dict))

    def get_representation(self, tweet_id, mode = 'avg'):
        if type(tweet_id) is str:
            tweet_id = int(tweet_id)
        assert(tweet_id in self.tweet_dict)
        seq = self.tweet_dict[tweet_id]
        seq = [str(idx) for idx in seq]

        valid_modes = ['avg', 'sum', 'max', 'min', 'd2v']
        assert(mode in valid_modes)

        if mode == 'd2v':
            if self.d2v_model is None:
                print('No d2v_model; cannot get d2v representation')
                return None
            return self._get_docvec(seq)

        # mode is one of the word-level ones
        if self.idx2emb is None:
            print('No word-level embeddings; cannot get', mode, 'representation')
        if len(seq) == 0:
            return self.idx2emb['1']  # neutral
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

    # yield tweet-level reps for all tweets in data.pkl
    def get_all_representations(self, mode = 'avg'):
        for tweet_id in self.tweet_dict:
            yield tweet_id, self.get_representation(tweet_id, mode=mode)

    # inferred document embedding from doc2vec model
    def _get_docvec(self, seq):
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

# write tweet-level representations to file
def write_reps_to_file(emb_type, rep_modes = None):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(word_level='w2v_word_s300_w5_mc5_it20.bin', wl_file_type='w2v')
    elif emb_type == 'splex':
        tl = TweetLevel(word_level='splex_standard_svd_word_s300_seeds_hc.pkl', wl_file_type='pkl')
    else:
        tl = TweetLevel(d2v_model='d2v_word_s300_w5_mc5_ep20.mdl')

    if emb_type == 'w2v' or emb_type == 'splex':
        assert(rep_modes is not None)
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
    else:  # d2v
        fname = '../reps/d2v.txt'
        print('\nWriting embeddings to', fname)
        with open(fname, 'w') as f:
            count = 0
            for id,rep in tl.get_all_representations(mode='d2v'):
                if count % 50000 == 0: print(count)
                f.write(str(id) + '\t')
                rep = [str(x) for x in rep]
                f.write(','.join(rep) + '\n')
                count += 1
        print('Done. Wrote {} embeddings'.format(count))

# check first 100 written embeddings
def check_written_embeddings(emb_type, rep_mode = None):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(word_level='w2v_word_s300_w5_mc5_it20.bin', wl_file_type='w2v')
    elif emb_type == 'splex':
        tl = TweetLevel(word_level='splex_standard_svd_word_s300_seeds_hc.pkl', wl_file_type='pkl')
    else:
        tl = TweetLevel(d2v_model='d2v_word_s300_w5_mc5_ep20.mdl')

    if emb_type == 'w2v' or emb_type == 'splex':
        assert(rep_mode is not None)
        fname = '../reps/' + emb_type + '_' + rep_mode + '.txt'
    else:
        rep_mode = 'd2v'
        fname = '../reps/d2v.txt'

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

# visualize representations of loss vs aggression tweets
# labeled_tweets is a list of (tweet_id, label) tuples
# include_sub only applies to emb_type 'splex' -- whether to include the Substance score or not
def visualize_reps(labeled_tweets, emb_type, rep_mode = None, include_sub = False):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        assert(rep_mode is not None)
        tl = TweetLevel(word_level='w2v_word_s300_w5_mc5_it20.bin', wl_file_type='w2v')
    elif emb_type == 'splex':
        assert(rep_mode is not None)
        tl = TweetLevel(word_level='splex_standard_svd_word_s300_seeds_hc.pkl', wl_file_type='pkl')
    else:
        rep_mode = 'd2v'
        tl = TweetLevel(d2v_model='d2v_word_s300_w5_mc5_ep20.mdl')

    label_to_color = {'Aggression':'r', 'Loss':'b'}
    X = []
    color_map = []
    for (tweet_id, label) in labeled_tweets:
        if label == 'Loss' or label == 'Aggression':
            rep = tl.get_representation(tweet_id, rep_mode)
            if emb_type == 'splex' and include_sub is False:
                rep = rep[:2]
            X.append(rep)
            color_map.append(label_to_color[label])
    X = np.array(X)
    print('Built tweet by embedding matrix of shape', X.shape)

    if X.shape[1] > 2:
        print('Transforming with TSNE...')
        X = TSNE(n_components=2).fit_transform(X)

    print('Plotting X with dimensions', X.shape)
    plt.figure(figsize=(6,6))  # make sure figure is square
    plt.scatter(X[:, 0], X[:, 1], c=color_map)

    specs = emb_type
    if emb_type != 'd2v':
        specs += '_' + rep_mode
    title = 'Visualization of tweet-level embeddings ({})'.format(specs)
    plt.title(title)

    plt.show()


if __name__ == '__main__':
    # modes = ['max', 'min']
    # write_reps_to_file(emb_type='w2v', rep_modes=modes)
    # write_reps_to_file(emb_type='splex', rep_modes=modes)
    # check_written_embeddings(emb_type='splex', rep_mode='avg')
    max_len = 53
    vocab_size = 30000
    option = 'word'
    print('Initializing Data Loader')
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)

    tr, val, tst = dl.cv_data(fold_idx=0)
    labeled_tweets = tr + val + tst
    labeled_tweets = [(x['tweet_id'], x['label']) for x in labeled_tweets]
    print('Number of labeled tweets:', len(labeled_tweets))

    visualize_reps(labeled_tweets, emb_type='w2v', rep_mode='avg')