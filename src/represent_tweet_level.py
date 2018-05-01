"""
===================
represent_tweet_level.py
===================
Authors: Serina Chang
Date: 5/01/2018
Generate and visualize tweet-level embeddings, write them to file.
"""

from data_loader import Data_loader
from gensim.models import KeyedVectors, Doc2Vec
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE


class TweetLevel:
    def __init__(self, emb_file, tweet_dict = None):
        print('Initializing TweetLevel...')
        # store embeddings
        if 'splex' in emb_file:
            self.emb_type = 'splex'
            self.idx2emb = pickle.load(open(emb_file, 'rb'))
            print('Number of word vectors in {}: {}'.format(emb_file, len(self.idx2emb)))
        elif 'w2v' in emb_file:
            self.emb_type = 'w2v'
            wv = KeyedVectors.load_word2vec_format(emb_file, binary=True)
            self.idx2emb = dict((idx, wv[idx]) for idx in wv.vocab)
            print('Number of word vectors in {}: {}'.format(emb_file, len(self.idx2emb)))
        elif 'd2v' in emb_file:
            self.emb_type = 'd2v'
            self.d2v = Doc2Vec.load(emb_file)
            print('Number of doc vectors in {}: {}'.format(emb_file, len(self.d2v.docvecs)))
        else:
            raise ValueError('Cannot init TweetLevel with', emb_file)

        # dictionary of tweet_id to idx_seq
        if tweet_dict is None:
            complete_dict = pickle.load(open('../data/data.pkl', 'rb'))['data']
            tweet_dict = dict((tweet_id, complete_dict[tweet_id]['word_int_arr']) for tweet_id in complete_dict)
            print('Built tweet_dict. Sample tweet_dict item:', next(iter(tweet_dict.items())))
        self.tweet_dict = tweet_dict
        print('Size of tweet_dict:', len(self.tweet_dict))

    def get_representation(self, tweet_id, mode = 'avg'):
        if type(tweet_id) is str:
            tweet_id = int(tweet_id)
        assert(tweet_id in self.tweet_dict)

        seq = self.tweet_dict[tweet_id]
        seq = [str(idx) for idx in seq]

        if self.emb_type == 'd2v':
            return self._get_docvec(seq)

        # get word-level embeddings
        if len(seq) == 0:
            return self.get_neutral_word_level()
        found_embeddings = []
        for idx in seq:
            if idx in self.idx2emb:
                found_embeddings.append(self.idx2emb[idx])
        if len(found_embeddings) == 0:
            return self.get_neutral_word_level()

        # combine word-level embeddings
        if mode == 'avg':
            return self._get_average(found_embeddings)
        elif mode == 'sum':
            return self._get_sum(found_embeddings)
        elif mode == 'max':
            return self._get_max(found_embeddings)
        elif mode == 'min':
            return self._get_min(found_embeddings)
        else:
            raise ValueError('Invalid word-level mode:', mode)

    # yield tweet-level reps for all tweets in data.pkl
    def get_all_representations(self, mode = 'avg'):
        for tweet_id in self.tweet_dict:
            yield tweet_id, self.get_representation(tweet_id, mode=mode)

    # get dimension of tweet-level representation
    def get_dimension(self):
        if self.emb_type == 'd2v':
            sample_vec = self._get_docvec(['1'])
        else:
            sample_vec = self.get_neutral_word_level()
        return sample_vec.shape[0]

    # get representation of neutral word
    def get_neutral_word_level(self):
        assert(self.idx2emb is not None and '1' in self.idx2emb)
        return self.idx2emb['1']

    # private: inferred vector from doc2vec model, given list of indices in doc
    def _get_docvec(self, seq):
        return self.d2v.infer_vector(seq)

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


def test_TL(emb_type):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

    print(tl.get_dimension())
    tweet_dict = tl.tweet_dict
    sample_id = list(tweet_dict.keys())[0]
    print(sample_id)
    print(tl.get_representation(sample_id))

    tl2 = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin', tweet_dict=tweet_dict)
    for mode in ['avg', 'sum', 'min', 'max']:
        print(mode, sum(tl2.get_representation(sample_id, mode=mode)))

# write tweet-level representations to file
def write_reps_to_file(emb_type, rep_modes = None):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

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
    else:
        fname = '../reps/d2v.txt'
        print('\nWriting embeddings to', fname)
        with open(fname, 'w') as f:
            count = 0
            for id,rep in tl.get_all_representations():  # no mode to specify
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
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

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
def visualize_reps(labeled_tweets, emb_type, rep_mode = None, include_sub = True):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin')
        assert(rep_mode is not None)
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
        assert(rep_mode is not None)
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

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

    visualize_reps(labeled_tweets, emb_type='splex', rep_mode='sum')

    # test_TL('d2v')