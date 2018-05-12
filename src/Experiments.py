from nn_experiment import Experiment
from represent_tweet_level import TweetLevel
from represent_context import Contextifier
import numpy as np


def baseline():
    options = ['word']
    experiment = Experiment(experiment_dir='test', input_name2id2np=None, adapt_train_vocab=True,
                            options=options, patience=5)
    experiment.cv()

def init_splex_tl():
    tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    return tl

def init_context(tweet_dict):
    tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin', tweet_dict=tweet_dict)
    tl_combine = 'avg'
    context_combine = 'avg'
    context_size = 60
    context_hl_ratio = .5
    post_types = [Contextifier.SELF, Contextifier.RT]
    cl = Contextifier(tl, post_types, context_size, context_hl_ratio, context_combine, tl_combine)
    return cl

def nn_with_additions(include_splex = True, include_context = True):
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    print('Loaded labeled_tids:', labeled_tids.shape)

    tweet_dict = None
    input_name2id2np = {}
    if include_splex:
        tl = init_splex_tl()
        tweet_dict = tl.tweet_dict
        input_name2id2np['splex'] = dict((tid, tl.get_representation(tid, mode='sum')) for tid in labeled_tids)
    if include_context:
        cl = init_context(tweet_dict)
        input_name2id2np['context'] = dict((tid, cl.get_context_embedding(tid, keep_stats=False)[0]) for tid in labeled_tids)

    options = ['word']
    experiment = Experiment(experiment_dir='test', input_name2id2np=input_name2id2np, adapt_train_vocab=True,
                            options=options, patience=5)
    experiment.cv()

if __name__ == '__main__':
    nn_with_additions()
