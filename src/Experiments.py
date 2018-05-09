from nn_experiment import Experiment
from represent_tweet_level import TweetLevel
import numpy as np


def baseline():
    options = ['word']
    experiment = Experiment(experiment_dir='test', input_name2id2np=None, adapt_train_vocab=True,
                            options=options, patience=5)
    experiment.cv()


def with_splex_tl():
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    print('Loaded labeled_tids:', labeled_tids.shape)
    sp_tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    input_name2id2np = {'splex': dict([(tid, sp_tl.get_representation(tid, mode='sum')) for tid in labeled_tids])}

    options = ['word']
    experiment = Experiment(experiment_dir='test', input_name2id2np=input_name2id2np, adapt_train_vocab=True,
                            options=options, patience=5)
    experiment.cv()

if __name__ == '__main__':
    baseline()
