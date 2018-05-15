"""
===================
nn_feature_support.py
===================
Authors: Serina Chang
Date: 05/14/2018
Preparing features for neural net.
"""

import numpy as np
import pickle
from gensim.models import KeyedVectors
from data_loader import Data_loader
from represent_tweet_level import TweetLevel
from represent_context import Contextifier
from model_def import input_name_is_user_idx
from sklearn.preprocessing import StandardScaler

def init_tl(emb_type):
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin')
    else:
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    return tl

def init_context(emb_type, dl, tweet_dict, user_ct_tweets = None, id_to_location = None):
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin', tweet_dict=tweet_dict)
        tl_combine = 'avg'
        context_combine = 'avg'
        context_size = 60
        context_hl_ratio = .5
    else:
        # default - will not include substance scores
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl', tweet_dict=tweet_dict)
        tl_combine = 'sum'
        context_combine = 'sum'
        context_size = 2
        context_hl_ratio = 1
    post_types = [Contextifier.SELF, Contextifier.RETWEET]

    cl = Contextifier(tl, post_types, context_size, context_hl_ratio, context_combine, tl_combine)

    if user_ct_tweets is None or id_to_location is None:
        user_ct_tweets, id_to_location = cl.assemble_context(dl.all_data())
    cl.set_context(user_ct_tweets, id_to_location)

    return cl


'''Preparing embeddings'''
def make_word_embeds(include_w2v = True, include_splex = False):
    save_file, dim, w2v, splex = 'word_emb', 0, None, None
    if include_w2v:
        w2v = KeyedVectors.load_word2vec_format('../data/w2v_word_s300_w5_mc5_ep20.bin', binary=True)
        dim += 300
        save_file += '_w2v'
    if include_splex:
        splex_w_sub = pickle.load(open('../data/splex_minmax_svd_word_s300_seeds_hc.pkl', 'rb'))
        splex = dict((idx, splex_w_sub[idx][:2]) for idx in splex_w_sub)  # exclude substance scores
        dim += 2
        save_file += '_splex'
    save_file += '.np'

    vocab_size = 40000
    embeds = np.zeros((vocab_size, dim), dtype=np.float)
    for idx in range(1, vocab_size):
        str_idx = str(idx)
        if include_w2v:
            if str_idx in w2v.vocab:
                embeds[idx][:300] = w2v[str_idx]  # first 300 dims
            else:
                embeds[idx][:300] = w2v['1']
        if include_splex:
            if str_idx in splex:
                embeds[idx][-2:] = splex[str_idx]  # last two dims
            else:
                embeds[idx][-2:] = splex['1']

    np.savetxt(save_file, embeds)
    print('Saved embeddings in', save_file)

def check_embeds(fname):
    embeds = np.loadtxt(fname)
    print('Shape:', embeds.shape)
    for i in range(100, 110):
        print(i, embeds[i])

def make_user_embeds(emb_type, num_users):
    assert(emb_type == 'w2v' or emb_type == 'loss_rand' or emb_type == 'agg_rand')
    save_file = 'user_emb_' + str(num_users) + '_' + emb_type + '.np'
    user_dim = 300
    embeds = np.random.rand(num_users, user_dim)

    if emb_type == 'w2v':
        print('Initializing Data Loader...')
        dl = Data_loader()
        tl = init_tl('w2v')
        found = 0
        for user_idx in range(2, num_users):  # reserve 0 for padding (i.e. no user), 1 for unknown user
            tweet_dicts = dl.tweets_by_user(user_idx)  # tweets WRITTEN by this user
            if tweet_dicts is not None and len(tweet_dicts) > 0:
                found += 1
                all_tweets_sum = np.zeros(user_dim, dtype=np.float)
                for tweet_dict in tweet_dicts:
                    tid = tweet_dict['tweet_id']
                    tweet_avg = tl.get_representation(tid, mode='avg')
                    all_tweets_sum += tweet_avg
                all_tweets_avg = all_tweets_sum / len(tweet_dicts)
                embeds[user_idx] = all_tweets_avg
        print('Found tweets for {} out of {} users'.format(found, num_users-2))

    embeds = StandardScaler().fit_transform(embeds)  # mean 0, variance 1
    embeds[0] = np.zeros(user_dim)  # make sure padding is all 0's

    np.savetxt(save_file, embeds)
    print('Saved embeddings in', save_file)


'''Preparing inputs'''
# splex_tl: splex tweet-level, summing word-level splex scores
# w2v_cl: w2v context-level, averaging word-level and tweet-level context scores, 60 days, .5 hl ratio
# splex_cl: splex context-level, summing word-level and tweet-level splex scores, 2 days, 1 hl ratio
# post_user_index: index of user who is posting, 1 if unknown user
# mention_user_index: index of the first user mentioned or 0 if no mentions
# retweet_user_index: index of the user being retweeted or 0 if no user retweet
# time: time features
def make_inputs(num_users):
    print('Initializing Data Loader...')
    dl = Data_loader(labeled_only=True)
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    print('Loaded labeled_tids:', labeled_tids.shape)

    all_inputs = {}

    # USER INPUTS
    tweets = dl.all_data()
    print('Size of data:', len(tweets))
    tid2post = {}
    tid2retweet = {}
    tid2mention = {}
    for tweet in tweets:
        tid = tweet['tweet_id']
        tid2post[tid] = tweet['user_post']
        tid2mention[tid] = tweet['user_mentions'][0] if 'user_mentions' in tweet and len(tweet['user_mentions']) > 0 else None
        tid2retweet[tid] = tweet['user_retweet'] if 'user_retweet' in tweet else None

    all_inputs['post_user_index'] = tid2post
    all_inputs['mention_user_index'] = tid2mention
    all_inputs['retweet_user_index'] = tid2retweet
    edit_user_inputs(all_inputs, num_users)

    # SPLEX AND W2V INPUTS
    splex_tl = init_tl('splex')
    tweet_dict = splex_tl.tweet_dict
    all_inputs['splex_tl'] = dict((tid, splex_tl.get_representation(tid, mode='sum')) for tid in labeled_tids)
    print('Built tweet-level splex.')

    w2v_cl = init_context('w2v', dl, tweet_dict)
    user_ct_tweets = w2v_cl.user_ct_tweets
    id_to_location = w2v_cl.id_to_location
    all_inputs['w2v_cl'] = dict((tid, w2v_cl.get_context_embedding(tid, keep_stats=False)[0]) for tid in labeled_tids)
    print('Built context-level word2vec.')

    splex_cl = init_context('splex', dl, tweet_dict, user_ct_tweets, id_to_location)
    all_inputs['splex_cl'] = dict((tid, splex_cl.get_context_embedding(tid, keep_stats=False)[0]) for tid in labeled_tids)
    print('Build context-level splex.')

    # TIME INPUT
    id2time = pickle.load(open('id2time_feat.pkl', 'rb'))
    all_inputs['time'] = id2time

    save_file = 'all_inputs.pkl'
    pickle.dump(all_inputs, open(save_file, 'wb'))
    print('Saved', save_file)

# change None's to 0
# change user id's >= user_nums to 1
# change all user id's to nd arrays of length 1
def edit_user_inputs(inputs, num_users):
    for input_name in inputs:
        if input_name_is_user_idx(input_name):
            print(input_name)
            for id, user_idx in inputs[input_name].items():
                if user_idx is None:
                    user_idx = 0   # if there is no retweet or mention, user index is 0
                assert 'int' in str(type(user_idx))
                if user_idx >= num_users:
                    user_idx = 1  # if user index is not under num_users, user index is 1
                inputs[input_name][id] = np.array([user_idx])

# take unlabeled tids out of all_inputs.pkl to save space
def edit_inputs_pkl():
    save_file = 'all_inputs.pkl'
    inputs = pickle.load(open(save_file, 'rb'))
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    labeled_tids = set(labeled_tids.flatten())
    print('Size of tid set:', len(labeled_tids))
    for input_name in inputs:
        print(input_name)
        print('original size of id2np:', len(inputs[input_name]))
        only_labeled = {}
        for id,val in inputs[input_name].items():
            if id in labeled_tids:
                only_labeled[id] = val
        inputs[input_name] = only_labeled
        print('new size of id2np:', len(inputs[input_name]))
    pickle.dump(inputs, open(save_file, 'wb'))
    print('Edited inputs, saved in', save_file)

if __name__ == '__main__':
    # make_word_embeds(include_w2v=True, include_splex=False)
    # check_embeds('word_emb_w2v_splex.np')
    # add_user_info()
    # make_user_embeds(emb_type='agg_rand', num_users=700)
    # make_inputs(num_users=700)
    edit_inputs_pkl()