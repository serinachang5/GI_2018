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


def init_splex_tl():
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

def make_input_name2id2np():
    print('Initializing Data Loader...')
    dl = Data_loader()
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    print('Loaded labeled_tids:', labeled_tids.shape)

    input_name2id2np = {}

    splex_tl = init_splex_tl()
    tweet_dict = splex_tl.tweet_dict
    input_name2id2np['splex_tl'] = dict((tid, splex_tl.get_representation(tid, mode='sum')) for tid in labeled_tids)
    print('Built tweet-level splex.')

    w2v_cl = init_context('w2v', dl, tweet_dict)
    user_ct_tweets = w2v_cl.user_ct_tweets
    id_to_location = w2v_cl.id_to_location
    input_name2id2np['w2v_cl'] = dict((tid, w2v_cl.get_context_embedding(tid, keep_stats=False)[0]) for tid in labeled_tids)
    print('Built context-level word2vec.')

    splex_cl = init_context('splex', dl, tweet_dict, user_ct_tweets, id_to_location)
    input_name2id2np['splex_cl'] = dict((tid, splex_cl.get_context_embedding(tid, keep_stats=False)[0]) for tid in labeled_tids)
    print('Build context-level splex.')

    save_file = 'input_name2id2np.pkl'
    pickle.dump(input_name2id2np, open(save_file, 'wb'))
    print('Saved in', save_file)

def add_user_info():
    print('Initializing Data Loader...')
    dl = Data_loader(labeled_only=True)
    labeled_tweets = dl.all_data()
    print('size of labeled data:', len(labeled_tweets))

    tid2post = {}
    tid2mentions = {}
    tid2retweet = {}
    # need to all be arrays, even if singular features
    for tweet in labeled_tweets:
        tid = tweet['tweet_id']
        tid2post[tid] = np.array([tweet['user_post']], dtype=np.int)
        if 'user_mentions' in tweet:
            tid2mentions[tid] = np.array(tweet['user_mentions'], dtype=np.int)
        else:
            tid2mentions[tid] = np.array([0], dtype=np.int)
        if 'user_retweet' in tweet:
            tid2retweet[tid] = np.array([tweet['user_retweet']], dtype=np.int)
        else:
            tid2retweet[tid] = np.array([0], dtype=np.int)

    save_file = 'input_name2id2np.pkl'
    complete_input = pickle.load(open(save_file, 'rb'))
    complete_input['user_post'] = tid2post
    print('post:', list(tid2post.values())[:10])
    complete_input['user_mentions'] = tid2mentions
    print('mentions:', list(tid2mentions.values())[:10])
    complete_input['user_retweet'] = tid2retweet
    print('retweet:', list(tid2retweet.values())[:10])
    pickle.dump(complete_input, open(save_file, 'wb'))
    print('Saved', save_file)


def check_input():
    save_file = 'input_name2id2np.pkl'
    complete_input = pickle.load(open(save_file, 'rb'))
    for input_name in complete_input:
        if input_name_is_user_idx(input_name):
            print(input_name)
            id2np = complete_input[input_name]
            for id in id2np:
                assert('int' in str(type(id2np[id])))

if __name__ == '__main__':
    # make_word_embeds(include_w2v=True, include_splex=False)
    # check_embeds('word_emb_w2v_splex.np')
    add_user_info()