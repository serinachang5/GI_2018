"""
===================
svm_pipeline.py
===================
Authors: Ethan Adams & Serina Chang
Date: 04/30/2018
Pipeline for cross-validating SVM.
"""

import argparse
import numpy as np
import pickle
from represent_context import Contextifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC


def init_contextifier():
    context_size = args['context_size']
    context_hl = args['context_hl']
    use_rt_user = args['use_rt']
    use_mentions = args['use_mentions']
    use_rt_mentions = args['use_rt_mentions']

    word_emb_file = './w2v_word_s300_w5_mc5_it20.bin'
    word_emb_type = 'w2v'
    word_tl_mode= args['word_tl_mode']
    word_cl_mode = args['word_cl_mode']

    splex_file='./splex_standard_svd_word_s300_seeds_hc.pkl'
    splex_tl_mode = args['splex_tl_mode']
    splex_cl_mode = args['splex_cl_mode']

    keep_stats = True

    contextifier = Contextifier(context_size=context_size, context_hl=context_hl,
                                use_rt_user=use_rt_user, use_mentions=use_mentions, use_rt_mentions=use_rt_mentions,
                                word_emb_file=word_emb_file, word_emb_type=word_emb_type, word_emb_mode=word_tl_mode,
                                splex_emb_file=splex_file, splex_emb_mode=splex_tl_mode,
                                keep_stats=keep_stats)

    contextifier.create_user_context_tweets()
    return contextifier

def init_count_model():
    if args['include_unigrams']:
        vocab = [str(idx) for idx in range(1,args['unigram_size']+1)]
        count_model = CountVectorizer(vocabulary=vocab, token_pattern='\d+')  # get top x indices
    else:
        count_model = None
    return count_model

def transform_data(data, ctfr, cm):
    label_to_idx = {'Loss':0, 'Aggression':1, 'Other':2}
    y = np.array([label_to_idx[t['label']] for t in data])

    # build tweet and context representations
    tweet_embs, context_embs = [], []
    for i,tweet in enumerate(data):
        if args['include_tweet_level']:
            tweet_embs.append(ctfr.get_tweet_embedding(tweet['tweet_id']))
        if args['include_context']:
            context_embs.append(ctfr.get_context_embedding(tweet['tweet_id']))

    # stack representations together
    to_stack = []
    if cm is not None:
        sentences = []  # tweets in index form but as strings
        for tweet in data:
            sentences.append(' '.join([str(x) for x in tweet['int_arr']]))
        print('Check sentence0:', sentences[0])
        to_stack.append(cm.transform(sentences))
    if args['include_tweet_level']:
        to_stack.append(csr_matrix(np.array(tweet_embs)))
    if args['include_context']:
        to_stack.append(csr_matrix(np.array(context_embs)))
    X = hstack(to_stack)

    return X, y

def print_scores(cv_scores, args=None):
    if args is not None:
        print(args)
    for fold_i,scores in enumerate(cv_scores):
        print('Fold:', fold_i)

def cross_validate(ctfr, cm):
    scores = []
    total_f1 = 0
    dl = ctfr.dl
    for fold_i in range(5):
        print('Fold:', fold_i)
        tr,val,tst = dl.cv_data(fold_i)
        # if tuning parameters, test on val; else, train on train+val and test on test
        if args['tuning']:
            tst = val
        else:
            tr += val

        print('Transforming training data...')
        X, y = transform_data(tr, ctfr, cm)
        print('Training dimensions:', X.shape, y.shape)

        weights = {0: 0.35, 1: 0.5, 2: 0.15}
        args['weights'] = weights
        clf = SVC(kernel='linear', class_weight=weights)
        clf.fit(X, y)

        print('Transforming testing data...')
        X, y = transform_data(tst, ctfr, cm)
        print('Testing dimensions:', X.shape, y.shape)

        pred = clf.predict(X)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average='macro')
        scores.append([p,r,f])
        print('Macro F1:', round(f, 5))
        total_f1 += f

    print('AVERAGE F1:', round(total_f1/5, 5))
    return scores


def main(args):
    assert(args['include_unigrams'] or args['include_tweet_level'] or args['include_context'])

    ctfr = init_contextifier()
    cm = init_count_model()
    cv_scores = cross_validate(ctfr, cm)

    specs = []
    if args['include_unigrams']:
        specs.append('uni')
        specs.append(str(args['unigram_size']))
    if args['include_tweet_level']:
        specs.append('tl')
    if args['include_context']:
        specs.append('ct')
    if args['tuning']:
        specs.append('tun')
    else:
        specs.append('tst')
    out_file = '../cv_results/' + '_'.join(specs) + '.pkl'

    with open(out_file, 'wb') as f:
        pickle.dump((args, cv_scores), f)
    print('Args and cross-val scores saved to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-iu', '--include_unigrams', type = bool, default = True, help = 'whether to include unigrams')
    parser.add_argument('-it', '--include_tweet_level', type = bool, default = True, help = 'whether to include tweet-level -- if false, all tweet-level parameters are ignored')
    parser.add_argument('-ic', '--include_context', type = bool, default = False, help = 'whether to include context -- if false, all context parameters are ignored')

    parser.add_argument('-usize', '--unigram_size', type = int, default = 5000, help = 'number of unigrams to include')

    parser.add_argument('-csize', '--context_size', type = int, default = 14, help = 'number of days to look back')
    parser.add_argument('-hl', '--context_hl', type = int, default = 1, help = 'half-life of context in days')
    parser.add_argument('-rt', '--use_rt', type = bool, default = True, help = 'User A retweets User B\'s tweet -- if true,this tweet will be counted in User A and User B\'s context')
    parser.add_argument('-men', '--use_mentions', type = bool, default = True, help = 'User A tweets, mentioning User B -- if true, this tweet will be in User A and User B\'s context')
    parser.add_argument('-rtmen', '--use_rt_mentions', type = bool, default = True, help = 'User A retweets User B\'s tweet, which mentioned User C -- if true,this tweet will counted in User A and User C\'s history')

    parser.add_argument('-iw', '--include_word_emb', type = bool, default = True, help = 'whether to include word embeddings in tweet-level')
    parser.add_argument('-wtl', '--word_tl_mode', type = str, default = 'avg', help = 'how to combine word embeddings into tweet-level')
    parser.add_argument('-wcl', '--word_cl_mode', type = str, default = 'avg', help = 'how to combine tweet-level word embeddings into context-level')

    parser.add_argument('-is', '--include_splex', type = bool, default = True, help = 'whether to include splex in tweet-level')
    parser.add_argument('-stl', '--splex_tl_mode', type = str, default = 'sum', help = 'how to combine splex scores into tweet-level')
    parser.add_argument('-scl', '--splex_cl_mode', type = str, default = 'sum', help = 'how to combine tweet-level splex scores into context-level')

    parser.add_argument('-tn', '--tuning', type = bool, default = False, help = 'whether parameters are being tuned -- if true, cross-val will train on train and test on val per folder; if false, cross-val will train on train+val and test on test')

    args = vars(parser.parse_args())
    print(args)

    main(args)