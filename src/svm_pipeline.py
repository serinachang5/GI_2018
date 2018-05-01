"""
===================
svm_pipeline.py
===================
Authors: Ethan Adams & Serina Chang
Date: 05/01/2018
Pipeline for cross-validating SVM.
"""

import argparse
from data_loader import Data_loader
import numpy as np
import pickle
from represent_context import Contextifier
from represent_tweet_level import TweetLevel
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

def init_models(dl):
    models = {}

    if args['include_unigrams']:
        models['count'] = init_count_model()

    tweet_dict = None
    if args['include_w2v_tl']:
        models['w2v_tl'] = init_TL('w2v', tweet_dict)
        tweet_dict = models['w2v_tl'].tweet_dict
    if args['include_splex_tl']:
        models['splex_tl'] = init_TL('splex', tweet_dict)

    user_hist = None
    if args['include_w2v_cl']:
        models['w2v_cl'] = init_CL('w2v', dl, user_hist)
        user_hist = models['w2v_cl'].user_history # need to change param names
    if args['include_splex_cl']:
        models['splex_cl'] = init_CL('splex', dl, user_hist)

    return models

def init_count_model():
    vocab = [str(idx) for idx in range(1,args['unigram_size']+1)]  # get top <unigram_size> indices
    count_model = CountVectorizer(vocabulary=vocab, token_pattern='\d+')
    return count_model

def init_TL(emb_type, tweet_dict):
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_it20.bin', tweet_dict=tweet_dict)
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl', tweet_dict=tweet_dict)
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl', tweet_dict=tweet_dict)
    return tl

def init_CL(emb_type, data_loader, user_history):
    assert(emb_type == 'w2v' or emb_type == 'splex')
    mode = args[emb_type + '_cl_mode']
    size = args[emb_type + '_size']
    hl = args[emb_type + '_hl']
    use_rt_user = args[emb_type + '_use_rt']
    use_mentions = args[emb_type + '_use_mentions']
    use_rt_mentions = args[emb_type + '_use_rt_mentions']
    keep_stats = True

    cl = Contextifier(data_loader, user_history) # params TBD

    cl.create_user_context_tweets()
    return cl


def transform_data(data, models):
    label_to_idx = {'Loss':0, 'Aggression':1, 'Other':2}
    y = np.array([label_to_idx[t['label']] for t in data])

    # model_type mapped to representation matrix
    reps = dict((model_type, []) for model_type in models)

    # build unigram representation matrix
    if 'count' in reps:
        sentences = []  # tweets in index form but as strings
        for tweet in data:
            sentences.append(' '.join([str(x) for x in tweet['int_arr']]))
        print('Check sentence0:', sentences[0])
        reps['count'] = models['count'].transform(sentences)

    # build tweet and context representation matrices
    for tweet in data:
        t_id = tweet['tweet_id']
        if 'w2v_tl' in reps:
            reps['w2v_tl'].append(models['w2v_tl'].get_representation(t_id, args['w2v_tl_mode']))
        if 'splex_tl' in reps:
            reps['splex_tl'].append(models['splex_tl'].get_representation(t_id, args['splex_tl_mode']))
        if 'w2v_cl' in reps:
            reps['w2v_cl'].append(models['w2v_cl'].get_representation(t_id, args['w2v_cl_mode']))
        if 'splex_cl' in reps:
            reps['splex_cl'].append(models['splex_cl'].get_representation(t_id, args['splex_cl_mode']))

    to_stack = []
    # sort by name of representation: splex_cl, splex_tl, w2v_cl, w2v_tl, unigrams
    for rep_type, rep in sorted(reps.items(), key=lambda x: x[0]):
        if rep_type == 'count':
            to_stack.append(rep)
        else:
            to_stack.append(csr_matrix(np.array(rep)))
    X = hstack(to_stack)

    return X, y

def cross_validate(dl, models):
    scores = []
    total_f1 = 0
    for fold_i in range(5):
        print('Fold:', fold_i)
        tr,val,tst = dl.cv_data(fold_i)

        # if tuning parameters, test on val; else, train on train+val and test on test
        if args['tuning']:
            tst = val
        else:
            tr += val

        print('Transforming training data...')
        X, y = transform_data(tr, models)
        print('Training dimensions:', X.shape, y.shape)

        weights = {0: 0.35, 1: 0.5, 2: 0.15}
        args['weights'] = weights
        clf = SVC(kernel='linear', class_weight=weights)
        clf.fit(X, y)

        print('Transforming testing data...')
        X, y = transform_data(tst, models)
        print('Testing dimensions:', X.shape, y.shape)

        pred = clf.predict(X)
        per_class = precision_recall_fscore_support(y, pred, average=None)
        macros = precision_recall_fscore_support(y, pred, average='macro')
        scores.append(per_class)
        print('Loss F1: {}. Agg F1: {}. Other F1: {}. Macro F1: {}.'.format(round(per_class[2][0],5), round(per_class[2][1],5),
                                                                            round(per_class[2][2],5), round(macros[2],5)))
        total_f1 += macros[2]

    print('AVERAGE F1:', round(total_f1/5, 5))
    return scores

# print precision, recall, and F1 per class in each fold if verbose
# print macro F1 in each fold
# print final average of macro F1's
def print_scores(per_class, verbose=True):
    avg_mac_f = 0.0
    for fold_i, fold in enumerate(per_class):
        print('Fold:', fold_i)
        if verbose:
            for metric, metric_results in zip(['Precision', 'Recall', 'F1'], fold):
                l, a, o = metric_results
                print(metric, '- Loss: {}. Agg: {}. Other: {}.'.format(round(l, 5), round(a, 5), round(o, 5)))
        print('Macro F1:', round(np.mean(fold[2]), 5))
        avg_mac_f += np.mean(fold[2])
        print()
    print('AVG MACRO F1:', round(avg_mac_f/len(per_class), 5))

def main(args):
    # check representation specifications
    specs = []
    if args['include_unigrams']:
        specs.append('uni')
        specs.append(str(args['unigram_size']))
    if args['include_w2v_tl']:
        specs.append('wt')
    if args['include_splex_tl']:
        specs.append('st')
    if args['include_w2v_cl']:
        specs.append('wc')
    if args['include_splex_cl']:
        specs.append('sc')
    assert(len(specs) > 0)

    if args['tuning']:
        specs.append('tun')
    else:
        specs.append('tst')

    # initialize data_loader
    max_len = 53
    vocab_size = 30000
    option = 'word'
    print('Initializing Data Loader')
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)

    # initialize representation models
    models = init_models(dl)

    # run cv experiment using these representations
    cv_scores = cross_validate(dl, models)

    # save results
    out_file = '../cv_results/' + '_'.join(specs) + '.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump((args, cv_scores), f)
    print('Args and cross-val scores saved to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-iu', '--include_unigrams', type = bool, default = False, help = 'whether to include unigrams')
    parser.add_argument('-usize', '--unigram_size', type = int, default = 5000, help = 'number of unigrams to include')

    parser.add_argument('-iwt', '--include_w2v_tl', type = bool, default = True, help = 'whether to include w2v embeddings at tweet-level; if false, w2v-tweet params are ignored')
    parser.add_argument('-wtmode', '--w2v_tl_mode', type = str, default = 'avg', help = 'how to combine w2v embeddings at tweet-level')

    parser.add_argument('-ist', '--include_splex_tl', type = bool, default = True, help = 'whether to include splex at tweet-level')
    parser.add_argument('-stmode', '--splex_tl_mode', type = str, default = 'sum', help = 'how to combine splex scores into tweet-level')

    parser.add_argument('-iwc', '--include_w2v_cl', type = bool, default = False, help = 'whether to include w2v embeddings in context; if false, w2v-context params are ignored')
    parser.add_argument('-wcmode', '--w2v_cl_mode', type = str, default = 'avg', help = 'how to combine tweet-level w2v embeddings into context-level')
    parser.add_argument('-wcsize', '--w2v_size', type = int, default = 14, help = 'w2v-context: number of days to look back')
    parser.add_argument('-wchl', '--w2v_hl', type = int, default = 1, help = 'w2v-context: half-life in days')
    parser.add_argument('-wcrt', '--w2v_use_rt', type = bool, default = True, help = 'w2v-context: User A retweets User B\'s tweet -- if true,this tweet will be counted in User A and User B\'s context')
    parser.add_argument('-wcmen', '--w2v_use_mentions', type = bool, default = True, help = 'w2v-context: User A tweets, mentioning User B -- if true, this tweet will be in User A and User B\'s context')
    parser.add_argument('-wcrtmen', '--w2v_use_rt_mentions', type = bool, default = True, help = 'w2v-context: User A retweets User B\'s tweet, which mentioned User C -- if true,this tweet will counted in User A and User C\'s history')

    parser.add_argument('-isc', '--include_splex_cl', type = bool, default = False, help = 'whether to include splex in context; if false, splex-context params are ignored')
    parser.add_argument('-scmode', '--splex_cl_mode', type = str, default = 'sum', help = 'how to combine tweet-level splex scores into context-level')
    parser.add_argument('-scsize', '--splex_size', type = int, default = 14, help = 'splex-context: number of days to look back')
    parser.add_argument('-schl', '--splex_context_hl', type = int, default = 1, help = 'splex-context: half-life in days')
    parser.add_argument('-scrt', '--splex_use_rt', type = bool, default = True, help = 'splex-context: User A retweets User B\'s tweet -- if true,this tweet will be counted in User A and User B\'s context')
    parser.add_argument('-scmen', '--splex_use_mentions', type = bool, default = True, help = 'splex-context: User A tweets, mentioning User B -- if true, this tweet will be in User A and User B\'s context')
    parser.add_argument('-scrtmen', '--splex_use_rt_mentions', type = bool, default = True, help = 'splex-context: User A retweets User B\'s tweet, which mentioned User C -- if true,this tweet will counted in User A and User C\'s history')

    parser.add_argument('-tn', '--tuning', type = bool, default = True, help = 'whether parameters are being tuned -- if true, cross-val will train on train and test on val per folder; if false, cross-val will train on train+val and test on test')

    args = vars(parser.parse_args())
    print(args)

    # main(args)