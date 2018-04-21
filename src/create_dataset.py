from sentence_tokenizer import int_array_rep
import pandas as pd
from dateutil import parser
import time
import random
import pickle as pkl

# a function that checks whether the idx dictionary satisfies the criteria
def assert_idx_correctness(idx_dictionary):
    cv_dictionary = idx_dictionary['cross_val']
    test_idxes = set()
    fold = len(cv_dictionary)
    for fold_idx in range(fold):
        test_ind = set(cv_dictionary[fold_idx]['test_ind'])
        if len(test_ind & test_idxes) != 0:
            raise ValueError('Test indexes overlap between folds')
        test_idxes = test_idxes | test_ind
        train_ind, val_ind = set(cv_dictionary[fold_idx]['train_ind']), set(cv_dictionary[fold_idx]['val_ind'])
        if len(train_ind) + len(val_ind) + len(test_ind) != len(train_ind | val_ind | test_ind):
            raise ValueError('train, val, test overlaps within a fold')
        cv_idxes = train_ind | val_ind | test_ind
    heldout_test_ind = set(idx_dictionary['heldout_test_ind'])
    ensemble_ind = set(idx_dictionary['ensemble_ind'])
    if len(cv_idxes) + len(heldout_test_ind) + len(ensemble_ind) != len(cv_idxes | heldout_test_ind | ensemble_ind):
        raise ValueError('overlap between cross validation, ensember and heldout test')

# print the meta data of the index dictionary for the train-test split
def print_meta_info(idx_dictionary):
    cv_dictionary = idx_dictionary['cross_val']
    heldout_test_size = len(set(idx_dictionary['heldout_test_ind']))
    ensemble_size = len(set(idx_dictionary['ensemble_ind']))
    cv_train_size = len(cv_dictionary[0]['train_ind'])
    cv_val_size = len(cv_dictionary[0]['val_ind'])
    cv_test_size = len(cv_dictionary[0]['test_ind'])
    print('cross validation train set size %s.' % cv_train_size)
    print('cross validation val set size %s.' % cv_val_size)
    print('cross validation test size %s.' % cv_test_size)
    print('ensemble size %s.' % ensemble_size)
    print('heldout test size %s.' % heldout_test_size)

# "appending dict2 to dict1"
def append_dict(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]

# reading all data from the directories, regardless of where they are from
def retrieve_content(labeled_corpuses, unlabeled_corpuses, verbose):
    all_data = {}
    for corpus_dir in labeled_corpuses + unlabeled_corpuses:
        if verbose:
            print('reading data from %s ...' % corpus_dir)
        df = pd.read_csv(corpus_dir)
        records = df.to_dict('records')
        tweetid2tweet = {}
        for record in records:
            tweet_id = record['tweet_id']
            tweetid2tweet[tweet_id] = record
        append_dict(all_data, tweetid2tweet)
    return all_data

# each tweet is now a dictionary of attributes
def process_data_entries(record):
    # tokenizing at different level
    record['word_int_arr'] = int_array_rep(str(record['text']))
    record['char_int_arr'] = int_array_rep(str(record['text']), option='char')

    # time posted
    time_str = record['created_at']
    date_time_created = parser.parse(str(time_str))
    record['created_at'] = date_time_created

    # the text is deliberately deleted to ensure coherence of indexing across implementation
    del record['text']

def create_cv_idx(tr, val, idx_dictionary, fold):
    """
    Create train, val, test indexes for each fold

    Parameters
    ----------
    tr: training csv file
    val: validation csv file
    idx_dictionary: the index dictionary

    """
    idx_dictionary['cross_val'] = [{} for _ in range(fold)]

    # reading tr, val file
    df_tr, df_val = pd.read_csv(tr), pd.read_csv(val)
    tr_val_test_idx = df_tr['tweet_id'].values.tolist() + df_val['tweet_id'].values.tolist()
    num_cv_data = len(tr_val_test_idx)
    random.shuffle(tr_val_test_idx)

    # creating test index for each cross validation
    cutoffpoints = [int(p *  num_cv_data / fold) for p in range(fold + 1)]
    cutoffpoints[-1] = num_cv_data
    for fold_idx in range(fold):
        idx_dictionary['cross_val'][fold_idx]['test_ind'] = set(tr_val_test_idx[cutoffpoints[fold_idx]:cutoffpoints[fold_idx + 1]])
        fold_tr_val_idx = tr_val_test_idx[:cutoffpoints[fold_idx]] + tr_val_test_idx[cutoffpoints[fold_idx + 1]:]
        random.shuffle(fold_tr_val_idx)
        cv_tr_size = int(len(fold_tr_val_idx) * 0.8)
        idx_dictionary['cross_val'][fold_idx]['train_ind'] = set(fold_tr_val_idx[:cv_tr_size])
        idx_dictionary['cross_val'][fold_idx]['val_ind'] = set(fold_tr_val_idx[cv_tr_size:])


def create_data_idx(labeled_corpuses, fold=5):
    """
    Creating a dictionary that assigns a tweet to a fold/set
    e.g. training set in the ith cross validation, ensemble, test, etc

    Parameters
    ----------
    labeled_corpuses: directories of tr, val, held_out_test, ensemble
    """
    # unpack the array
    tr, val, held_out_test, ensemble = labeled_corpuses
    #the index dictionary
    idx_dictionary = {}

    # read the ensemble and heldout test dataframe, which is the same
    df_ensemble = pd.read_csv(ensemble)
    df_heldout_test = pd.read_csv(held_out_test)
    idx_dictionary['ensemble_ind'] = set(df_ensemble['tweet_id'].values.tolist())
    idx_dictionary['heldout_test_ind'] = set(df_heldout_test['tweet_id'].values.tolist())

    create_cv_idx(tr, val, idx_dictionary, fold)
    return idx_dictionary

def create_dataset(labeled_corpuses, unlabeled_corpuses, verbose=False):
    data_dictionary = retrieve_content(labeled_corpuses, unlabeled_corpuses, verbose)

    if verbose:
        print('processing each tweet ...')
    for key in data_dictionary:
        process_data_entries(data_dictionary[key])

    if verbose:
        print('creating indexes for train test splitting')
    idx_dictionary = create_data_idx(labeled_corpuses)

    if verbose:
        print('Checking correctness of the index dictionary')
        print_meta_info(idx_dictionary)
    assert_idx_correctness(idx_dictionary)

    data = {'data': data_dictionary, 'ind': idx_dictionary}
    if verbose:
        print('dumping data')
    pkl.dump(data, open('../data/data.pkl', 'wb'))

labeled_corpuses = ['../data/tweets_2018_03_21/tweets_2018_03_21_' + t + '.csv'
                    for t in ['tr', 'val', 'test', 'ensemble']]
unlabeled_corpuses = ['../data/ScrapedTweets_26thOct.csv', '../data/gnip_data.csv']

if __name__ == '__main__':
    create_dataset(labeled_corpuses, unlabeled_corpuses, verbose=True)
