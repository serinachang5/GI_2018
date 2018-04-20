"""
===================
create_vocab
===================
Author: Ruiqi Zhong
Date: 04/20/2018
This module implements a function that would take in the labeled and unlabeld corpus
and create a word dictionary in json
"""
from preprocess import preprocess, isemoji, to_char_array
import pandas as pd
import pickle as pkl

# count the number of ocurrences of char and word in a list of data directory
def count_occurrence(corpus_dirs):
    corpus_word_count, corpus_char_count = {}, {}
    tweet_id_read = set()
    for corpus_dir in corpus_dirs:
        df = pd.read_csv(corpus_dir)
        texts = df['text'].values
        tweet_ids = df['tweet_id'].values
        for idx in range(len(texts)):
            # avoid reading the same tweet from two files twice
            text, tweet_id = texts[idx], tweet_ids[idx]
            if tweet_id in tweet_id_read:
                continue
            else:
                tweet_id_read.add(tweet_id)

            # preprocess the text
            text_bytes = preprocess(str(text))
            char_array = to_char_array(text_bytes)

            for word in text_bytes.split(b' '):
                if corpus_word_count.get(word) is None:
                    corpus_word_count[word] = 0
                corpus_word_count[word] += 1

            for c in char_array:
                if corpus_char_count.get(c) is None:
                    corpus_char_count[c] = 0
                corpus_char_count[c] += 1
    return corpus_word_count, corpus_char_count


# set the count to 0 if one entry is in another dictionary
def merge_dict(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            dict2[key] = 0

    for key in dict2:
        if key not in dict1:
            dict1[key] = 0

# each word is mapped to a dictionary that describes its property
def new_property_dict(id):
    return {'id': id}


def get_token_properties(labeled_corpus_token_count, unlabeled_corpus_token_count):
    merge_dict(labeled_corpus_token_count, unlabeled_corpus_token_count)
    token2property = {b'_PAD_': new_property_dict(0), b'_UNKNOWN_': new_property_dict(1)}
    offset = len(token2property)
    # only consider tokens that occur more than once
    # ranked tokens first by number of occurence in labeled corpus
    # then in unlabeled corpus
    token_rank = sorted([w for w in labeled_corpus_token_count if
                         labeled_corpus_token_count[w] + unlabeled_corpus_token_count[w] >= 2],
                        key=lambda w: (-labeled_corpus_token_count[w],
                                       -unlabeled_corpus_token_count[w]))
    for idx in range(len(token_rank)):
        w = token_rank[idx]
        token2property[w] = new_property_dict(idx + offset)
        token2property[w]['occurence_in_labeled'] = labeled_corpus_token_count[w]
        token2property[w]['occurence_in_unlabeled'] = unlabeled_corpus_token_count[w]
        token2property[w]['isemoji'] = isemoji(w)

    for special_token in [b'_PAD_', b'_UNKNOWN_']:
        token2property[special_token]['isemoji'] = False
        token2property[special_token]['occurence_in_labeled'] = 0
        token2property[special_token]['occurence_in_unlabeled'] = 0

    return token2property

def create_vocab(labeled_corpuses, unlabeled_corpuses, word_file_dir, char_file_dir, verbose=False):
    """
    A function that takes in labeled and unlabeld corpuses and create vocabulary-index lookup

    Parameters
    ----------
    labeled_corpuses: a list of labeled corpus file directory in csv format
    unlabeled_corpuses: a list of unlabeled corpus file directory in csv format
    word_file_dir: output directory of the vocab-index lookup
    char_file_dir: output directory of the char-index lookup

    Returns
    -------
    word_dict: a word-dictionary that contains the information of a vocab
    char_dict: a char-dictionary that contains the information of a char
    """

    # counting the token occurrences in the labeled and unlabeled corpus
    if verbose:
        print('reading labeled corpus')
    labeled_corpus_word_count, labeled_corpus_char_count = count_occurrence(labeled_corpuses)

    if verbose:
        print('reading unlabeled corpus')
    unlabeled_corpus_word_count, unlabeled_corpus_char_count = count_occurrence(unlabeled_corpuses)

    # based on the word count, create a property dictionary for each word
    if verbose:
        print('calculating properties for words')
    word2property = get_token_properties(labeled_corpus_word_count, unlabeled_corpus_word_count)

    if verbose:
        print('calculating properties for characters')
    char2property = get_token_properties(labeled_corpus_char_count, unlabeled_corpus_char_count)

    with open(word_file_dir, 'wb') as out_word_file:
        pkl.dump(word2property, out_word_file)

    with open(char_file_dir, 'wb') as out_char_file:
        pkl.dump(char2property, out_char_file)

if __name__ == '__main__':
    labeled_corpuses = ['../data/tweets_2018_03_21/tweets_2018_03_21_' + t + '.csv'
                        for t in ['tr', 'val', 'test', 'ensemble']]
    unlabeled_corpuses = ['../data/ScrapedTweets_26thOct.csv', '../data/gnip_data.csv']
    word_file_dir, char_file_dir = ['../model/' + s + '.pkl' for s in ['word', 'char']]
    create_vocab(labeled_corpuses, unlabeled_corpuses, word_file_dir, char_file_dir, verbose=True)
