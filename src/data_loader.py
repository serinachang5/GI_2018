"""
===================
data_loader
===================
Author: Ruiqi Zhong
Date: 04/21/2018
This module contains a class that will provide most functionalities needed for tokenizing and preprocessing
"""
import pickle as pkl
from sentence_tokenizer import int_array_rep

class Data_loader:
    """
    A Data_loader class
    Read the corresponding processed files and
    return provides data needed for classification tasks
    each datapoint/tweet is a dictionary
    """

    def __init__(self, vocab_size=40000, max_len=50, option='word', verbose=True):
        """
        Parameters
        ----------
        vocab_size: number of vocabularies to consider, including _PAD_ and _UNKNWON_
        max_len: the maximum length of a tweet
        option: the level of tokenization, "word" or "char"
        verbose: print progress while initializing
        """

        # loading vocabulary level data
        assert(option == 'word' or option == 'char') # the tokenziation level must be either at char or word
        self.option, self.vocab_size, self.max_len = option, vocab_size, max_len
        if verbose:
            print('Loading vocabulary ...')
        self.token2property = pkl.load(open('../model/' + option + '.pkl', 'rb')) # loading the preprocessed token file
        self.separator = ' ' if option == 'word' else '' # chars are not seperated, words are by spaces
        if option == 'word': # creating an id2wtoken dictionary
            self.id2token = dict([(self.token2property[word]['id'], word.decode()) for word in self.token2property])
        else:
            self.id2token = dict([(self.token2property[c]['id'], chr(c) if bytes(c) < bytes(256) else c.decode())
                                  for c in self.token2property])
        if verbose:
            print('%d vocab is considered.' % min(len(self.id2token), self.vocab_size))

        # loading user information
        self.user2property = pkl.load(open('../model/user.pkl', 'rb'))
        self.id2user = dict([(self.user2property[user_name]['id'], user_name) for user_name in self.user2property])
        if verbose:
            print('Loading user information finished')

        # loading tweet level data
        if verbose:
            print('Loading tweets ...')
        self.data = pkl.load(open('../data/data.pkl', 'rb'))
        # pad and trim the int representations of a tweet given the parameters of this Data_loader
        if verbose:
            print('Processing tweets ...')
        for tweet_id in self.data['data']:
            self.process_tweet_dictionary(self.data['data'][tweet_id])
        if verbose:
            print('Data loader initialization finishes')

    def cv_data(self, fold_idx):
        """
        Get the cross validation data. Each set is a list of dictionaries reprensenting a data point

        Parameters
        ----------
        fold_idx: the fold index of this 5-fold cross validation task

        Returns
        -------
        tr, val, test: train, val, test data for the current fold
        """
        cv_ind = self.data['classification_ind']['cross_val'][fold_idx]
        tr, val, test = (self.get_records_by_idxes(cv_ind['train_ind']),
                         self.get_records_by_idxes(cv_ind['val_ind']),
                         self.get_records_by_idxes(cv_ind['test_ind']))
        return tr, val, test

    # retrieving data for ensemble
    # similar to cv_data function
    def ensemble_data(self):
        return self.get_records_by_idxes(self.data['classification_ind']['ensemble_ind'])

    # retrieving data for testing
    # similar to cv_data function
    def test_data(self):
        return self.get_records_by_idxes(self.data['classification_ind']['heldout_test_ind'])

    # return all the data for unsupervised learning
    def all_data(self):
        result = []
        for tweet_idx in self.data['data']:
            result.append(self.data['data'][tweet_idx])
        return result

    # tokenize a string and convert it to int representation given the parameters of this data loader
    def convert2int_arr(self, s):
        int_arr = int_array_rep(str(s), option=self.option, vocab_count=self.vocab_size)
        int_arr = self.pad_int_arr(int_arr)
        return int_arr

    def tweets_by_user(self, user_id):
        """
        Given a user id, return a list of tweets posted by that user, sorted by time

        Parameters
        ----------
        user_id: the user_id of interest

        Returns
        -------
        a list of tweet dictionaries, None if that user id does not have a list
        """
        tweet_ids = self.data['user_time_ind'].get(user_id)
        if tweet_ids is None:
            return None
        return self.get_records_by_idxes(self.data['user_time_ind'][user_id])

    # convert an int array to the unicode representation
    def convert2unicode(self, int_arr):
        return self.separator.join([self.id2token[id] for id in int_arr])

    def print_recovered_tweet(self, tweet_property):
        for key in tweet_property:
            print("%s: %s" % (key, tweet_property[key]))
        print('User %s posted the tweet.' % self.id2user[tweet_property['user_post']])
        print('Users being mentioned: ' + ', '.join([self.id2user[user_id] for user_id in tweet_property['user_mentions']]))
        if tweet_property.get('user_retweet') is not None:
            print('Retweet from %s.' % self.id2user[tweet_property['user_retweet']])
        print('original tweet content: ' + self.convert2unicode(tweet_property['int_arr']))

    # get the user name of an id
    def id2user_name(self, id):
        return self.id2user[id]

    # ========== Below are the helper functions of the class ==========

    def process_tweet_dictionary(self, record):
        record['int_arr'] = record[self.option + '_int_arr']
        del record['word_int_arr']
        del record['char_int_arr']
        record['int_arr'] = self.pad_int_arr(record['int_arr'])
        self.trim2vocab_size(record['int_arr'])

    def pad_int_arr(self, int_arr):
        int_arr += [0] * self.max_len
        return int_arr[:self.max_len]

    def trim2vocab_size(self, int_arr):
        for idx in range(len(int_arr)):
            if int_arr[idx] >= self.vocab_size:
                int_arr[idx] = 1

    def get_records_by_idxes(self, idxes):
        return [self.data['data'][idx] for idx in idxes]

if __name__ == '__main__':
    dl = Data_loader(vocab_size=40000, max_len=50, option='word')
    fold_idx = 0
    tr, val, test = dl.cv_data(fold_idx)
    for idx in range(10):
        print('-------------')
        dl.print_recovered_tweet(tr[idx])

    user_tweets = dl.tweets_by_user(2)
    for idx in range(10):
        print('-------------')
        dl.print_recovered_tweet(user_tweets[idx])
