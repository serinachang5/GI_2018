"""
===================
represent_tweet_level.py
===================
Authors: Serina Chang
Date: 04/27/2018
Generate and write context embeddings.
"""
import csv
import numpy as np
from data_loader import Data_loader
from represent_tweet_level import TweetLevel

class Contextifier:
    '''
        Creates the context for tweets.
    '''
    def __init__(self, context_size=1, context_combine='avg', use_rt_user=False, 
                 use_mentions=False, use_rt_mentions=False, context_hl=1.0,
                 word_emb_file='../data/w2v_word_s300_w5_mc5_it20.bin',
                 word_emb_type='w2v',
                 splex_emb_file='../data/splex_standard_svd_word_s300_seeds_hc.pkl'
                 ):
        '''
        Create it!
        Args:
            context_size (int): Number of days to look back
            context_combine (str): Method of combining tweet embeddings of tweets in context
            use_rt_user (bool): User A retweets User B's tweet -- if true,
                    this tweet will be counted in User A and User B's context
            use_mentions (bool): User A tweets, mentioning User B -- if true, 
                    this tweet will be in User A and User B's context
            use_rt_mentions (bool): User A retweets User B's tweet, which mentioned User C -- if true,
                    this tweet will counted in User A and User C's history
            context_hl (int): Half life of context, in days. Tweet embeddings will be weighed according to
                    (self.decay_rate)^(t/context_hl) where t is the number of days the previous tweet is 
                    from the current one.
        '''
        # Save variables
        self.context_size = context_size
        self.context_combine = context_combine
        self.use_rt_user = use_rt_user
        self.use_mentions = use_mentions
        self.use_rt_mentions = use_rt_mentions
        self.context_hl = context_hl
        self.decay_rate = 0.5 # hardcoded!
        
        # Tweets in a user's "context"
        self.user_ct_tweets = {}

        # All Data
        option = 'word'
        max_len = 20
        vocab_size = 30000
        dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
        self.all_data = dl.all_data()
        
        # Map from tweet id to tuple of (user, idx in sorted list)
        # Note that "user" is user_post, the user who posted the tweet
        self.id_to_location = {}
        
        # Tweet to context embedding
        self.tweet_to_ct = {}
        
        # Cache for calculated tweet embeddings
        self.tweet_emb_cache = {}
        
        # Initializing tools to get tweet-level embeddings
        self.tl_word = TweetLevel(word_level=word_emb_file, wl_file_type=word_emb_type)
        self.tl_splex = TweetLevel(word_level=splex_emb_file, wl_file_type='pkl')

        # Hardcoding embedding size -- unsure how to change this
        self.embeddings_dim = 300 + 3
    
    
    def create_user_context_tweets(self):
        '''
        Sorts the tweets into self.user_ct_tweets, based on the variables
            self.use_rt_user, self.use_rt_mentions, and self.use_mentions
        '''
        
        # For every tweet in the dataset (labled and unlabeled)
        for tweet in self.all_data:
            incl_users = set()
            # Always include poster
            incl_users.add(tweet['user_post'])
            # Check if tweet is a retweet
            if 'user_retweet' in tweet:
                # Include retweeted user
                if self.use_rt_user:
                    incl_users.add(tweet['user_retweet'])
                # Include users mentioned in retweet
                if use_rt_mentions:
                    incl_users.union(tweet['user_mentions'])
            # Include mentioned users (non-retweet case)
            elif use_mentions:
                incl_users.union(tweet['user_mentions'])
            
            # Add tweets to users' context tweets
            for u in incl_users:
                if u in self.user_ct_tweets:
                    self.user_ct_tweets[u].append(tweet)
                else:
                    self.user_ct_tweets[u] = [tweet]
        
        # Sort context tweets chronologically
        for u in self.user_ct_tweets:
            self.user_ct_tweets[u] = sorted(self.user_ct_tweets[u], key=lambda t: t['created_at'])
            
        # Go through the tweets to save their location
        for u, tweets in self.user_ct_tweets.items():
            for idx, t in enumerate(tweets):
                if u == t['user_post']:
                    self.id_to_location[t['tweet_id']] = (u, idx)
    
    
    def get_tweet_embedding(self, tweet_id):
        '''
        Get the tweet embedding for the given tweet.
        Args:
            tweet_id (int): the id of the tweet, according to twitter's ID system
        Returns:
            the tweet embedding
        '''
        if tweet_id in self.tweet_emb_cache: # Check cache for embedding
            return self.tweet_emb_cache[tweet_id]
        else:
            w_emb = self.tl_word.get_representation(tweet_id, mode='avg')
            sp_emb =  self.tl_splex.get_representation(tweet_id, mode='avg')
            full_emb = np.concatenate([w_emb, sp_emb])
            self.tweet_emb_cache[tweet_id] = full_emb # Save embedding to cache
            return full_emb
    
    
    def create_context_embedding(self, user_id, tweet_idx):
        '''
        Get the context embedding for the given tweet, determined by user and index.
        Args:
            user_id (int): the id of the user, according to data_loader's user ids
            tweet_idx (int): the index of the tweet in self.user_ct_tweets[user_id]
        '''
        # Check if context embedding is in the cache
        tweet_id = self.user_ct_tweets[user_id][tweet_idx]['tweet_id']
        if tweet_id in self.tweet_to_ct:
            return self.tweet_to_ct[tweet_id]
        
        # Return difference in days, as a float
        def days_diff(d1, d2):
            return (d1 - d2).seconds/60/60/24
        
        tweet_embs = []
        
        today = self.user_ct_tweets[user_id][tweet_idx]['created_at']
        i = tweet_idx-1
        while i >= 0 and days_diff(today, self.user_ct_tweets[user_id][i]['created_at']) \
                                     < self.context_size:
            # Get embedding -- may need to change
            emb = self.get_tweet_embedding(self.user_ct_tweets[user_id][i]['tweet_id'])
            # Weigh embedding
            diff = days_diff(today, self.user_ct_tweets[user_id][i]['created_at'])
            weight = self.decay_rate ** (diff/self.context_hl)
            emb = emb * weight
            # Save
            tweet_embs.append(emb)
            i -= 1
        
        result = None
        if len(tweet_embs) == 0:
            result = np.zeros(self.embeddings_dim, )
        else:
            if self.context_combine == 'avg':
                result = np.mean(np.array(tweet_embs), axis=0)
            elif self.context_combine == 'sum':
                result = sum(tweet_embs)
            elif self.context_combine == 'max':
                result = np.max(np.array(tweet_embs), axis=0)
            else:
                raise ValueError('Unknown settting for context_combine:', context_combine)
        
        # Cache the result
        self.tweet_to_ct[tweet_id] = result
        return result

    def reset_context_embeddings(self):
        self.tweet_to_ct = {} # Reset embeddings
    
    
    def create_context_embeddings(self):
        '''
        Create the context embeddings for the tweets.
        '''
        for fold_idx in range(0, 5):
            tr, val, test = dl.cv_data(fold_idx)
            all_tweets = [t for l in [tr, val, test] for t in l ]
            for tweet in all_tweets: 
                self.tweet_to_ct[tweet['tweet_id']] = self.create_context_embedding(
                    *self.id_to_location[tweet['tweet_id']])
    
    
    def get_context_embedding(self, tweet_id):
        '''
        Get the context embedding for the specified tweet, determined by tweet_id
        Args:
            tweet_id (int): the id of the tweet, according to the twitter tweet ids
        Returns:
            (np.array(int)): the context embedding 
        '''
        if len(self.tweet_to_ct) == 0:
            raise ValueError('Context embeddings have not been created yet. Call create_context_embeddings().')
        if tweet_id not in self.tweet_to_ct:
            raise ValueError('No calcualted context embedding for given tweet_id:', tweet_id)
        
        return self.tweet_to_ct[tweet_id]

    
    def from_file(self, in_file):
        '''
        Reads the context embeddings in from a file.
        Args:
            in_file (str): the path to the file, in csv format, <tweet_id>, <embedding>
        Returns:
            None
        '''
        with open(in_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tweet_to_ct[int(row['tweet_id'])] = np.fromstring(row['context_embedding'],
                                                                    dtype=float, sep=' ')
        

    
    def write_context_embeddings(self, out_file=None):
        '''
        Writes the embeddings to a file.
        Args:
            out_file (str): the path of the file to write to
        Returns:
            None
        '''
        if not out_file:
            out_file = 'context_emb_{0}_{1}_rt{2}_men{3}_rtmen{4}_hl{5}.csv' \
                        .format(self.context_size, self.context_combine, self.use_rt_user, 
                                self.use_mentions, self.use_rt_mentions, self.context_hl)
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['tweet_id', 'context_embedding'])
            for tweet_id, ct_emb in self.tweet_to_ct.items():
                ct_emb_str = ' '.join([str(x) for x in ct_emb])
                writer.writerow([tweet_id, ct_emb_str])