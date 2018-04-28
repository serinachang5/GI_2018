"""
===================
represent_tweet_level.py
===================
Authors: Ethan Adams
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
                 word_emb_mode='avg',
                 splex_emb_file='../data/splex_standard_svd_word_s300_seeds_hc.pkl',
                 splex_emb_mode='sum',
                 keep_stats=False
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
                    from the current one. Set to 0 for no weighting/decay.
            word_emb_file (str): the path to the file to saved word embeddings
            word_emb_file (str): the type of the word embedding file, e.g. 'w2v'. See TweetLevel for more info.
            word_emb_mode (str): the mode to use when combining word embeddings at TweetLevel, e.g. 'avg'
            splex_emb_file (str): the pickle file that contains the splex embeddings.
            splex_emb_mode (str): the mode to use when combining splex scores at TweetLevel, e.g. 'sum'
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
        self.dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
        self.all_data = self.dl.all_data()
        
        # Map from tweet id to tuple of (user, idx in sorted list)
        # Note that "user" is user_post, the user who posted the tweet
        self.id_to_location = {}
        
        # Tweet to context embedding
        self.tweet_to_ct = {}
        

        # Cache for combined tweet-level embeddings
        self.tweet_emb_cache = {}

        # Cache for calculated tweet-level word embeddings
        self.tweet_word_cache = {}
        # Cache for calculated tweet-level splex embeddings
        self.tweet_splex_cache = {}
        
        # Initializing tools to get tweet-level embeddings
        self.tl_word = TweetLevel(word_level=word_emb_file, wl_file_type=word_emb_type)
        self.word_emb_mode = word_emb_mode
        self.tl_splex = TweetLevel(word_level=splex_emb_file, wl_file_type='pkl')
        self.splex_emb_mode = splex_emb_mode

        # Hardcoding embedding size -- unsure how to change this
        self.embeddings_dim = 300 + 3

        # Keeping stats
        self.keep_stats = keep_stats
        if self.keep_stats:
            # Tweet id to tweet ids in context window
            self.tweet_to_ct_tweets = {}

    
    
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
                if self.use_rt_mentions:
                    incl_users.union(tweet['user_mentions'])
            # Include mentioned users (non-retweet case)
            elif self.use_mentions:
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
            w_emb = self.tl_word.get_representation(tweet_id, mode=self.word_emb_mode)
            sp_emb =  self.tl_splex.get_representation(tweet_id, mode=self.splex_emb_mode)
            full_emb = np.concatenate([w_emb, sp_emb])
            self.tweet_emb_cache[tweet_id] = full_emb # Save embedding to cache
            return full_emb

    def get_word_embedding(self, tweet_id):
        # add cache back in here
        if tweet_id in self.tweet_word_cache:
            return self.tweet_word_cache[tweet_id]
        else:
            res = self.tl_word.get_representation(tweet_id, mode=self.word_emb_mode)
            self.tweet_word_cache[tweet_id] = res
            return res

    def get_splex_embedding(self, tweet_id):
        # add cache back in here
        if tweet_id in self.tweet_splex_cache:
            return self.tweet_splex_cache[tweet_id]
        else:
            res = self.tl_splex.get_representation(tweet_id, mode=self.splex_emb_mode)
            self.tweet_splex_cache[tweet_id] = res
            return res


    def combine_embeddings(self, embeddings, mode):
        # documentation
        result = None
        if mode == 'avg':
            result = np.mean(np.array(embeddings), axis=0)
        elif mode == 'sum':
                result = sum(embeddings)
        elif mode == 'max':
            result = np.max(np.array(embeddings), axis=0)
        else:
            raise ValueError('Unknown combination method:', mode)
        return result
    
    
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
            return (d1 - d2).total_seconds() / 60 / 60 / 24
        
        w_embs = []
        splex_embs = []
        tweet_ids = [] # for stats
        
        today = self.user_ct_tweets[user_id][tweet_idx]['created_at']
        i = tweet_idx-1
        while i >= 0 and days_diff(today, self.user_ct_tweets[user_id][i]['created_at']) \
                                     < self.context_size:

            # Save tweet ids
            if self.keep_stats:
                tweet_ids.append(self.user_ct_tweets[user_id][i]['tweet_id'])

            # Get embeddings -- may need to change
            w_emb = self.get_word_embedding(self.user_ct_tweets[user_id][i]['tweet_id'])
            splex_emb = self.get_splex_embedding(self.user_ct_tweets[user_id][i]['tweet_id'])

            # Weigh embedding
            if self.context_hl != 0:
                diff = days_diff(today, self.user_ct_tweets[user_id][i]['created_at'])
                weight = self.decay_rate ** (diff/self.context_hl)
                w_emb = w_emb * weight
                splex_emb = splex_emb * weight

            # Save
            w_embs.append(w_emb)
            splex_embs.append(splex_emb)
            i -= 1

        # Save stats
        if self.keep_stats:
            self.tweet_to_ct_tweets[tweet_id] = tweet_ids
        
        # Combine word embeddings
        w_comb = None
        if len(w_embs) == 0:
            w_comb = np.zeros(300, ) #AH! i don't have to hardcode these now
        else:
            w_comb = self.combine_embeddings(w_embs, self.word_emb_mode)

        # Combine splex embeddings
        splex_comb = None
        if len(splex_embs) == 0:
            splex_comb = np.zeros(3, ) # still hardcoded
        else:
            splex_comb = self.combine_embeddings(splex_embs, self.splex_emb_mode)    
        
        # Concatenate to get result
        result = np.concatenate([w_comb, splex_comb])

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
            tr, val, test = self.dl.cv_data(fold_idx)
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
        if len(self.user_ct_tweets) == 0:
            raise ValueError('User contexts have not been created. First call .create_user_context_tweets().')
        if tweet_id in self.tweet_to_ct:
            return self.tweet_to_ct[tweet_id]
        else:
            # note: some weirdness going on here with loading from files
            return self.create_context_embedding(*self.id_to_location[tweet_id])


    def get_context_tweets(self, tweet_id):
        # return ids of tweets in context
        if tweet_id in self.tweet_to_ct_tweets:
            return self.tweet_to_ct_tweets[tweet_id]
        else:
            raise ValueError('no calculated tweet ids in context') # fix this

    
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
            out_file = 'context_emb_{0}_{1}_rt{2}_men{3}_rtmen{4}_hl{5}_.csv' \
                        .format(self.context_size, self.context_combine, self.use_rt_user, 
                                self.use_mentions, self.use_rt_mentions, self.context_hl)
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['tweet_id', 'context_embedding'])
            for tweet_id, ct_emb in self.tweet_to_ct.items():
                ct_emb_str = ' '.join([str(x) for x in ct_emb])
                writer.writerow([tweet_id, ct_emb_str])



if __name__ == '__main__':

    # Tester/usage
    context_size = 2
    context_combine = 'avg' 
    use_rt_user = True
    use_mentions = True
    use_rt_mentions = True
    context_hl = 2
    word_emb_file='../data/w2v_word_s300_w5_mc5_it20.bin'
    word_emb_type='w2v'
    splex_emb_file='../data/splex_standard_svd_word_s300_seeds_hc.pkl'
    contextifier = Contextifier(context_size, context_combine, use_rt_user, use_mentions,
         use_rt_mentions, context_hl, word_emb_file, word_emb_type, splex_emb_file)

    print('Creating user contexts...')
    contextifier.create_user_context_tweets()

    # Only necessary if you want to write them all to a file.
    # Can be done "on-demand" with .get_context_embedding()
    print('Creating context embeddings...')
    contextifier.create_context_embeddings()

    print('Writing context embeddings...')
    contextifier.write_context_embeddings()

    # Alternatively, to load from a file, do:
    # contextifier.from_file('../data/'context_emb_5_avg_rtFalse_menTrue_rtmenFalse_hl1.0_.csv')