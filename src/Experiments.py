from nn_experiment import Experiment
from generator_util import simplest_tweet2data
import pickle as pkl
'''
# experiment that is only on the word level
# without pretraining
options = ['word']
experiment = Experiment(tweet2data=simplest_tweet2data, experiment_dir='nonpretrained_word',
                        comments='Before the update on word level, no pretraining', adapt_train_vocab=True,
                        options=options)
experiment.cv()
'''

'''
# experiment that is only on the char level
# without pretraining
options = ['char']
experiment = Experiment(tweet2data=simplest_tweet2data, experiment_dir='nonpretrained_word',
                        comments='Before the update on word level, no pretraining', adapt_train_vocab=True,
                        options=options)
experiment.cv()
'''

id2rep = pkl.load(open('id2rep.pkl', 'rb'))

def tweet_splex(tweet_dict):
    r = simplest_tweet2data(tweet_dict)
    r['context_input'] = id2rep[tweet_dict['tweet_id']]
    return r

options = ['word', 'context']
experiment = Experiment(tweet2data=tweet_splex, experiment_dir='toy_context',
                        comments='toy context, testing whether code runs with context', adapt_train_vocab=True,
                        options=options, context_dim=3, context_dense_size=32)
experiment.cv()
