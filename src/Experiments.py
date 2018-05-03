from nn_experiment import Experiment
import numpy as np

labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')

# this is the variable for incorporating splex and context
# it is a map from component_name to id2np
# where each id2np is a map from tweet id to an numpy array
input_name2id2np = {'dummy_splex': # compoenent name
                    dict([(tid, np.zeros(20, )) for tid in labeled_tids]),
                    # the id2np map should contain all the labeled tweetid
                    # suppose 20 is the dimension for splex scores
                    'context_representation':
                    dict([(tid, np.zeros(320, )) for tid in labeled_tids])}
options = ['word']
experiment = Experiment(experiment_dir='test', input_name2id2np=input_name2id2np, adapt_train_vocab=True,
                            options=options)
experiment.cv()