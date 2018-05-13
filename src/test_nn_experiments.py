from nn_experiment import Experiment
import numpy as np

if __name__ == '__main__':
    # plain words CNN, without pretraining
    options = ['word']
    experiment = Experiment(experiment_dir='test_word', input_name2id2np=None, adapt_train_vocab=True,
                            options=options)
    experiment.cv()

    # plain char level CNN, without pretraining
    options = ['char']
    experiment = Experiment(experiment_dir='test_char', input_name2id2np=None, adapt_train_vocab=True,
                            options=options)
    experiment.cv()

    # with word2vec embeddings
    pretrained_weight_dirs = ({'aggression_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              'loss_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np']})
    experiment = Experiment(experiment_dir='test_word2vec_' + str(_), input_name2id2np=None,
                            pretrained_weight_dirs=pretrained_weight_dirs, options=options)
    experiment.cv()

    # combining word, char and other inputs
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    input_name2id2np = {'dummpy': dict([(tid, np.random.normal(shape=(300,)))
                                        for tid in labeled_tids])}
    experiment = Experiment(experiment_dir='test_combine_',
                            input_name2id2np=input_name2id2np, adapt_train_vocab=True,
                            options=options)
