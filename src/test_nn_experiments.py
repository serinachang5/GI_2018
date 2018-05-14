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
    '''
        with word2vec embeddings
        pretrained_weight_dirs is a dictionary, where each key is the layer name
        and the value is a list of weight directories in numpy file format
        
        Currently, since the program trains models for both aggression and loss class
        Except the input layer, all the other layers will now have a "class" prefix
        Therefore, if one weight can be used by two models, you need to initialize both of them
        
        About the keys: each of them should match a layer name in the model
        After a weight of a layer is loaded, the program will print a message that weight loaded successfully
        (after model.summary() shows the architecture)
        
        About the weights:
        Each of them is an array of numpy directories that can be called by np.loadtxt
        The list of numpy array should be exactlyt the same as layer.get_weights
        (see keras documetentation for details)
        In particular, word embedding layer has weight: [<a matrix of size vocab * embed_dim>]
    '''
    options = ['word']
    # note that each of the key has appeared as a layer name in the model
    pretrained_weight_dirs = ({'aggression_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              'loss_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np']})
    experiment = Experiment(experiment_dir='test_word2vec_' + str(_), input_name2id2np=None,
                            pretrained_weight_dirs=pretrained_weight_dirs, options=options)
    experiment.cv()

    options = ['word', 'char']
    # combining word, char and other inputs
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    input_name2id2np = {'dummpy': dict([(tid, np.random.normal(size=(300,)))
                                        for tid in labeled_tids])}
    experiment = Experiment(experiment_dir='test_combine',
                            input_name2id2np=input_name2id2np, adapt_train_vocab=True,
                            options=options)
    experiment.cv()
