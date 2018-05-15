from nn_experiment import Experiment
import numpy as np

if __name__ == '__main__':
    epochs, patience = 1, 1
    
    num_users = 3000 # number of users to be considered
    user_embed_dim = 300 # dimension of user embedding
    user_embed_dropout = 0.5 # drop out for the user embedding drop out layer
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int') # as usual, all ids of the labeled tweets
    pretrained_weight_dirs = ({
                              # word embedding can be loaded here
                              'aggression_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              'loss_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              # user embedding can be loaded here
                              # must be of the size num_users * user_embed_dim
                              # if not loading weights from other directory
                              # then randomly initialized
                              # user embedding is trainable
                              # if you want fixed user embedding, we shall discuss this further
                              'aggression_user_embed': ['../weights/dummy_user_embed.np'],
                              'loss_user_embed': ['../weights/dummy_user_embed.np']
                              })
    input_name2id2np = ({
                        'dummy': dict([(tid, np.random.normal(size=(300,))) for tid in labeled_tids]),
                        # note that I made special treatment towards user embeddings
                        # the input name of user index must end with "user_index"
                        # or other spelling variations, see input_name_is_user_idx function in model_def.py
                        # each id should map to an int value, not an array
                        # it will automaticlly be transformed into an array in my script
                        # since keras does not accept single integer as input
                        # but it is already handled on my side so you don't need to worry about that
                        'post_user_index': dict([(tid, np.array([np.random.randint(3000)])) for tid in labeled_tids]),
                        'retweet_user_index': dict([(tid, np.array([np.random.randint(3000)])) for tid in labeled_tids])
                        })
    options = ['word']
    experiment = Experiment(experiment_dir='test',
                            pretrained_weight_dirs=pretrained_weight_dirs,
                            input_name2id2np=input_name2id2np,
                            options=options,
                            
                            # arguments for user embeddings
                            num_users=num_users, user_embed_dim=user_embed_dim,
                            user_embed_dropout= user_embed_dropout,
                            
                            # the interaction layer of after what is previously the last layer
                            # if dim = -1 (default) then there is no interaction layer
                            # interaction_layer_drop_out: dropout for the interaction layer
                            interaction_layer_dim=100, interaction_layer_drop_out=0.5)
    experiment.cv()

    """
        TODO: note that all of the new parameters probably need to be tuned
        All of these default values are just preliminary
    """
