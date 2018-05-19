from nn_experiment import Experiment
import pickle


# randomly initialized
def baseline():
    options = ['word']
    experiment = Experiment(experiment_dir='baseline',
                            pretrained_weight_dirs=None,
                            input_name2id2np=None,
                            options=options)
    experiment.cv()


# use list of modes to specify how to integrate word2vec and splex:
    # 'wl' word-level - word embedding
    # 'tl' tweet-level - built by TweetLevel, concatenated after CNN
    # 'cl' context-level - built by Contextifier, concatenated after CNN
# use list of modes to specify which user embeddings to include (concatenated after CNN)
    # 'po' user who posted the tweet
    # 'rt' user retweeted in the post
    # 'men' user mentioned in the post
# int_dim and int_dropout are for interaction layer
# run_num indicates which run we are on
# if testing, this just checks the set-up, running only 1 epoch and patience = 1
def nn_with_additions(w2v_modes, splex_modes, user_modes, include_time,
                      user_rand = True, user_emb_dim = 32,
                      int_dim=-1, int_dropout=.5,
                      run_num = None, testing=False):
    assert(user_emb_dim == 300 or user_emb_dim == 32)
    if len(w2v_modes) == 0 and splex_modes == 0:
        baseline()
        return

    dir_name = ''
    if len(w2v_modes) > 0:
        dir_name += 'W2V_' + '_'.join(w2v_modes)
    if len(splex_modes) > 0:
        dir_name += 'SP_' + '_'.join(splex_modes)
    if len(user_modes) > 0:
        if user_rand:
            dir_name += 'USR'
        else:
            dir_name += 'USP'
        dir_name += '_'.join(user_modes)
    if include_time:
        dir_name += 'TI'
    if int_dim > 0:
        dir_name += 'INT_' + str(int_dim) + '_' + str(int_dropout)
    if run_num is not None:
        dir_name += '#' + str(run_num)
    print('Will save experiment in', dir_name)

    # set word embedding
    if 'wl' in w2v_modes and 'wl' in splex_modes:
        word_emb = 'word_emb_w2v_splex.np'
        word_emb_dim = 302
    elif 'wl' in w2v_modes:
        word_emb = 'word_emb_w2v.np'
        word_emb_dim = 300
    elif 'wl' in splex_modes:
        word_emb = 'word_emb_splex.np'
        word_emb_dim = 2
    else:
        word_emb = None
        word_emb_dim = None

    # set user embedding
    if len(user_modes) > 0:
        num_users = 700
        if user_rand:
            if user_emb_dim == 300:
                user_emb_agg = 'user_emb_700_agg_rand_300.np'
                user_emb_loss = 'user_emb_700_loss_rand_300.np'
            elif:
                user_emb_agg = 'user_emb_700_agg_rand_32.np'
                user_emb_loss = 'user_emb_700_loss_rand_32.np'
        else:
            user_emb_agg = 'user_emb_700_w2v.np'
            user_emb_loss = 'user_emb_700_w2v.np'
    else:
        user_emb_agg = None
        user_emb_loss = None
        num_users = None

    # specify embeddings in pretrained
    pretrained = {}
    if word_emb is not None:
        pretrained['aggression_word_embed'] = [word_emb]
        pretrained['loss_word_embed'] = [word_emb]
    if user_emb_agg is not None:
        pretrained['aggression_user_embed'] = [user_emb_agg]
        pretrained['loss_user_embed'] = [user_emb_loss]

    # include as concatenated feature
    input_name2id2np = {}
    all_inputs = pickle.load(open('all_inputs.pkl', 'rb'))
    if 'tl' in splex_modes:
        input_name2id2np['splex_tl'] = all_inputs['splex_tl']
    if 'cl' in w2v_modes:
        input_name2id2np['w2v_cl'] = all_inputs['w2v_cl']
    if 'cl' in splex_modes:
        input_name2id2np['splex_cl'] = all_inputs['splex_cl']
    if 'po' in user_modes:
        input_name2id2np['post_user_index'] = all_inputs['post_user_index']
    if 'rt' in user_modes:
        input_name2id2np['retweet_user_index'] = all_inputs['retweet_user_index']
    if 'men' in user_modes:
        input_name2id2np['mention_user_index'] = all_inputs['mention_user_index']
    if include_time:
        input_name2id2np['time'] = all_inputs['time']

    options = ['word']

    if testing:
        experiment = Experiment(experiment_dir=dir_name,
                                pretrained_weight_dirs=pretrained,
                                input_name2id2np=input_name2id2np,
                                epochs=1, patience=1,
                                options=options,
                                embed_dim=word_emb_dim,
                                num_users=num_users, user_embed_dim=user_emb_dim,
                                interaction_layer_dim=int_dim,
                                interaction_layer_drop_out=int_dropout)
    else:
        experiment = Experiment(experiment_dir=dir_name,
                                pretrained_weight_dirs=pretrained,
                                input_name2id2np=input_name2id2np,
                                patience=7,
                                options=options,
                                embed_dim=word_emb_dim,
                                num_users=num_users, user_embed_dim=user_emb_dim,
                                interaction_layer_dim=int_dim,
                                interaction_layer_drop_out=int_dropout)
    experiment.cv()


if __name__ == '__main__':
    interaction_dim = -1  # edit later
    interaction_dropout = .5   # edit later

    # # base
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl'], splex_modes=['tl'], user_modes=[], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)

    # # w2v as context
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl','cl'], splex_modes=['tl'], user_modes=[], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)

    # # splex as context
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl'], splex_modes=['tl','cl'], user_modes=[], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)

    # w2v and splex as context
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl','cl'], splex_modes=['tl','cl'], user_modes=[], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)

    # random user embeddings as context
    for num in range(1,2):
        nn_with_additions(w2v_modes=['wl'], splex_modes=['tl'], user_modes=['po'], include_time=False,
                          user_rand=True, user_emb_dim=32,
                          int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl'], splex_modes=['tl'], user_modes=['po', 'rt'], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)
    # for num in range(1,6):
    #     nn_with_additions(w2v_modes=['wl'], splex_modes=['tl'], user_modes=['po', 'men'], include_time=False,
    #                       int_dim=interaction_dim, int_dropout=interaction_dropout, run_num=num)
